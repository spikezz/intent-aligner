"""Everything about a call that the code fixes, read from the code's own constants where one exists."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))
from intent_aligner import compress as Z
from intent_aligner import extract as X
from intent_aligner import llm as M
from intent_aligner import segment as S
from intent_aligner import window as W


def slots(text):
    """Split a template into fixed text and {{variable}} slots."""
    parts = re.split(r"(\{\{.*?\}\})", text, flags=re.S)
    return [["v", p[2:-2]] if p.startswith("{{") else ["f", p] for p in parts if p]


RELATIONS = json.dumps(
    [{"relation": n, "from": a, "to": b, "meaning": m} for n, (a, b, m) in X._C_RELATIONS.items()],
    ensure_ascii=False,
    indent=2,
)
UNIT_FORMAT = {
    "user_segment": "【用户】\n{{这条消息在本段之前的文字}}<当前段落>{{本段文字}}</当前段落>{{这条消息在本段之后的文字}}{{每张图片一行：换行 + [图片] 路径}}",
    "tool_call": "【AI 的工具调用】\n{{工具名}} 意图：{{第一级压缩的意图}} 对象：{{第一级压缩的 targets}}",
    "tool_result": "【工具结果】\n{{第一级压缩的概括}}",
    "user_signal": "【用户信号】\n{{信号原文}}",
    "task_notification": "【任务通知】\n{{通知原文}}",
    "system": "【系统】\n{{系统消息原文}}",
}
ACTOR = {
    "user_segment": "user",
    "user_signal": "user",
    "tool_call": "ai",
    "observation": "environment",
}


def head(unit_kind, extra):
    chars = '\n    "chars": {{字符范围 [起点, 终点]}}' if unit_kind == "user_segment" else ""
    comma = "," if chars else ""
    body = (
        '<header>\n{\n  "topics": {{已登记的 topic 列表，每项含 id 和 title}},\n  "unit": {\n'
        f'    "kind": "{unit_kind}",\n    "actor": "{ACTOR[unit_kind]}",\n'
        '    "session": {{会话编号}},\n    "timestamp": {{时间戳}}' + comma + chars + "\n  }"
    )
    return body + "".join(f",\n  {line}" for line in extra) + "\n}\n</header>"


GRAPH = ['"current_topic": {{当前 topic 编号}}', '"graph": {{当前 topic 的整张图谱：全部节点和边}}']
WINDOW = "\n\n<此前的对话>\n{{窗口内容，条目格式见下方列表}}\n</此前的对话>\n\n<当前单元>\n"
LISTED = "\n</当前单元>\n\n<本单元的意图>\n{{每个意图一行：序号. 意图文本（依据：evidence）}}\n</本单元的意图>"

COUNT = "answers 有 {n} 项，但 <本单元的意图> 有 {m} 个，要按顺序给每个意图回答一项。"
FEEDBACK = {
    "extract": [
        ["回答不是要求的 JSON", "\n\n你上一次的回答不是要求的 JSON，请重新回答：{pydantic 错误}"],
        ["校验规则不通过", "\n\n你上一次的回答有问题，请重新回答：{下面这条规则的错误文字}"],
        ["回答被截断", "\n\n" + M.TRUNCATED_FEEDBACK],
    ],
    "compress": [
        ["回答不合格", "\n\n你上一次的回答不符合要求，请重新回答：{pydantic 错误}"],
        ["回答被截断", "\n\n" + M.TRUNCATED_FEEDBACK],
    ],
    "segment": [
        ["所有反馈的外层", "\n\n你上一次的回答有问题，请重新回答。\n{下面的反馈文字}"],
        ["回答不合格", "回答不符合要求：{pydantic 错误}"],
        ["回答被截断", M.TRUNCATED_FEEDBACK],
    ],
}
EVIDENCE_TAIL = "evidence 被 … 分开的每个片段都必须原样出现在{where}，并且按原文的先后顺序排列。"
NOTE = [
    "explicit 为 false 时 note 必须写推断依据",
    "意图 {意图文本!r} 的 explicit 是 false，note 必须写推断依据",
]


def spec(system, schema, template, checks, limit, family="extract"):
    return {
        "system": system,
        "schema": schema,
        "user": slots(template),
        "checks": checks,
        "limit": limit,
        "feedback": FEEDBACK[family],
    }


def call_spec(call, unit_key="", kind=""):
    """The fixed content of one call; `unit_key` picks how the current unit is rendered."""
    unit = UNIT_FORMAT.get(unit_key, "")
    n = X.CALL_ATTEMPTS
    if call == "压缩·调用":
        return spec(
            Z.CALL_SYSTEM_PROMPT,
            Z.CALL_SCHEMA,
            "<调用>\n{{这次工具调用的完整原文}}\n</调用>",
            [
                [
                    "每条 evidence 都是 <调用> 原文的摘录",
                    "这些 evidence 不是 <调用> 原文的摘录：{strays!r}。"
                    + EVIDENCE_TAIL.format(where=" <调用> 里"),
                ],
                NOTE,
            ],
            Z.MAX_ATTEMPTS,
            "compress",
        )
    if call == "压缩·结果":
        return spec(
            Z.RESULT_SYSTEM_PROMPT,
            Z.SUMMARY_SCHEMA,
            "<调用>\n意图：{{第一级压缩的意图}}\n原文：\n{{这次调用的完整原文}}\n</调用>\n\n<结果>{{结果太长被分块时：（第 i/n 段）}}\n{{返回内容的原文或其中一块}}\n</结果>",
            [],
            Z.MAX_ATTEMPTS,
            "compress",
        )
    if call == "压缩·整轮":
        return spec(
            Z.ROUND_SYSTEM_PROMPT,
            Z.SUMMARY_SCHEMA,
            "<用户消息>{{AI 工作时发出的消息加：（AI 工作时发出）；超过 2,000 token 加：（太长，只保留开头）}}\n{{这一轮开头的用户消息}}\n</用户消息>\n<轨迹>{{轨迹太长被分块时：（第 i/n 段）}}\n{{这一轮每个调用和结果的压缩，一行一个}}\n</轨迹>",
            [],
            Z.MAX_ATTEMPTS,
            "compress",
        )
    if call == "切分":
        return spec(
            S.SYSTEM_PROMPT,
            S.SCHEMA,
            "<此前的对话>\n{{窗口内容}}\n</此前的对话>\n\n<已处理>\n{{消息里指针之前的文字}}\n</已处理>\n\n<剩余>\n{{消息里指针之后的文字}}\n</剩余>{{有图片时：\n\n这条消息附带的图片（只有路径，内容不可见）：\n路径列表}}",
            [
                [
                    "segment_end 在 <剩余> 里找得到",
                    "你给的 segment_end 是 {segment_end!r}。segment_end 在 <剩余> 里找不到。它必须是 <剩余> 里逐字照抄的一段文本。",
                ],
                [
                    "segment_end 在 <剩余> 里只出现一次",
                    "你给的 segment_end 是 {segment_end!r}。segment_end 在 <剩余> 里出现了 {n} 次，无法确定是哪一处：\n- …{上下文}…\n"
                    + S.FROM_START_RULE,
                ],
                [
                    "改成从头抄之后，<剩余> 以它开头",
                    "你给的 segment_end 是 {segment_end!r}。<剩余> 不是以你给的 segment_end 开头。<剩余> 的开头是：{开头文字!r}\n"
                    + S.FROM_START_RULE,
                ],
                [
                    "每条 evidence 都是这一段的摘录",
                    "这些 evidence 不是这一段的原文摘录：{strays!r}。这一段按你给的 segment_end 定位为：{text!r}。"
                    + EVIDENCE_TAIL.format(where="这一段里"),
                ],
                NOTE,
            ],
            S.MAX_ATTEMPTS,
            "segment",
        )
    if call == "T":
        extra = (
            ['"round_topic_hint": {{这一轮上一个工具调用的 topic，没有时这一行不出现}}']
            if kind == "tool_call"
            else []
        )
        return spec(
            X.T_SYSTEM_PROMPT,
            X.T_SCHEMA,
            head(kind, extra) + WINDOW + unit + "\n</当前单元>",
            [
                ["回答 NEW 时标题非空", "topic 为 NEW 时 title 不能为空。"],
                [
                    "编号在 topics 列表里",
                    "{topic!r} 不在 topics 列表里，只能回答列表里的编号或 NEW。",
                ],
            ],
            X.T_ATTEMPTS,
        )
    if call == "A":
        return spec(
            X.A_SYSTEM_PROMPT,
            X.A_SCHEMA,
            head(kind, GRAPH) + WINDOW + unit + "\n</当前单元>",
            [
                [
                    "每条 evidence 都是单元原文的摘录",
                    "这些 evidence 不是 <当前单元> 原文的摘录：{strays!r}。"
                    + EVIDENCE_TAIL.format(where="原文里"),
                ],
                NOTE,
            ],
            n,
        )
    candidates = '"candidates": {{%s}}'
    if call == "B":
        return spec(
            X.B_SYSTEM_PROMPT,
            X.B_SCHEMA,
            head(kind, [*GRAPH, candidates % "当前 topic 的用户意图编号"]) + WINDOW + unit + LISTED,
            [
                ["答案和本单元的意图一一对应", COUNT],
                [
                    "选的编号在候选里或是 NEW",
                    "{wrong!r} 不在 candidates 里，match 只能是 candidates 里的编号或 NEW。",
                ],
            ],
            n,
        )
    if call == "G":
        tail = (
            LISTED + f"\n\n<此后的对话>\n{{{{当前单元之后最多 {W.FOLLOWING_ITEMS} 个条目，"
            f"最多 {W.FOLLOWING_BUDGET:,} token}}}}\n</此后的对话>"
        )
        return spec(
            X.G_SYSTEM_PROMPT,
            X.G_SCHEMA,
            head(kind, [*GRAPH, candidates % "当前 topic 的聚合编号"]) + WINDOW + unit + tail,
            [
                ["答案和本单元的意图一一对应", COUNT],
                [
                    "选的编号在候选里或是 NEW",
                    "{wrong!r} 不在 candidates 里，aggregate 只能是 candidates 里的编号或 NEW。",
                ],
                ["NEW 时 title 非空", "aggregate 为 NEW 时 title 不能为空。"],
            ],
            n,
        )
    if call == "L":
        tail = LISTED + "\n\n<已有的底层意图>\n{{每行：L 编号：底层意图文本}}\n</已有的底层意图>"
        return spec(
            X.L_SYSTEM_PROMPT,
            X.L_SCHEMA,
            head(kind, [*GRAPH, candidates % "这个聚合的 L 成员编号"]) + WINDOW + unit + tail,
            [
                ["答案和这个聚合的意图一一对应", COUNT],
                [
                    "选的编号在候选里或是 NEW",
                    "{wrong!r} 不在 candidates 里，match 只能是 candidates 里的编号或 NEW。",
                ],
            ],
            n,
        )
    if call == "C":
        extra = [
            *GRAPH,
            candidates
            % "当前 topic 全部节点，加窗口内看得到的未建节点观测（每项含 id：P序号 和 text）",
            '"relations": ' + RELATIONS.replace("\n", "\n  "),
            '"unit_intents": {{本单元的意图节点}}',
        ]
        return spec(
            X.C_SYSTEM_PROMPT,
            X.C_SCHEMA,
            head(kind, extra) + WINDOW + unit + "\n</当前单元>",
            [
                [
                    "两端都在 candidates 或 unit_intents 里",
                    "{wrong!r} 不在 candidates 或 unit_intents 里。",
                ],
                [
                    "两端不同，且有一端在 unit_intents 里",
                    "{ends!r} 的两端必须不同，并且有一端在 unit_intents 里。",
                ],
                ["方向和节点类型匹配", "{relation} 的方向必须是 {start} -> {end}，{ends!r} 不是。"],
                [
                    "evidence 是单元或此前对话的摘录",
                    "evidence {evidence!r} 不是当前单元或此前对话的原文摘录。",
                ],
                [
                    "“不完整”的意图都有 refines 边",
                    "这些意图补充了已有意图，缺少 refines 边：{missing!r}。",
                ],
            ],
            n,
        )
    if call == "D":
        summary = unit_key != "tool_result"
        extra = [
            *GRAPH,
            candidates % "当前 topic 的用户意图和聚合编号",
            '"relations": [\n    "satisfies",\n    "blocks"\n  ]',
        ]
        checks = [["目标编号在候选里", "{wrong!r} 不在 candidates 里。"]]
        if summary:
            checks.append(["summary 非空", "summary 不能为空。"])
        return spec(
            X.D_SUMMARY_SYSTEM_PROMPT if summary else X.D_SYSTEM_PROMPT,
            X.D_SUMMARY_SCHEMA if summary else X.D_SCHEMA,
            head("observation", extra) + WINDOW + unit + "\n</当前单元>",
            checks,
            n,
        )
    raise KeyError(call)


COMMON = {
    "request": json.dumps(
        {
            "model": "deployment/pc-qwen38-np8",
            "messages": [
                {"role": "system", "content": "（下面的 system 全文）"},
                {"role": "user", "content": "（下面的 user 模板）"},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True},
            "思考预算字段（字段名由 Hive 网关声明）": M._HIVE_THINKING_BUDGET["xhigh"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": "（下面的 JSON schema 全文）"},
            },
        },
        ensure_ascii=False,
        indent=1,
    ),
    "budget": f"prompt 预算 {W.PROMPT_BUDGET:,} token，其中预留 {X.FEEDBACK_ROOM} token 给重问反馈；窗口先试前 {W.ROUNDS[0]} 轮，放不下退到 {W.ROUNDS[-1]} 轮",
    "entries": [
        "【用户】\\n消息正文（图片各一行 [图片] 路径）",
        "【AI】\\n回复文字",
        "【用户信号】\\n信号原文",
        "【任务通知】\\n通知原文",
        "【系统】\\n系统消息原文",
        "【AI 的工具操作】\\n更早轮次的第二级压缩",
        "【AI 此前的工具操作（已压缩）】\\n当前轮超预算时的压缩",
        "【AI 的工具调用】\\n工具名 意图：… 对象：…",
        "【工具结果】\\n第一级压缩的概括",
    ],
}
