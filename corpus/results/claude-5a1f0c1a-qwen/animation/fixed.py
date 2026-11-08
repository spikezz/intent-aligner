"""Everything about a call that the code fixes, taken from the code itself.

Text the code writes as literals is parsed out of its source with ast; text the code
assembles at run time (the header JSON, a rendered unit, a window entry) comes from
calling the code on placeholder inputs. When a function, literal or expression this
file expects is gone, the build raises CodeChanged: the animation must follow the code.
"""

import ast
import json
import re
import sys
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[4]))
from intent_aligner import compress as Z
from intent_aligner import extract as X
from intent_aligner import intent as I
from intent_aligner import llm as M
from intent_aligner import segment as S
from intent_aligner import window as W
from intent_aligner.intent import Intent
from intent_aligner.segment import Segment
from intent_aligner.units import AIText, Observation, Source, ToolCall, UserMessage, UserSignal
from intent_aligner.window import CallDigest, Digests, ResultDigest


class CodeChanged(LookupError):
    """What this file reads from the code is no longer there."""


class Inline(str):
    """Text put in place of an expression as is, instead of as a {{slot}}."""


def _fail(module, path, what):
    raise CodeChanged(f"{module.__name__}.{path}：{what}。代码改了，动画的 fixed.py 要跟着改。")


@cache
def _tree(module):
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _node(module, path):
    """The def or class at dotted `path`, each name looked up inside the one before."""
    scope = _tree(module)
    for name in path.split("."):
        found = [
            n
            for n in ast.walk(scope)
            if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name == name and n is not scope
        ]
        if len(found) != 1:
            _fail(module, path, f"找到 {len(found)} 个 {name}，应该正好一个")
        scope = found[0]
    return scope


def _strings(fn):
    """The string literals and f-strings in `fn`, docstrings and f-string pieces left out."""
    skip = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
        if isinstance(n, ast.JoinedStr):
            skip |= {id(v) for v in ast.walk(n) if v is not n}
    found = [
        n
        for n in ast.walk(fn)
        if (
            isinstance(n, ast.JoinedStr)
            or (isinstance(n, ast.Constant) and isinstance(n.value, str))
        )
        and id(n) not in skip
    ]
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _literal(string):
    if isinstance(string, ast.Constant):
        return string.value
    return "".join(v.value for v in string.values if isinstance(v, ast.Constant))


def _pieces(module, string):
    """(text, None) for literal text and (None, source) for an expression.

    An expression that is a module-level string constant is literal text too,
    because the model sees its value.
    """
    if isinstance(string, ast.Constant):
        return [(string.value, None)]
    out = []
    for v in string.values:
        if isinstance(v, ast.Constant):
            out.append((v.value, None))
            continue
        source = ast.unparse(v.value)
        value = getattr(module, source, None) if isinstance(v.value, ast.Name) else None
        if isinstance(value, str) and v.conversion == -1 and v.format_spec is None:
            out.append((value, None))
            continue
        if v.conversion != -1:
            source += "!" + chr(v.conversion)
        if v.format_spec is not None:
            source += ":" + _literal(v.format_spec)
        out.append((None, source))
    return out


def find(module, path, marker):
    """The one string literal in `path` whose literal text contains `marker`."""
    hits = [s for s in _strings(_node(module, path)) if marker in _literal(s)]
    if len(hits) != 1:
        _fail(module, path, f"含 {marker!r} 的字符串有 {len(hits)} 个，应该正好一个")
    return hits[0]


def render(module, path, string, names, *, slot=True):
    """`string` with each expression replaced as `names` says.

    A name maps to a label, shown as a {{slot}} (bare with slot=False), or to Inline
    text. Every expression must have a name and every name must be used.
    """
    out, used = [], set()
    for text, source in _pieces(module, string):
        if source is None:
            out.append(text)
            continue
        if source not in names:
            _fail(module, path, f"表达式 {source} 在动画里没有说明")
        used.add(source)
        value = names[source]
        out.append(value if isinstance(value, Inline) or not slot else f"{{{{{value}}}}}")
    if set(names) - used:
        _fail(module, path, f"代码里已经没有表达式 {sorted(set(names) - used)}")
    return "".join(out)


def template(module, path, marker, names, *, slot=True):
    return render(module, path, find(module, path, marker), names, slot=slot)


def message(module, string):
    """A message the model is sent, its expressions shown as {source}."""
    return "".join(t if s is None else f"{{{s}}}" for t, s in _pieces(module, string))


def _assigned(module, path, name):
    """The string literals assigned to `name` in `path`, in source order."""
    found = [
        n.value
        for n in ast.walk(_node(module, path))
        if isinstance(n, ast.Assign | ast.AnnAssign)
        and any(
            ast.unparse(t) == name for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
    ]
    if not found:
        _fail(module, path, f"没有给 {name} 赋值")
    return sorted(found, key=lambda n: n.lineno)


def _returned(module, path):
    found = [
        n.value
        for n in ast.walk(_node(module, path))
        if isinstance(n, ast.Return) and isinstance(n.value, ast.JoinedStr)
    ]
    if len(found) != 1:
        _fail(module, path, f"返回 f-string 的 return 有 {len(found)} 个，应该正好一个")
    return found[0]


def _call(module, path, callee):
    found = [
        n
        for n in ast.walk(_node(module, path))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == callee
    ]
    if len(found) != 1:
        _fail(module, path, f"对 {callee} 的调用有 {len(found)} 处，应该正好一处")
    return found[0]


def _require(module, path, statement):
    """Fail unless `path` has `statement`, whose behaviour this file reproduces."""
    if all(
        ast.unparse(n) != statement
        for n in ast.walk(_node(module, path))
        if isinstance(n, ast.stmt)
    ):
        _fail(module, path, f"没有语句 {statement}")


def _evaluate(module, path, name):
    """The value of the one expression assigned to `name`, evaluated in the module."""
    (value,) = _assigned(module, path, name)
    return eval(compile(ast.Expression(value), module.__file__, "eval"), vars(module))


def _raises(module, path):
    """(message, raise node) for each raise in `path` whose message is a string literal."""
    found = [
        n
        for n in ast.walk(_node(module, path))
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and n.exc.args
        and isinstance(n.exc.args[0], ast.JoinedStr | ast.Constant)
    ]
    return [(message(module, n.exc.args[0]), n) for n in sorted(found, key=lambda n: n.lineno)]


def checks(module, path, labels):
    """[label, message] for each raise in `path`; each `labels` marker names one message."""
    out = []
    for text, _ in _raises(module, path):
        named = [label for marker, label in labels if marker in text]
        if len(named) != 1:
            _fail(module, path, f"消息 {text!r} 对上了 {len(named)} 条说明，应该正好一条")
        out.append([named[0], text])
    if len(out) != len(labels):
        _fail(module, path, f"代码有 {len(out)} 条校验消息，动画说明了 {len(labels)} 条")
    return out


def feedback(module, path, labels, elsewhere=()):
    """[label, text] of what each except clause of the retry loop sets `feedback` to.

    `labels` maps an exception name to its label; `elsewhere` are the exceptions whose
    feedback is shown as a check instead.
    """
    out, seen = [], set()
    for handler in (n for n in ast.walk(_node(module, path)) if isinstance(n, ast.ExceptHandler)):
        caught = ast.unparse(handler.type)
        seen.add(caught)
        if caught in elsewhere:
            continue
        if caught not in labels:
            _fail(module, path, f"except {caught} 在动画里没有说明")
        (value,) = [
            s.value
            for s in handler.body
            if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "feedback"
        ]
        text = getattr(module, value.id) if isinstance(value, ast.Name) else message(module, value)
        out.append([labels[caught], text])
    if set(labels) | set(elsewhere) != seen:
        _fail(
            module,
            path,
            f"except 子句是 {sorted(seen)}，动画说明的是 {sorted(set(labels) | set(elsewhere))}",
        )
    return out


def slots(text):
    """Split a template into fixed text and {{variable}} slots."""
    parts = re.split(r"(\{\{.*?\}\})", text, flags=re.S)
    return [["v", p[2:-2]] if p.startswith("{{") else ["f", p] for p in parts if p]


# Placeholder inputs for the code that renders at run time.

STAMP = datetime(2000, 1, 1, tzinfo=UTC)
SOURCE = Source(session="{{会话编号}}", uuid="u", timestamp=STAMP)
IMAGES = ("{{图片路径，每张图片一行}}",)
BEFORE, PART, AFTER = "{{这条消息在本段之前的文字}}", "{{本段文字}}", "{{这条消息在本段之后的文字}}"
DIGESTS = Digests(
    calls={
        "t": CallDigest(
            tool_use_id="t",
            intents=[Intent(text="{{第一级压缩的意图}}", evidence="e", explicit=True, note="")],
            targets="{{第一级压缩的 targets}}",
        )
    },
    results={"t": ResultDigest(tool_use_id="t", summaries=["{{第一级压缩的概括}}"])},
)
TOPICS = [("{{topic 编号}}", "{{topic 标题}}")]
TOPIC = "{{当前 topic 编号}}"


def _observation(origin, text, tool_use_id=None):
    return Observation(
        source=SOURCE, origin=origin, text=text, tool_use_id=tool_use_id, is_error=False
    )


def unit(key):
    """A placeholder unit of `key`: a unit kind, or an observation's origin."""
    if key == "user_segment":
        body = BEFORE + PART + AFTER
        message = UserMessage(source=SOURCE, body=body, queued=False, images=IMAGES)
        start = len(BEFORE)
        segment = Segment(start=start, end=start + len(PART), text=PART, intents=[], attempts=1)
        return X.UserSegment(message=message, segment=segment)
    if key == "tool_call":
        return ToolCall(source=SOURCE, tool_use_id="t", name="{{工具名}}", input={}, text="")
    if key == "tool_result":
        return _observation("tool_result", "", "t")
    if key == "user_signal":
        return UserSignal(source=SOURCE, signal="interrupt", text="{{信号原文}}", tool_use_id=None)
    if key == "task_notification":
        return _observation(key, "{{通知原文}}")
    if key == "system":
        return _observation(key, "{{系统消息原文}}")
    raise KeyError(key)


class _Graph:
    def model_dump(self, **_):
        return "{{当前 topic 的整张图谱：全部节点和边}}"


def _json_slots(header, shown):
    """The header with placeholder strings unquoted and placeholder numbers named."""
    header = re.sub(r'"(\{\{.*?\}\})"', r"\1", header)
    header = _swap(header, json.dumps(STAMP.isoformat()), "{{时间戳}}")
    if isinstance(shown, X.UserSegment):
        start, end = shown.segment.start, shown.segment.end
        header = _swap(
            header,
            rf"\[\s*{start},\s*{end}\s*\]",
            "{{本段在消息正文里的字符区间 [起点, 终点]}}",
            regex=True,
        )
    return header


def _swap(text, old, new, *, regex=False):
    pattern = old if regex else re.escape(old)
    if len(re.findall(pattern, text)) != 1:
        raise CodeChanged(f"{old!r} 不是正好出现一次。代码改了，动画的 fixed.py 要跟着改。")
    return re.sub(pattern, lambda _: new, text)


def _graph_header(path, shown, labels):
    """The header `path` builds with _graph_header, its extra arguments named by `labels`.

    `labels` maps the source of each argument after `stores` to its placeholder.
    """
    call = _call(X, path, "_graph_header")
    extra = []
    for arg in call.args[4:]:
        source = ast.unparse(arg)
        if source in labels:
            extra.append(labels.pop(source))
        else:
            try:
                extra.append(ast.literal_eval(arg))
            except ValueError:
                _fail(X, path, f"_graph_header 的参数 {source} 在动画里没有说明")
    if labels:
        _fail(X, path, f"_graph_header 已经不收参数 {sorted(labels)}")
    stores = SimpleNamespace(stores={TOPIC: SimpleNamespace(meta=SimpleNamespace(graph=_Graph()))})
    step, context = SimpleNamespace(unit=shown), SimpleNamespace(topics=TOPICS)
    return _json_slots(X._graph_header(step, context, TOPIC, stores, *extra), shown)


def _t_header(shown):
    _require(X, "extract_units", "hint = round_hint if isinstance(step.unit, ToolCall) else None")
    hint = "{{这一轮上一个工具调用的 topic}}" if isinstance(shown, ToolCall) else None
    call = _call(X, "_assign", "_header")
    extra = {
        ast.literal_eval(k): ast.unparse(v)
        for k, v in zip(call.args[2].keys, call.args[2].values, strict=True)
    }
    if extra != {"round_topic_hint": "hint"}:
        _fail(X, "_assign", f"_header 的附加字段是 {extra}")
    return _json_slots(X._header(shown, TOPICS, {"round_topic_hint": hint}), shown)


def _c_more():
    (more,) = _assigned(X, "_relate.head", "more")
    values = {
        "relations": _evaluate(X, "_relate", "relations"),
        "list(unit_nodes.values())": "{{本单元的意图节点}}",
    }
    out = {}
    for key, value in zip(more.keys, more.values, strict=True):
        source = ast.unparse(value)
        if source not in values:
            _fail(X, "_relate.head", f"more 的字段 {ast.unparse(key)} = {source} 在动画里没有说明")
        out[ast.literal_eval(key)] = values.pop(source)
    if values:
        _fail(X, "_relate.head", f"more 已经没有 {sorted(values)}")
    return out


def _listed():
    line = template(
        X,
        "_listed",
        "（依据：",
        {"n": "序号", "i.text": "意图文本", "i.evidence": "evidence"},
        slot=False,
    )
    return template(X, "_listed", "<本单元的意图>", {"lines": f"每个意图一行：{line}"})


def _tail(path, names):
    """What follows the current unit in the prompt `path` builds; `names` fills an assigned tail."""
    call = _call(X, path, "_fitted_prompt")
    if len(call.args) < 6:
        return ""
    source = ast.unparse(call.args[5])
    if source == "_listed(intents)":
        return _listed()
    if source == "tail":
        (tail,) = _assigned(X, path, "tail")
        return render(X, path, tail, names)
    _fail(X, path, f"_fitted_prompt 的 tail 参数 {source} 在动画里没有说明")


def _system(path):
    return getattr(X, ast.unparse(_call(X, path, "_fitted_prompt").args[2]))


def _user(head, shown, tail):
    first, then = _assigned(X, "_prompt", "current")
    current = render(
        X,
        "_prompt",
        first,
        {"_render_unit(step.unit, digests)": Inline(X._render_unit(shown, DIGESTS))},
    )
    if tail:
        current = render(X, "_prompt", then, {"current": Inline(current), "tail": Inline(tail)})
    names = {
        "head": Inline(head),
        "window": "窗口内容，条目格式见下方列表",
        "current": Inline(current),
    }
    return render(X, "_prompt", _returned(X, "_prompt"), names)


def _chunk_label(module, path):
    label = template(module, path, "段）", {"number": "i", "len(parts)": "n"}, slot=False)
    return f"太长被分块时：{label}"


NOTE = checks(
    I,
    "Intent._inferred_has_grounds",
    [("note 必须写推断依据", "explicit 为 false 时 note 必须写推断依据")],
)
FEEDBACK = {
    "extract": feedback(
        X,
        "_ask_until_valid",
        {
            "TruncatedReply": "回答被截断",
            "ValidationError": "回答不是要求的 JSON",
            "ValueError": "校验规则不通过",
        },
    ),
    "compress": feedback(
        Z, "_ask_until_valid", {"TruncatedReply": "回答被截断", "ValidationError": "回答不合格"}
    ),
    "segment": [
        ["所有反馈的外层", message(S, find(S, "build_prompt", "请重新回答"))],
        *feedback(
            S,
            "_next_segment",
            {"TruncatedReply": "回答被截断", "ValidationError": "回答不合格"},
            elsewhere=("AnchorError",),
        ),
    ],
}
COUNT = ("answers 有", "答案和本单元的意图一一对应")
PICK = ("不在 candidates 里", "选的编号在候选里或是 NEW")


def _segment_checks():
    _require(S, "_next_segment", "from_start = from_start or error.ambiguous")
    _require(S, "_next_segment", "rule = FROM_START_RULE if from_start else ''")
    (prefix,) = [
        text
        for label, text in feedback(
            S, "_next_segment", {"AnchorError": ""}, ("TruncatedReply", "ValidationError")
        )
    ]
    labels = {
        "找不到": "segment_end 在 <剩余> 里找得到",
        "出现了": "segment_end 在 <剩余> 里只出现一次",
        "不是以你给的 segment_end 开头": "改成从头抄之后，<剩余> 以它开头",
    }
    out = []
    for text, raised in _raises(S, "locate_end"):
        (label,) = [label for marker, label in labels.items() if marker in text]
        ambiguous = next(
            ast.literal_eval(k.value) for k in raised.exc.keywords if k.arg == "ambiguous"
        )
        under_from_start = any(
            isinstance(n, ast.If) and ast.unparse(n.test) == "from_start" and raised in ast.walk(n)
            for n in ast.walk(_node(S, "locate_end"))
        )
        rule = S.FROM_START_RULE if ambiguous or under_from_start else ""
        out.append([label, _swap(_swap(prefix, "{error}", text), "{rule}", rule)])
    if len(out) != len(labels):
        _fail(S, "locate_end", f"有 {len(out)} 条定位错误，动画说明了 {len(labels)} 条")
    strays = message(S, find(S, "_next_segment", "不是这一段的原文摘录"))
    return [*out, ["每条 evidence 都是这一段的摘录", strays], *NOTE]


def spec(system, schema, template_text, check_list, limit, family="extract"):
    return {
        "system": system,
        "schema": schema,
        "user": slots(template_text),
        "checks": check_list,
        "limit": limit,
        "feedback": FEEDBACK[family],
    }


def _compress_spec(call):
    if call == "压缩·调用":
        user = template(Z, "_compress_call", "<调用>", {"item.text": "这次工具调用的完整原文"})
        evidence = ("不是 <调用> 原文的摘录", "每条 evidence 都是 <调用> 原文的摘录")
        check_list = [*checks(Z, "_CallAnswer._evidence_quotes_the_call", [evidence]), *NOTE]
        return spec(
            Z.CALL_SYSTEM_PROMPT, Z.CALL_SCHEMA, user, check_list, Z.MAX_ATTEMPTS, "compress"
        )
    if call == "压缩·结果":
        names = {"digest.purpose": "第一级压缩的意图", "call.text": "这次调用的完整原文"}
        head = template(Z, "_compress_result", "意图：", names)
        names = {
            "head": Inline(head),
            "label": "结果" + _chunk_label(Z, "_compress_result"),
            "part": "返回内容的原文或其中一块",
        }
        user = template(Z, "_compress_result", "<结果>", names)
        return spec(Z.RESULT_SYSTEM_PROMPT, Z.SUMMARY_SCHEMA, user, [], Z.MAX_ATTEMPTS, "compress")
    queued = _literal(find(Z, "_message_block", "AI 工作时发出"))
    long = template(Z, "_message_block", "太长", {"label": ""}, slot=False)
    label = f"AI 工作时发出的消息加：{queued}；超过 {Z.MESSAGE_ROOM:,} token 再加：{long}"
    names = {"label": label, "text": "这一轮开头的用户消息，图片各一行 [图片] 路径"}
    block = render(Z, "_message_block", _returned(Z, "_message_block"), names)
    names = {
        "head": Inline(block),
        "label": "轨迹" + _chunk_label(Z, "summarize_trajectory"),
        "part": "这一轮每个调用和结果的压缩，一行一个",
    }
    user = template(Z, "summarize_trajectory", "<轨迹>", names)
    return spec(Z.ROUND_SYSTEM_PROMPT, Z.SUMMARY_SCHEMA, user, [], Z.MAX_ATTEMPTS, "compress")


def _segment_spec():
    (parts,) = _assigned(S, "build_prompt", "parts")
    names = {
        "window": "窗口内容",
        "message.body[:start]": "消息里指针之前的文字",
        "message.body[start:]": "消息里指针之后的文字",
    }
    shown = []
    for part in parts.elts:
        used = {k: v for k, v in names.items() if k in {s for _, s in _pieces(S, part)}}
        shown.append(render(S, "build_prompt", part, used))
    images = _literal(find(S, "build_prompt", "附带的图片"))
    joiner = _call(S, "build_prompt", "'\\n\\n'.join")
    separator = ast.literal_eval(joiner.func.value)
    user = separator.join(shown) + f"{{{{有图片时：{separator}{images}路径列表}}}}"
    return spec(S.SYSTEM_PROMPT, S.SCHEMA, user, _segment_checks(), S.MAX_ATTEMPTS, "segment")


def call_spec(call, unit_key=""):
    """The fixed content of one call; `unit_key` picks the unit it is shown with."""
    if call.startswith("压缩"):
        return _compress_spec(call)
    if call == "切分":
        return _segment_spec()
    shown = unit(unit_key)
    n = X.CALL_ATTEMPTS
    if call == "T":
        user = _user(_t_header(shown), shown, "")
        check_list = checks(
            X,
            "_assign.parse",
            [("不能为空", "回答 NEW 时标题非空"), ("不在 topics 列表里", "编号在 topics 列表里")],
        )
        return spec(_system("_assign"), X.T_SCHEMA, user, check_list, X.T_ATTEMPTS)
    if call == "A":
        path = "_with_intents"
        user = _user(_graph_header(path, shown, {}), shown, _tail(path, {}))
        evidence = ("不是 <当前单元> 原文的摘录", "每条 evidence 都是单元原文的摘录")
        check_list = [*checks(X, "_with_intents.parse_a", [evidence]), *NOTE]
        return spec(_system(path), X.A_SCHEMA, user, check_list, n)
    if call == "B":
        path = "_match_intents"
        header = _graph_header(path, shown, {"candidates": "{{当前 topic 的用户意图编号}}"})
        user = _user(header, shown, _tail(path, {}))
        return spec(
            _system(path), X.B_SCHEMA, user, checks(X, "_match_intents.parse_b", [COUNT, PICK]), n
        )
    if call == "G":
        path = "_file_under_aggregates"
        header = _graph_header(path, shown, {"candidates": "{{当前 topic 的聚合编号}}"})
        following = (
            f"当前单元之后最多 {W.FOLLOWING_ITEMS} 个条目，最多 {W.FOLLOWING_BUDGET:,} token"
        )
        tail = _tail(path, {"_listed(intents)": Inline(_listed()), "following": following})
        labels = [COUNT, PICK, ("aggregate 为 NEW 时 title", "NEW 时 title 非空")]
        return spec(
            _system(path),
            X.G_SCHEMA,
            _user(header, shown, tail),
            checks(X, f"{path}.parse", labels),
            n,
        )
    if call == "L":
        path = "_match_tool_intents"
        header = _graph_header(path, shown, {"list(members)": "{{这个聚合的 L 成员编号}}"})
        member = template(X, path, "：", {"i": "L 编号", "text": "底层意图文本"}, slot=False)
        tail = _tail(
            path, {"_listed(intents)": Inline(_listed()), "registered": f"每个成员一行：{member}"}
        )
        labels = [("answers 有", "答案和这个聚合的意图一一对应"), PICK]
        return spec(
            _system(path),
            X.L_SCHEMA,
            _user(header, shown, tail),
            checks(X, f"{path}.parse", labels),
            n,
        )
    if call == "C":
        offered = (
            "{{当前 topic 全部节点，加窗口内看得到的未建节点观测（每项含 id：P序号 和 text）}}"
        )
        header = _graph_header(
            "_relate.head", shown, {"[*known, *offered]": offered, "more": _c_more()}
        )
        labels = [
            ("不在 candidates 或 unit_intents 里", "两端都在 candidates 或 unit_intents 里"),
            ("两端必须不同", "两端不同，且有一端在 unit_intents 里"),
            ("的方向必须是", "方向和节点类型匹配"),
            ("自己的工具调用产生的结果", "triggers 不回到产生观测的聚合"),
            ("不是当前单元或此前对话的原文摘录", "evidence 是单元或此前对话的摘录"),
            ("缺少 refines 边", "“不完整”的意图都有 refines 边"),
        ]
        user = _user(header, shown, _tail("_relate", {}))
        return spec(_system("_relate"), X.C_SCHEMA, user, checks(X, "_relate.parse", labels), n)
    if call == "D":
        _require(X, "_observe", "needs_summary = unit.tool_use_id is None")
        _require(
            X, "_observe", "system = D_SUMMARY_SYSTEM_PROMPT if needs_summary else D_SYSTEM_PROMPT"
        )
        summary = shown.tool_use_id is None
        header = _graph_header(
            "_observe", shown, {"candidates": "{{当前 topic 的用户意图和聚合编号}}"}
        )
        check_list = checks(
            X,
            "_observe.parse",
            [("不在 candidates 里", "目标编号在候选里"), ("summary 不能为空", "summary 非空")],
        )
        if not summary:
            check_list = [c for c in check_list if c[0] != "summary 非空"]
        return spec(
            X.D_SUMMARY_SYSTEM_PROMPT if summary else X.D_SYSTEM_PROMPT,
            X.D_SUMMARY_SCHEMA if summary else X.D_SCHEMA,
            _user(header, shown, _tail("_observe", {})),
            check_list,
            n,
        )
    raise KeyError(call)


def _request():
    """The body llm.hive_request_body sends, for a Qwen deployment that declares a budget field."""
    budget_field = "（思考预算字段，字段名由 Hive 网关声明）"
    client = {
        "reasoning": True,
        "thinking_format": "qwen-chat-template",
        "thinking_token_budget_field": budget_field,
    }
    call = M.LLMCall(
        backend="hive",
        model="（Hive 部署名或算力池名）",
        effort="xhigh",
        system_prompt="（下面的 system 全文）",
        prompt="（下面的 user 模板）",
        json_schema={},
    )
    body = M.hive_request_body(call, client)
    if budget_field not in body:
        raise CodeChanged(
            "hive_request_body 没有用 client 声明的思考预算字段。代码改了，动画的 fixed.py 要跟着改。"
        )
    body["response_format"]["json_schema"]["schema"] = "（下面的 JSON schema 全文）"
    return json.dumps(body, ensure_ascii=False, indent=1)


def _entry(text):
    return re.sub(r"\{\{(.*?)\}\}", r"\1", text).replace("\n", "\\n")


def _entries():
    user = UserMessage(source=SOURCE, body="{{消息正文}}", queued=False, images=IMAGES)
    plain = [
        user,
        AIText(source=SOURCE, text="{{回复文字}}"),
        unit("user_signal"),
        unit("task_notification"),
        unit("system"),
    ]
    earlier = template(
        W,
        "render_items",
        "【AI 的工具操作】",
        {"_round_summary(digests, round_uuid)": "更早轮次的第二级压缩"},
    )
    compressed = template(
        W, "render_round_so_far", "已压缩", {"covering.summary": "当前轮超预算时的压缩"}
    )
    tool = [W.round_entry(unit(k), DIGESTS) for k in ("tool_call", "tool_result")]
    return [_entry(e) for e in [*map(W._plain_entry, plain), earlier, compressed, *tool]]


COMMON = {
    "request": _request(),
    "budget": f"prompt 预算 {W.PROMPT_BUDGET:,} token，其中预留 {X.FEEDBACK_ROOM} token 给重问反馈；窗口先试前 {W.ROUNDS[0]} 轮，放不下退到 {W.ROUNDS[-1]} 轮",
    "entries": _entries(),
}
