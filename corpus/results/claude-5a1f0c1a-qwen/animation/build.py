"""Rebuild the per-endpoint-call animation from the records of the Qwen run on disk."""

import json
import re
import sys
from pathlib import Path

import fixed

ROOT = Path(__file__).parent
RUN = ROOT.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "extraction-flow.html"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


records = jsonl(RUN / "extract/assignments.jsonl")
for record in records:
    for edge in record["edges"]:
        edge["from"] = edge.pop("from_")
digests = jsonl(RUN / "digests.jsonl")
segments = {m["uuid"]: m["segments"] for m in jsonl(RUN / "segments.jsonl")}
overflow = jsonl(RUN / "extract/tool_compressions.jsonl")
calls = {d["tool_use_id"]: d for d in digests if d["kind"] == "call"}
results = {d["tool_use_id"]: d for d in digests if d["kind"] == "result"}
rounds = [d for d in digests if d["kind"] == "round"]

SYS = "system：这次调用的任务说明和 JSON 格式要求"
WIN = "<此前的对话>：最近两轮（放不下时退到一轮），工具操作以压缩形式出现"
CUR = "<当前单元>：这个单元的内容"


def header(*fields):
    return "<header>：" + "、".join(fields)


def graph_note(g):
    return f"当前 topic 的整张图谱（{len(g['nodes'])} 个节点，{len(g['edges'])} 条边）"


frames = []
origins = {
    i.source.uuid: i.origin
    for i in fixed.X.conversation_units(next(RUN.parents[1].glob("merged/claude-5a1f0c1a*.jsonl")))
    if type(i).__name__ == "Observation"
}


def frame(**kw):
    frames.append({"attempts": 1, "writes": [], "fail": None, "skip": None, **kw})


for d in [d for d in digests if d["kind"] == "call"][:2]:
    rid = d["tool_use_id"]
    frame(
        phase="L1",
        call="压缩·调用",
        title="第一级压缩：提炼这次工具调用的目的",
        unit="工具调用 " + d["targets"][:150],
        parts=[SYS, "<调用>：这次工具调用的完整原文（命令、参数等）"],
        reply={"intents": d["intents"], "targets": d["targets"][:200]},
        checks=[
            "回答符合 JSON 格式",
            "每条 evidence 都是调用原文的摘录",
            "explicit 为 false 时 note 非空",
        ],
        writes=["写入 digests.jsonl（kind=call）"],
        record=d,
    )
    frame(
        phase="L1",
        call="压缩·结果",
        title="第一级压缩：把这次调用的返回结果压成一句话",
        unit="工具结果（原文可能很长）",
        parts=[SYS, "<调用>：产生这个结果的调用", "<结果>：返回内容的原文"],
        reply={"summary": " ".join(results[rid]["summaries"])},
        checks=["回答符合 JSON 格式"],
        writes=["写入 digests.jsonl（kind=result）"],
        record=results[rid],
    )
frames[-1]["skip"] = (
    f"第一级压缩一共是 {len(calls) + len(results)} 次端点调用"
    f"（{len(calls)} 个调用 + {len(results)} 个结果），8 路并行。这里只播放前 4 次。"
)
for d in rounds[:1]:
    frame(
        phase="L2",
        call="压缩·整轮",
        title="第二级压缩：把一整轮的工具操作写成一段话",
        unit="一轮对话里全部工具操作的第一级压缩结果",
        parts=[
            SYS,
            "这一轮开头的用户消息（超长时截断）",
            "<轨迹>：这一轮每个调用的意图和每个结果的概括",
        ],
        reply={"summary": d["summary"]},
        checks=["回答符合 JSON 格式"],
        writes=["写入 digests.jsonl（kind=round）"],
        record=d,
    )
frames[-1]["skip"] = f"第二级压缩一共是 {len(rounds)} 轮，这里只播放第 1 轮。"

first_uuid = records[0]["unit"]["uuid"]
for s in segments[first_uuid]:
    frame(
        phase="SEG",
        call="切分",
        title="切分：从指针位置往后切出一段用户消息",
        unit=s["text"][:260],
        attempts=s["attempts"],
        parts=[
            SYS,
            "<此前的对话>",
            f"<已处理>：消息的前 {s['start']} 个字符",
            f"<剩余>：从第 {s['start']} 个字符到消息末尾",
        ],
        reply={"intents": s["intents"], "segment_end": s["text"][-16:]},
        checks=["segment_end 在 <剩余> 里能唯一定位", "每条 evidence 都是这一段原文的摘录"],
        writes=[
            f"这一段是字符 [{s['start']}, {s['end']})，指针移到 {s['end']}，写入 segments.jsonl"
        ],
        record=s,
    )
frames[-1]["skip"] = (
    f"切分对每条用户消息循环，这份对话共 {len(segments)} 条用户消息，"
    f"{sum(len(v) for v in segments.values())} 段。这里只播放第 1 条消息的切分。"
)

topics, stores, graphs = {}, {}, []
KIND_ZH = {
    "user_segment": "用户段落",
    "user_signal": "用户信号",
    "tool_call": "工具调用",
    "observation": "观测",
}


def store(topic):
    return stores.setdefault(topic, {"nodes": {}, "edges": [], "members": {}})


def snapshot(topic):
    if topic is None:
        return {"t": None, "title": "未归属", "nodes": [], "edges": []}
    s = store(topic)
    return {
        "t": topic,
        "title": topics.get(topic, ""),
        "nodes": list(s["nodes"].values()),
        "edges": list(s["edges"]),
    }


def unit_text(r):
    u, intents = r["unit"], r["intents"]
    if u["kind"] == "user_segment":
        return next(s["text"] for s in segments[u["uuid"]] if [s["start"], s["end"]] == u["chars"])[
            :260
        ]
    if u["kind"] == "tool_call":
        return calls[u["tool_use_id"]]["targets"][:220]
    if u["tool_use_id"]:
        return "工具结果的概括：" + " ".join(results[u["tool_use_id"]]["summaries"])[:220]
    if u["kind"] == "user_signal":
        return (intents[0]["evidence"] if intents else "（用户信号，这个单元没有产生意图）")[:220]
    return (
        r["pending_text"]
        or (r["observations"][0]["label"] if r["observations"] else "（任务通知或系统消息）")
    )[:220]


for n, r in enumerate(records, 1):
    u, topic, intents = r["unit"], r["topic"], r["intents"]
    kind = u["kind"]
    label = f"单元 {n} · {KIND_ZH[kind]}" + (
        "（工具结果）" if kind == "observation" and u["tool_use_id"] else ""
    )
    text = unit_text(r)
    failed = re.match(r"call ([TABGLCD]) ", r["review_reason"] or "")
    failed = failed.group(1) if failed else None
    before = snapshot(topic)
    known = {x["id"] for x in before["nodes"]}
    s = store(topic) if topic else None
    existing_intents = [i for i in known if i.startswith("I")]
    seq = []
    if r["attempts"] > 0:
        answer = (
            {"topic": "NEW", "title": r["new_title"]}
            if r["new_title"]
            else {"topic": topic or "（未得到合法回答）", "title": ""}
        )
        hint = ["round_topic_hint（这一轮上一个工具调用的 topic）"] if kind == "tool_call" else []
        seq.append(
            dict(
                call="T",
                title="调用 T：这个单元属于哪个 topic？",
                attempts=r["attempts"],
                parts=[SYS, header(f"topic 列表（{len(topics)} 个）", "单元信息", *hint), WIN, CUR],
                reply=answer,
                checks=["编号在 topic 列表里", "回答 NEW 时标题非空"],
            )
        )
    intent_view = [
        {k: i.get(k) for k in ("evidence", "explicit", "note")}
        | {"text": i.get("label") or f"（沿用已有编号 {i['id']}）"}
        for i in intents
    ]
    graphed = header("topic 列表", "当前 topic", "单元信息", graph_note(before))
    if kind == "user_signal":
        seq.append(
            dict(
                call="A",
                title="调用 A：这个用户信号有哪些意图？",
                parts=[SYS, graphed, WIN, CUR],
                reply={"intents": intent_view},
                checks=[
                    "每项字段齐全",
                    "evidence 是单元原文的摘录",
                    "explicit 为 false 时 note 非空",
                ],
            )
        )
    if kind in ("user_segment", "user_signal") and intents and existing_intents:
        ans = [
            {"match": i["refines"], "complete": False}
            if i.get("refines")
            else {"match": i["id"], "complete": True}
            if i["id"] in known
            else {"match": "NEW", "complete": True}
            for i in intents
        ]
        seq.append(
            dict(
                call="B",
                title="调用 B：这些意图是已有的哪一个？",
                parts=[
                    SYS,
                    graphed + f"、候选（用户意图 {len(existing_intents)} 个）",
                    WIN,
                    CUR,
                    "<本单元的意图>",
                ],
                reply={"answers": ans},
                checks=[
                    "答案和本单元的意图一一对应",
                    "选的编号在候选列表里",
                    "选已有编号的都回答了 complete",
                ],
            )
        )
    if kind == "tool_call" and intents:
        cand = [i for i in known if i.startswith("G")]
        titles = {i["aggregate"]: i["label"] for i in intents if i.get("label")}
        ans = [
            {"aggregate": "NEW", "title": titles[a]}
            if a not in known
            else {"aggregate": a, "title": ""}
            for a in (i["aggregate"] for i in intents)
        ]
        seq.append(
            dict(
                call="G",
                title="调用 G：每个底层意图属于哪个聚合？",
                parts=[
                    SYS,
                    graphed + f"、候选（聚合 {len(cand)} 个）",
                    WIN,
                    CUR,
                    "<本单元的意图>",
                    "<此后的对话>：当前单元之后的 8 个单元",
                ],
                reply={"answers": ans},
                checks=[
                    "答案和本单元的意图一一对应",
                    "选的编号是当前 topic 的聚合",
                    "NEW 时标准文本非空",
                ],
            )
        )
        for aggregate in dict.fromkeys(i["aggregate"] for i in intents):
            members = s["members"].get(aggregate) if s else None
            if members:
                mine = [i for i in intents if i["aggregate"] == aggregate]
                seq.append(
                    dict(
                        call="L",
                        title=f"调用 L：这些底层意图是 {aggregate} 已有的哪一个？",
                        parts=[
                            SYS,
                            graphed + f"、候选（{aggregate} 的 {len(members)} 个成员）",
                            WIN,
                            CUR,
                            "<本单元的意图>",
                            "<已有的底层意图>",
                        ],
                        reply={
                            "answers": [
                                {"match": i["id"] if i["id"] in members else "NEW"} for i in mine
                            ]
                        },
                        checks=["答案和这个聚合的底层意图一一对应", "选的编号是这个聚合的成员"],
                    )
                )
    if (
        kind != "observation"
        and intents
        and failed != "G"
        and failed != "L"
        and failed != "B"
        and failed != "A"
    ):
        seq.append(
            dict(
                call="C",
                title="调用 C：这个单元的意图和已有节点有什么关系？",
                parts=[
                    SYS,
                    graphed + "、候选（全部节点 + 尚未建节点的观测）、relations、unit_intents",
                    WIN,
                    CUR,
                ],
                reply={
                    "edges": [
                        {k: e[k] for k in ("from", "to", "relation", "evidence")}
                        for e in r["edges"]
                    ]
                },
                checks=[
                    "一端是本单元的意图，另一端在候选或本单元的意图里",
                    "relation 的方向和节点类型匹配",
                    "evidence 是单元或对话原文的摘录",
                ],
            )
        )
    if kind == "observation" and topic:
        summary = (
            {
                "summary": r["pending_text"]
                or (r["observations"][0]["label"] if r["observations"] else "")
            }
            if not u["tool_use_id"]
            else {}
        )
        out = [{"relation": e["relation"], "target": e["to"]} for e in r["edges"]]
        seq.append(
            dict(
                call="D",
                title="调用 D：这个观测达成或阻碍了哪些已有意图？",
                parts=[
                    SYS,
                    graphed + "、候选（用户意图和聚合）、relations",
                    WIN,
                    "<当前单元>：这个观测"
                    + ("（第一级压缩的概括）" if u["tool_use_id"] else "（原文）"),
                ],
                reply={"outcomes": out} | summary,
                checks=["relation 只能是 satisfies 或 blocks", "目标编号在候选列表里"]
                + ([] if u["tool_use_id"] else ["概括非空"]),
            )
        )
    if failed:
        keep = next((i for i, f in enumerate(seq) if f["call"] == failed), len(seq) - 1)
        seq = seq[: keep + 1]
    if r["new_title"]:
        topics[topic] = r["new_title"]
    pending_nodes = []
    for i in intents:
        node = i.get("aggregate") or i["id"]
        if node not in known and node not in pending_nodes:
            pending_nodes.append(node)
    pending_nodes += [o["id"] for o in r["observations"]]
    if s is not None:
        for i in intents:
            if i.get("aggregate"):
                a = i["aggregate"]
                if i.get("label"):
                    s["nodes"][a] = {"id": a, "kind": "G", "label": i["label"]}
                if i.get("text"):
                    s["members"].setdefault(a, {})[i["id"]] = i["text"]
            elif i.get("label"):
                s["nodes"][i["id"]] = {"id": i["id"], "kind": "I", "label": i["label"]}
        for o in r["observations"]:
            s["nodes"][o["id"]] = {"id": o["id"], "kind": "O", "label": o["label"]}
        for e in r["edges"]:
            if not any(
                x["from"] == e["from"] and x["to"] == e["to"] and x["relation"] == e["relation"]
                for x in s["edges"]
            ):
                s["edges"].append({k: e[k] for k in ("from", "to", "relation", "evidence")})
    after = snapshot(topic)
    graphs.append(after)
    graph_index = len(graphs) - 1
    writes = []
    new_i = [x for x in pending_nodes]
    if new_i:
        writes.append("新节点：" + "、".join(new_i))
    if r["edges"]:
        writes.append(
            "新边：" + "；".join(f"{e['from']} {e['relation']} {e['to']}" for e in r["edges"])
        )
    if r["pending_text"]:
        writes.append("这个观测暂不建节点，概括留在窗口里，以后可以被 triggers 边指到")
    writes.append(f"写入 assignments.jsonl 和 topics/{topic or '未归属'} 的存储文件")
    unit_key = (
        kind if kind != "observation" else "tool_result" if u["tool_use_id"] else origins[u["uuid"]]
    )
    log = [f"new topic {topic}: {r['new_title']}"] if r["new_title"] else []
    for k, f in enumerate(seq):
        last = k == len(seq) - 1
        frame(
            phase="LOOP",
            unit=text,
            unitLabel=label,
            unitNo=n,
            topic=topic,
            topicTitle=after["title"],
            graph=graph_index,
            pending=pending_nodes,
            edgesBefore=len(before["edges"]),
            commit=last,
            unitKey=unit_key,
            kind=kind,
            record=(r if last else None),
            log=(log if last else []),
            writes=writes
            if last
            else ["这一步的结果先留在本单元的记录里，等这个单元的调用全部结束才写入存储"],
            fail=(r["review_reason"] if last and r["needs_review"] else None),
            **f,
        )
    if not seq:
        frame(
            phase="LOOP",
            call="代码",
            title="这个单元没有调用端点",
            unit=text,
            unitLabel=label,
            unitNo=n,
            topic=topic,
            topicTitle=after["title"],
            graph=graph_index,
            pending=pending_nodes,
            edgesBefore=len(before["edges"]),
            commit=True,
            parts=[],
            reply={},
            checks=[],
            writes=writes,
            fail=r["review_reason"] if r["needs_review"] else None,
        )

specs = {}
for f in frames:
    if f["call"] == "代码":
        f["spec"] = None
        continue
    key = "|".join((f["call"], f.get("unitKey", ""), f.get("kind", "")))
    specs.setdefault(key, fixed.call_spec(f["call"], f.get("unitKey", ""), f.get("kind", "")))
    f["spec"] = key
    f.pop("parts", None)
    f.pop("checks", None)

data = {
    "specs": specs,
    "common": fixed.COMMON,
    "frames": frames,
    "graphs": graphs,
    "units": len(records),
    "overflow": len(overflow),
}
html = (
    (ROOT / "template.html")
    .read_text()
    .replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
)
html = (
    '<!doctype html>\n<html lang="zh-CN">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}</style>\n'
    + html
)
OUT.write_text(html, encoding="utf-8")
print(len(records), "units;", len(frames), "frames;", len(html) // 1000, "KB;", OUT)
