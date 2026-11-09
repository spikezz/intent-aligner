"""The pointer and call T over the small hand-made timeline of test_window."""

from __future__ import annotations

import json
import re
from pathlib import Path

import attrs
import pytest
import yaml

from intent_aligner.extract import (
    CALL_ATTEMPTS,
    FEEDBACK_ROOM,
    T_ATTEMPTS,
    T_SYSTEM_PROMPT,
    Asks,
    Assignment,
    UnitRef,
    extract_units,
    log_lines,
    topics_from,
    write_topic_files,
)
from intent_aligner.intent import Intent
from intent_aligner.llm import TRUNCATED_FEEDBACK, Ask, TruncatedReply
from intent_aligner.segment import Segment
from intent_aligner.store import (
    UNASSIGNED_FILE,
    Stores,
    TopicStore,
    dump_store,
    graph_with_status,
    load_store,
)
from intent_aligner.units import AIText, Item, Observation, UserSignal
from intent_aligner.window import (
    PROMPT_BUDGET,
    Count,
    Digests,
    ResultDigest,
    ToolCompression,
    render_round_so_far,
)
from tests.test_window import DIGESTS, TIMELINE, _call, _call_digest, _result, _source, _user

FULL = Digests(
    calls={c: _call_digest(c, f"意图{c}", f"对象{c}") for c in ("c1", "c2")},
    results={c: ResultDigest(tool_use_id=c, summaries=[f"结果{c}"]) for c in ("c1", "c2")},
    rounds=DIGESTS.rounds,
)


def _segment(start: int, end: int) -> Segment:
    return Segment(start=start, end=end, text="", intents=[], attempts=1)


SEGMENTS = {
    "u1": [_segment(0, 2), _segment(2, 3)],
    "u2": [_segment(0, 3)],
    "u3": [_segment(0, 3)],
}


def _header(prompt: str) -> dict[str, object]:
    match = re.search(r"<header>\n(.*?)\n</header>", prompt, re.S)
    assert match, prompt
    header: dict[str, object] = json.loads(match.group(1))
    return header


def _new_then_first(prompts: list[str]) -> Ask:
    """Opens a topic on the first call, then answers the first registered topic."""

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        topics = _header(prompt)["topics"]
        assert isinstance(topics, list)
        if not topics:
            return json.dumps({"topic": "NEW", "title": "做第一件事"})
        return json.dumps({"topic": topics[0]["id"], "title": ""})

    return ask


def _no_round(prompt: str) -> str:
    raise AssertionError("the small timeline never needs a compression")


def _unit_intents(prompt: str) -> list[str]:
    listed = prompt.split("<本单元的意图>\n")[1].split("\n</本单元的意图>")[0]
    return listed.split("\n")


def _all_new(prompt: str) -> str:
    return json.dumps(
        {"answers": [{"match": "NEW", "complete": True} for _ in _unit_intents(prompt)]}
    )


def _one_aggregate(prompt: str) -> str:
    """Files everything under the first aggregate, opening it when there is none."""
    candidates = _header(prompt)["candidates"]
    assert isinstance(candidates, list)
    answer = {"aggregate": candidates[0], "title": ""} if candidates else None
    answer = answer or {"aggregate": "NEW", "title": "调查项目"}
    return json.dumps({"answers": [answer for _ in _unit_intents(prompt)]})


def _no_repeats(prompt: str) -> str:
    return json.dumps({"answers": [{"match": "NEW"} for _ in _unit_intents(prompt)]})


def _no_signal_intents(prompt: str) -> str:
    return json.dumps({"intents": []})


def _no_edges(prompt: str) -> str:
    return json.dumps({"edges": []})


def _no_outcomes(prompt: str) -> str:
    return json.dumps({"outcomes": []})


def _no_outcomes_summary(prompt: str) -> str:
    return json.dumps({"outcomes": [], "summary": "概括"})


def _assignments(
    items: list[Item],
    segments: dict[str, list[Segment]],
    digests: Digests,
    ask: Ask,
    count: Count = len,
    done: list[Assignment] | None = None,
    *,
    signal: Ask = _no_signal_intents,
    registry: Ask = _all_new,
    aggregate: Ask = _one_aggregate,
    tool_intent: Ask = _no_repeats,
    stores: Stores | None = None,
    relation: Ask = _no_edges,
    outcome: Ask = _no_outcomes,
    outcome_update: Ask = _no_outcomes_summary,
) -> list[Assignment]:
    asks = Asks(
        topic=ask,
        signal=signal,
        registry=registry,
        aggregate=aggregate,
        tool_intent=tool_intent,
        relation=relation,
        outcome=outcome,
        outcome_summary=_no_outcomes_summary,
        outcome_update=outcome_update,
        round=_no_round,
    )
    stores = stores or Stores(merged_file="merged.jsonl", items=items)
    segments = {"a1": [], **segments}
    events = extract_units(items, segments, digests, done or [], [], asks, count, stores, 2)
    return [event for event in events if isinstance(event, Assignment)]


def _run(ask: Ask, done: list[Assignment] | None = None) -> list[Assignment]:
    return _assignments(TIMELINE, SEGMENTS, FULL, ask, done=done)


def test_every_unit_but_ai_text_is_assigned_in_order() -> None:
    got = [(a.unit.kind, a.unit.uuid, a.topic) for a in _run(_new_then_first([]))]

    assert got == [
        ("user_segment", "u1", "T001"),
        ("user_segment", "u1", "T001"),
        ("tool_call", "c1", "T001"),
        ("observation", "r-c1", "T001"),
        ("user_segment", "u2", "T001"),
        ("tool_call", "c2", "T001"),
        ("observation", "r-c2", "T001"),
        ("user_segment", "u3", "T001"),
    ], got


def test_tool_results_take_their_calls_topic_without_call_t() -> None:
    prompts: list[str] = []
    assignments = _run(_new_then_first(prompts))

    assert len(prompts) == 6, f"{len(prompts)} calls; results must not reach call T"
    results = [a for a in assignments if a.unit.kind == "observation"]
    assert all(a.attempts == 0 for a in results), results


def test_a_rejected_call_is_the_users_decision_and_asks_call_t() -> None:
    rejection = UserSignal(
        source=_source("rej-c1"),
        signal="rejection",
        text="The user doesn't want to proceed with this tool use.",
        tool_use_id="c1",
    )
    prompts: list[str] = []
    items: list[Item] = [_user("u1", "第一条"), _call("c1"), rejection]
    assignments = _assignments(items, {"u1": [_segment(0, 3)]}, FULL, _new_then_first(prompts))

    got = [(a.unit.kind, a.attempts) for a in assignments]
    assert got == [("user_segment", 1), ("tool_call", 1), ("user_signal", 1)], got
    assert len(prompts) == 3, f"{len(prompts)} calls; the rejection must reach call T"


def test_a_new_topic_is_listed_for_the_next_unit() -> None:
    prompts: list[str] = []
    first = _run(_new_then_first(prompts))[0]

    assert (first.topic, first.new_title) == ("T001", "做第一件事"), first
    assert _header(prompts[1])["topics"] == [{"id": "T001", "title": "做第一件事"}], prompts[1]


def test_only_tool_calls_get_the_round_hint() -> None:
    prompts: list[str] = []
    _run(_new_then_first(prompts))
    hints = [_header(p).get("round_topic_hint") for p in prompts]

    assert hints == [None, None, None, None, None, None], (
        f"{hints}: c1 and c2 are each the first call of their round"
    )
    timeline = [*TIMELINE[:4], TIMELINE[2].model_copy(update={"tool_use_id": "c2"})]
    prompts.clear()
    _assignments(timeline, SEGMENTS, FULL, _new_then_first(prompts))
    assert _header(prompts[-1]).get("round_topic_hint") == "T001", prompts[-1]


def test_a_segment_is_marked_inside_the_whole_message() -> None:
    prompts: list[str] = []
    _run(_new_then_first(prompts))

    assert "【用户】\n第一<当前段落>条</当前段落>" in prompts[1], prompts[1]
    assert _header(prompts[1])["unit"] == {
        "kind": "user_segment",
        "actor": "user",
        "session": "s",
        "timestamp": "2026-11-06T00:00:00+00:00",
        "chars": [2, 3],
    }, prompts[1]


def test_the_current_round_shows_level_one_digests_up_to_the_unit() -> None:
    prompts: list[str] = []
    _run(_new_then_first(prompts))
    c2 = prompts[4]

    assert "【AI 的工具操作】\n概括a1" in c2, "earlier rounds come as round digests"
    assert "概括u2" not in c2, "the round in hand must not show how it ended"
    assert c2.endswith(
        "<当前单元>\n【AI 的工具调用】\nBash 意图：意图c2 对象：对象c2\n</当前单元>"
    ), c2


def test_render_round_so_far_shows_calls_and_results() -> None:
    text = render_round_so_far(TIMELINE, 0, 4, FULL, [])

    assert text == (
        "【用户】\n第一条\n\n【AI】\n好的\n\n"
        "【AI 的工具调用】\nBash 意图：意图c1 对象：对象c1\n\n【工具结果】\n结果c1"
    ), text


def test_two_bad_answers_leave_the_unit_and_its_result_unassigned() -> None:
    def ask(prompt: str) -> str:
        if "【AI 的工具调用】\nBash 意图：意图c1" in prompt.split("<当前单元>")[1]:
            return json.dumps({"topic": "T999", "title": ""})
        return _new_then_first([])(prompt)

    by_id = {a.unit.uuid: a for a in _run(ask)}

    assert by_id["c1"].topic is None and by_id["c1"].needs_review, by_id["c1"]
    assert by_id["c1"].attempts == T_ATTEMPTS, by_id["c1"]
    assert "T999" in (by_id["c1"].review_reason or ""), by_id["c1"]
    assert by_id["r-c1"].topic is None, "the result follows its unassigned call"


def test_retry_feedback_reaches_the_model() -> None:
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return "不是 JSON" if len(prompts) == 1 else json.dumps({"topic": "NEW", "title": "事"})

    first = _run(ask)[0]

    assert first.attempts == 2, first
    assert "不是要求的 JSON" in prompts[1], prompts[1]


def test_a_truncated_reply_is_retried() -> None:
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        if len(prompts) == 1:
            raise TruncatedReply("length")
        return json.dumps({"topic": "NEW", "title": "事"})

    first = _run(ask)[0]

    assert first.attempts == 2, first
    assert TRUNCATED_FEEDBACK in prompts[1], prompts[1]


def test_a_window_that_does_not_fit_needs_review_without_a_call() -> None:
    def ask(prompt: str) -> str:
        raise AssertionError("no call T when the window does not fit")

    got = _assignments(TIMELINE, SEGMENTS, FULL, ask, lambda t: 1000 * len(t))

    assert all(a.topic is None and a.attempts == 0 for a in got), got
    assert all(a.needs_review for a in got), "no unit may be skipped silently"
    assert [a.observations[0].label for a in got if a.unit.kind == "observation"] == [
        "结果c1",
        "结果c2",
    ], got


def test_a_rerun_resumes_after_the_done_units() -> None:
    done = _run(_new_then_first([]))[:5]
    prompts: list[str] = []
    rest = _run(_new_then_first(prompts), done)

    assert [a.unit.uuid for a in rest] == ["c2", "r-c2", "u3"], rest
    assert _header(prompts[0])["topics"] == [{"id": "T001", "title": "做第一件事"}], prompts[0]


def test_a_rerun_refuses_assignments_that_no_longer_line_up() -> None:
    done = _run(_new_then_first([]))[1:3]

    with pytest.raises(ValueError, match="start a new --out"):
        _run(_new_then_first([]), done)


def test_a_message_without_segments_is_refused() -> None:
    with pytest.raises(ValueError, match=r"intent_aligner\.segment"):
        _assignments(TIMELINE, {}, FULL, _new_then_first([]))


def test_topic_files_follow_the_store_format(tmp_path: Path) -> None:
    stores = Stores(merged_file="merged.jsonl", items=TIMELINE)
    assignments = _assignments(TIMELINE, SEGMENTS, FULL, _new_then_first([]), stores=stores)
    write_topic_files(tmp_path, topics_from(assignments, tmp_path), stores)

    listing = yaml.safe_load((tmp_path / "topics.yaml").read_text(encoding="utf-8"))
    store = load_store(tmp_path / "topics" / "T001.yaml")
    assert listing == {
        "topics": [
            {"id": "T001", "title": "做第一件事", "file": str(tmp_path / "topics/T001.yaml")}
        ]
    }, listing
    nodes = {i: (n.type, n.label) for i, n in store.meta.graph.nodes.items()}
    assert nodes == {
        "G001": ("aggregate", "调查项目"),
        "O001": ("aggregate_observation", "概括"),
    }, nodes
    spans = [(s.id, s.kind, s.chars, [i.id for i in s.intents]) for s in store.spans]
    assert spans == [
        ("S001", "user_text", (0, 2), []),
        ("S002", "user_text", (2, 3), []),
        ("S003", "tool_calls", None, ["L001"]),
        ("S004", "user_text", (0, 3), []),
        ("S005", "tool_calls", None, ["L002"]),
        ("S006", "user_text", (0, 3), []),
    ], spans
    assert not (tmp_path / "topics" / UNASSIGNED_FILE).exists(), "nothing went unassigned"


def test_the_playground_store_loads_and_dumps_unchanged() -> None:
    path = Path(__file__).parents[1] / "playground" / "topics" / "T001.yaml"
    store = load_store(path)

    assert TopicStore.model_validate(yaml.safe_load(dump_store(store))) == store
    assert yaml.safe_load(dump_store(store)) == yaml.safe_load(path.read_text(encoding="utf-8"))


def _said(*texts: str) -> list[Intent]:
    return [Intent(text=t, evidence="第", explicit=True, note="") for t in texts]


def _with_user_intents(first: list[Intent], second: list[Intent]) -> dict[str, list[Segment]]:
    return {
        **SEGMENTS,
        "u1": [_segment(0, 2).model_copy(update={"intents": first}), _segment(2, 3)],
        "u2": [_segment(0, 3).model_copy(update={"intents": second})],
    }


def _matching(answers: list[dict[str, object]]) -> Ask:
    def registry(prompt: str) -> str:
        return json.dumps({"answers": answers})

    return registry


def test_call_b_is_skipped_until_the_topic_has_intents_and_sees_only_user_ones() -> None:
    registry_prompts: list[str] = []

    def registry(prompt: str) -> str:
        registry_prompts.append(prompt)
        return _all_new(prompt)

    segments = _with_user_intents(_said("做甲"), _said("做乙"))
    _assignments(TIMELINE, segments, FULL, _new_then_first([]), registry=registry)

    assert len(registry_prompts) == 1, "u1 opens the registry and tool calls never reach call B"
    header = _header(registry_prompts[0])
    assert header["candidates"] == ["I001"], header
    assert header["current_topic"] == "T001", header
    assert _unit_intents(registry_prompts[0]) == ["1. 做乙（依据：第）"], registry_prompts[0]


def test_ai_speech_gets_its_own_nodes_and_is_matched_only_against_the_ai() -> None:
    items: list[Item] = [
        _user("u1", "第一条"),
        AIText(source=_source("a1"), text="回答甲"),
        AIText(source=_source("a2"), text="回答乙"),
        _user("u2", "第二条"),
    ]
    segments = {
        uuid: [_segment(0, 3).model_copy(update={"intents": _said(uuid)})]
        for uuid in ("u1", "a1", "a2", "u2")
    }
    candidates: list[object] = []

    def registry(prompt: str) -> str:
        candidates.append(_header(prompt)["candidates"])
        return _all_new(prompt)

    stores = Stores(merged_file="merged.jsonl", items=items)
    made = _assignments(
        items, segments, Digests(), _new_then_first([]), registry=registry, stores=stores
    )
    store = stores.stores["T001"]

    assert candidates == [["I002"], ["I001"]], candidates
    assert [a.unit.kind for a in made] == [
        "user_segment",
        "ai_segment",
        "ai_segment",
        "user_segment",
    ], made
    actors = {i: getattr(n, "actor", None) for i, n in store.meta.graph.nodes.items()}
    assert actors == {"I001": "user", "I002": "ai", "I003": "ai", "I004": "user"}, actors
    assert [s.kind for s in store.spans] == ["user_text", "ai_text", "ai_text", "user_text"], store


def test_a_complete_match_keeps_the_registered_id_once_per_unit() -> None:
    stores = Stores(merged_file="merged.jsonl", items=TIMELINE)
    segments = _with_user_intents(_said("做甲"), _said("甲一", "甲二"))
    registry = _matching([{"match": "I001", "complete": True}] * 2)
    by_id = {
        a.unit.uuid + str(a.unit.chars): a
        for a in _assignments(
            TIMELINE, segments, FULL, _new_then_first([]), registry=registry, stores=stores
        )
    }

    assert [i.id for i in by_id["u2(0, 3)"].intents] == ["I001"], by_id
    assert list(stores.topic_nodes("T001")) == ["I001", "G001", "O001"], "a match registers nothing"


def test_an_incomplete_match_registers_a_refining_intent() -> None:
    stores = Stores(merged_file="merged.jsonl", items=TIMELINE)
    segments = _with_user_intents(_said("做甲"), _said("细化", "另一件"))
    registry = _matching([{"match": "I001", "complete": False}, {"match": "NEW", "complete": True}])
    records = _assignments(
        TIMELINE, segments, FULL, _new_then_first([]), registry=registry, stores=stores
    )
    second = next(a for a in records if a.unit.uuid == "u2")

    got = [(i.id, i.label, i.details_of) for i in second.intents]
    assert got == [("I002", "细化", "I001"), ("I003", "另一件", None)], got


def test_call_b_answers_that_never_fit_leave_the_segment_in_its_topic_for_review() -> None:
    prompts: list[str] = []

    def registry(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"answers": [{"match": "I999", "complete": True}]})

    stores = Stores(merged_file="merged.jsonl", items=TIMELINE)
    segments = _with_user_intents(_said("做甲"), _said("做乙"))
    records = _assignments(
        TIMELINE, segments, FULL, _new_then_first([]), registry=registry, stores=stores
    )
    second = next(a for a in records if a.unit.uuid == "u2")

    assert len(prompts) == CALL_ATTEMPTS, prompts
    assert "'I999'" in prompts[1], "the retry says which match was wrong"
    assert (second.topic, second.needs_review, second.intents) == ("T001", True, []), second


def _two_intents() -> Digests:
    intents = [
        Intent(text=f"操作{n}", evidence="Bash", explicit=False, note="推断") for n in (1, 2)
    ]
    return attrs.evolve(
        FULL, calls={**FULL.calls, "c1": FULL.calls["c1"].model_copy(update={"intents": intents})}
    )


def test_call_g_files_tool_intents_and_only_aggregates_enter_the_graph() -> None:
    prompts: list[str] = []
    stores = Stores(merged_file="merged.jsonl", items=TIMELINE)

    def aggregate(prompt: str) -> str:
        prompts.append(prompt)
        return _one_aggregate(prompt)

    _assignments(
        TIMELINE, SEGMENTS, _two_intents(), _new_then_first([]), aggregate=aggregate, stores=stores
    )

    assert _header(prompts[0])["candidates"] == [], prompts[0]
    assert _header(prompts[1])["candidates"] == ["G001"], prompts[1]
    assert list(stores.topic_nodes("T001")) == ["G001", "O001"], stores.topic_nodes("T001")
    span = next(s for s in stores.stores["T001"].spans if s.kind == "tool_calls")
    got = [(i.id, i.note) for i in span.intents]
    assert got == [("L001", "推断"), ("L002", "推断")], got
    mapping = stores.stores["T001"].meta.tool_intents
    assert mapping == {"G001": {"L001": "操作1", "L002": "操作2", "L003": "意图c2"}}, mapping


def test_new_aggregates_with_one_title_are_one_and_others_are_separate() -> None:
    stores = Stores(merged_file="merged.jsonl", items=TIMELINE)
    answers = [{"aggregate": "NEW", "title": t} for t in ("甲", " 甲 ")]
    calls = iter([answers, [{"aggregate": "NEW", "title": "乙"}]])

    def aggregate(prompt: str) -> str:
        return json.dumps({"answers": next(calls)})

    records = _assignments(
        TIMELINE, SEGMENTS, _two_intents(), _new_then_first([]), aggregate=aggregate, stores=stores
    )
    by_id = {a.unit.uuid: a for a in records}

    assert [(i.aggregate, i.label) for i in by_id["c1"].intents] == [("G001", "甲"), ("G001", None)]
    assert [(i.aggregate, i.label) for i in by_id["c2"].intents] == [("G002", "乙")]


def test_call_g_rejects_an_unknown_aggregate_and_an_empty_title() -> None:
    prompts: list[str] = []
    replies = iter(
        [
            {"answers": [{"aggregate": "G999", "title": ""}] * 2},
            {"answers": [{"aggregate": "NEW", "title": " "}] * 2},
            {"answers": [{"aggregate": "NEW", "title": ""}]},
        ]
    )

    def aggregate(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(next(replies))

    digests = _two_intents()
    items: list[Item] = [_user("u1", "第一条"), _call("c1")]
    stores = Stores(merged_file="merged.jsonl", items=items)
    got = _assignments(
        items,
        {"u1": [_segment(0, 3)]},
        digests,
        _new_then_first([]),
        aggregate=aggregate,
        stores=stores,
    )[-1]

    assert "'G999'" in prompts[1] and "title 不能为空" in prompts[2], prompts
    assert (got.topic, got.needs_review, got.intents) == ("T001", True, []), got
    assert "call G" in (got.review_reason or ""), got


def test_a_repeated_tool_intent_keeps_its_id_and_the_mapping_stays_one_entry() -> None:
    prompts: list[str] = []

    def tool_intent(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"answers": [{"match": "L001"}]})

    items: list[Item] = _same_intent_calls([])
    stores = Stores(merged_file="merged.jsonl", items=items)
    _assignments(
        items,
        {"u1": [_segment(0, 3)]},
        FULL,
        _new_then_first([]),
        tool_intent=tool_intent,
        stores=stores,
    )

    assert len(prompts) == 1, "the call that opened the aggregate has nothing to match"
    assert _header(prompts[0])["candidates"] == ["L001"], prompts[0]
    assert "L001：意图c1" in prompts[0], prompts[0]
    mapping = stores.stores["T001"].meta.tool_intents
    assert mapping == {"G001": {"L001": "意图c1"}}, mapping
    span = stores.stores["T001"].spans[-1]
    assert [i.id for i in span.intents] == ["L001", "L001"], span.intents


def test_call_l_answers_that_never_fit_leave_the_call_for_review() -> None:
    prompts: list[str] = []

    def tool_intent(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"answers": [{"match": "L999"}]})

    items: list[Item] = [_user("u1", "第一条"), _call("c1"), _result("c1"), _call("c2")]
    stores = Stores(merged_file="merged.jsonl", items=items)
    got = _assignments(
        items,
        {"u1": [_segment(0, 3)]},
        FULL,
        _new_then_first([]),
        tool_intent=tool_intent,
        stores=stores,
    )[-1]

    assert len(prompts) == CALL_ATTEMPTS and "'L999'" in prompts[1], prompts
    assert (got.topic, got.needs_review, got.intents) == ("T001", True, []), got
    assert "call L" in (got.review_reason or ""), got


def _same_intent_calls(between: list[Item]) -> list[Item]:
    return [_user("u1", "第一条"), _call("c1"), _result("c1"), *between, _call("c2"), _result("c2")]


def _tool_spans(items: list[Item], aggregate: Ask = _one_aggregate) -> list[list[str] | None]:
    stores = Stores(merged_file="merged.jsonl", items=items)
    _assignments(
        items,
        {"u1": [_segment(0, 3)]},
        FULL,
        _new_then_first([]),
        aggregate=aggregate,
        stores=stores,
    )
    return [s.source.tool_use_ids for s in stores.stores["T001"].spans if s.kind == "tool_calls"]


def test_adjacent_calls_in_the_same_aggregate_share_a_span() -> None:
    assert _tool_spans(_same_intent_calls([])) == [["c1", "c2"]]


def test_adjacent_calls_in_different_aggregates_get_separate_spans() -> None:
    titles = iter(["甲", "乙"])

    def aggregate(prompt: str) -> str:
        return json.dumps({"answers": [{"aggregate": "NEW", "title": next(titles)}]})

    assert _tool_spans(_same_intent_calls([]), aggregate) == [["c1"], ["c2"]]


def test_ai_text_between_calls_starts_a_new_span() -> None:
    spans = _tool_spans(_same_intent_calls([AIText(source=_source("a1"), text="接着")]))

    assert spans == [["c1"], ["c2"]], spans


def test_a_merged_span_carries_both_calls_text() -> None:
    items = _same_intent_calls([])
    stores = Stores(merged_file="merged.jsonl", items=items)
    _assignments(
        items,
        {"u1": [_segment(0, 3)]},
        FULL,
        _new_then_first([]),
        stores=stores,
    )
    span = stores.stores["T001"].spans[-1]

    assert span.text == "Bash:\ncommand: ls\n\nBash:\ncommand: ls", span.text
    assert [i.id for i in span.intents] == ["L001", "L002"], span.intents


def test_segments_that_leave_a_gap_fail_the_coverage_check() -> None:
    with pytest.raises(AssertionError, match="end to end"):
        _assignments(
            TIMELINE,
            {**SEGMENTS, "u1": [_segment(0, 1), _segment(2, 3)]},
            FULL,
            _new_then_first([]),
        )


def test_an_unassigned_call_goes_to_the_unassigned_file(tmp_path: Path) -> None:
    def ask(prompt: str) -> str:
        if "【AI 的工具调用】\nBash 意图：意图c1" in prompt.split("<当前单元>")[1]:
            return json.dumps({"topic": "T999", "title": ""})
        return _new_then_first([])(prompt)

    stores = Stores(merged_file="merged.jsonl", items=TIMELINE)
    assignments = _assignments(TIMELINE, SEGMENTS, FULL, ask, stores=stores)
    write_topic_files(tmp_path, topics_from(assignments, tmp_path), stores)

    unassigned = load_store(tmp_path / "topics" / UNASSIGNED_FILE)
    assert unassigned.meta.topic is None, unassigned.meta
    got = [(s.kind, s.source.tool_use_ids, s.needs_review, s.intents) for s in unassigned.spans]
    assert got == [("tool_calls", ["c1"], True, [])], got
    assert "topic: null" in (tmp_path / "topics" / UNASSIGNED_FILE).read_text(encoding="utf-8")


REJECTION = UserSignal(
    source=_source("rej-c1"),
    signal="rejection",
    text="The user doesn't want to proceed with this tool use. the user said:\n先别删",
    tool_use_id="c1",
)


def test_call_a_gives_a_signal_its_intents() -> None:
    prompts: list[str] = []

    def signal(prompt: str) -> str:
        prompts.append(prompt)
        evidence = "先别删" if len(prompts) > 1 else "不在原文里"
        intent = {"text": "暂时保留文件", "evidence": evidence, "explicit": True, "note": ""}
        return json.dumps({"intents": [intent]})

    items: list[Item] = [_user("u1", "第一条"), _call("c1"), REJECTION]
    stores = Stores(merged_file="merged.jsonl", items=items)
    got = _assignments(
        items,
        {"u1": [_segment(0, 3)]},
        FULL,
        _new_then_first([]),
        signal=signal,
        stores=stores,
    )[-1]

    assert len(prompts) == 2 and "不在原文里" in prompts[1], prompts
    assert [(i.id, i.label, i.explicit) for i in got.intents] == [("I001", "暂时保留文件", True)]
    node = stores.topic_nodes("T001")["I001"]
    assert getattr(node, "actor", None) == "user", node
    span = stores.stores["T001"].spans[-1]
    assert (span.kind, span.source.tool_use_id, span.intents[0].note) == (
        "user_signal",
        "c1",
        None,
    ), span


def test_an_observation_can_open_a_topic() -> None:
    notice = Observation(
        source=_source("n1"), origin="system", text="API 错误", tool_use_id=None, is_error=False
    )
    items: list[Item] = [notice, _user("u1", "第一条")]
    stores = Stores(merged_file="merged.jsonl", items=items)
    _assignments(items, {"u1": [_segment(0, 3)]}, FULL, _new_then_first([]), stores=stores)

    assert [s.kind for s in stores.stores["T001"].spans] == ["user_text"], stores.stores


def test_a_rerun_rebuilds_the_same_stores() -> None:
    whole = Stores(merged_file="merged.jsonl", items=TIMELINE)
    done = _assignments(TIMELINE, SEGMENTS, FULL, _new_then_first([]), stores=whole)[:5]
    resumed = Stores(merged_file="merged.jsonl", items=TIMELINE)
    _assignments(TIMELINE, SEGMENTS, FULL, _new_then_first([]), done=done, stores=resumed)

    assert resumed.stores == whole.stores
    got = (resumed.intents, resumed.aggregates, resumed.tool_intents, resumed.spans)
    assert got == (whole.intents, whole.aggregates, whole.tool_intents, whole.spans), got


PAIRS = 20
LONG: list[Item] = [_user("u1", "做一件事"), AIText(source=_source("a1"), text="好的，我开始")]
for _n in range(PAIRS):
    LONG += [_call(f"c{_n}"), _result(f"c{_n}")]
LONG_DIGESTS = Digests(
    calls={f"c{n}": _call_digest(f"c{n}", f"意图{n}") for n in range(PAIRS)},
    results={
        f"c{n}": ResultDigest(tool_use_id=f"c{n}", summaries=[f"结果{n}"]) for n in range(PAIRS)
    },
)
LONG_SEGMENTS: dict[str, list[Segment]] = {"u1": [_segment(0, 4)], "a1": []}


@pytest.fixture
def no_following(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call G's look-ahead also spends budget, which would hide where a compression is due."""
    monkeypatch.setattr("intent_aligner.extract.render_following", lambda *_: "")


def _heavy_results(text: str) -> int:
    """Each level-1 result costs 1,000 tokens, so the round outgrows the budget after a few."""
    return len(text) + 1000 * text.count("【工具结果】")


def _long_run(
    round_prompts: list[str], t_prompts: list[str], compressed: list[ToolCompression] | None = None
) -> list[Assignment | ToolCompression]:
    def ask_round(prompt: str) -> str:
        round_prompts.append(prompt)
        return json.dumps({"summary": f"压缩{len(round_prompts)}"})

    asks = Asks(
        topic=_new_then_first(t_prompts),
        signal=_no_signal_intents,
        registry=_all_new,
        aggregate=_one_aggregate,
        tool_intent=_no_repeats,
        relation=_no_edges,
        outcome=_no_outcomes,
        outcome_summary=_no_outcomes_summary,
        outcome_update=_no_outcomes_summary,
        round=ask_round,
    )
    stores = Stores(merged_file="merged.jsonl", items=LONG)
    events = extract_units(
        LONG, LONG_SEGMENTS, LONG_DIGESTS, [], compressed or [], asks, _heavy_results, stores, 2
    )
    return list(events)


def test_a_round_is_compressed_only_when_the_prompt_would_not_fit(no_following: None) -> None:
    t_prompts: list[str] = []
    events = _long_run([], t_prompts)
    first = next(e for e in events if isinstance(e, ToolCompression))
    after = events[events.index(first) + 1]

    assert isinstance(after, Assignment) and after.unit.uuid == LONG[first.through].source.uuid, (
        after
    )
    assert not any(isinstance(e, Assignment) and e.needs_review for e in events), events
    budget = PROMPT_BUDGET - FEEDBACK_ROOM
    before = [p for p in t_prompts if "已压缩" not in p]
    used = _heavy_results(T_SYSTEM_PROMPT) + _heavy_results(before[-1])
    assert used <= budget, f"the last uncompressed prompt must fit: {used:,}"
    assert used + 1000 > budget - 100, f"compressed too early: {used:,} of {budget:,} used"


def test_the_passage_replaces_only_tool_records() -> None:
    t_prompts: list[str] = []
    _long_run([], t_prompts)
    compressed = next(p for p in t_prompts if "已压缩" in p)
    window = compressed.split("<此前的对话>")[1].split("</此前的对话>")[0]

    assert window.startswith(
        "\n【用户】\n做一件事\n\n【AI】\n好的，我开始\n\n【AI 此前的工具操作（已压缩）】\n压缩"
    ), window
    assert "意图0 " not in window, "compressed calls must not show again"


def test_each_compression_is_its_own_level_two_passage() -> None:
    round_prompts: list[str] = []
    t_prompts: list[str] = []
    events = _long_run(round_prompts, t_prompts)
    compressions = [e for e in events if isinstance(e, ToolCompression)]

    assert len(compressions) >= 2, compressions
    first, second = compressions[:2]
    assert second.begin == first.through, "a compression covers the records after the last one"
    assert "压缩1" not in round_prompts[1], "an earlier passage is never compressed again"
    assert all("<发言>（AI）\n好的，我开始\n</发言>" in p for p in round_prompts), round_prompts
    last = t_prompts[-1]
    assert last.index("压缩1") < last.index("压缩2"), "every passage stays in the window, in order"


def test_a_rerun_reuses_the_compressions_in_force() -> None:
    events = _long_run([], [])
    first = next(e for e in events if isinstance(e, ToolCompression))
    again = _long_run([], [], [first])

    assert first not in again, "a compression already in force is not written again"
    assert next(e for e in again if isinstance(e, ToolCompression)).begin == first.through, again


def test_earlier_passages_count_against_the_budget(no_following: None) -> None:
    t_prompts: list[str] = []
    _long_run([], t_prompts)
    budget = PROMPT_BUDGET - FEEDBACK_ROOM
    with_first_only = [p for p in t_prompts if "压缩1" in p and "压缩2" not in p]
    used = _heavy_results(T_SYSTEM_PROMPT) + _heavy_results(with_first_only[-1])

    assert used <= budget, f"the last prompt before the second compression must fit: {used:,}"
    assert used + 1000 > budget - 100, (
        f"second compression too early: {used:,} of {budget:,} used with passage 1 counted"
    )


def test_what_call_g_sees_ahead_counts_against_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_through = [next(e for e in _long_run([], []) if isinstance(e, ToolCompression)).through]
    monkeypatch.setattr("intent_aligner.extract.render_following", lambda *_: "")
    first_through.append(
        next(e for e in _long_run([], []) if isinstance(e, ToolCompression)).through
    )

    assert first_through[0] < first_through[1], first_through


def test_call_g_alone_sees_what_follows_the_unit() -> None:
    aggregate_prompts: list[str] = []
    match_prompts: list[str] = []

    def aggregate(prompt: str) -> str:
        aggregate_prompts.append(prompt)
        return _one_aggregate(prompt)

    def tool_intent(prompt: str) -> str:
        match_prompts.append(prompt)
        return _no_repeats(prompt)

    _assignments(
        TIMELINE,
        SEGMENTS,
        _two_intents(),
        _new_then_first([]),
        aggregate=aggregate,
        tool_intent=tool_intent,
    )

    following = aggregate_prompts[0].split("<此后的对话>")[1]
    assert "意图c2" in following, aggregate_prompts[0]
    assert match_prompts, "c2 joins an aggregate that has members"
    assert all("<此后的对话>" not in p for p in match_prompts), match_prompts


ONE_ROUND: list[Item] = [
    _user("u1", "第一条"),
    _call("c1"),
    _result("c1"),
    _call("c2"),
    _result("c2"),
]
ONE_SEGMENT = {"u1": [_segment(0, 3)]}


def _one_round(
    relation: Ask = _no_edges,
    outcome: Ask = _no_outcomes,
    aggregate: Ask = _one_aggregate,
    outcome_update: Ask = _no_outcomes_summary,
) -> tuple[list[Assignment], Stores]:
    stores = Stores(merged_file="merged.jsonl", items=ONE_ROUND)
    got = _assignments(
        ONE_ROUND,
        ONE_SEGMENT,
        FULL,
        _new_then_first([]),
        aggregate=aggregate,
        stores=stores,
        relation=relation,
        outcome=outcome,
        outcome_update=outcome_update,
    )
    return got, stores


def _edges(stores: Stores) -> list[tuple[str, str, str, str | None]]:
    graph = stores.stores["T001"].meta.graph
    return [(e.from_, e.to, e.relation, e.span) for e in graph.edges]


def test_an_update_replaces_the_text_and_the_edges_of_the_aggregates_node() -> None:
    _, stores = _one_round(
        outcome=lambda prompt: '{"outcomes": [{"relation": "satisfies", "target": "G001"}]}',
        outcome_update=lambda prompt: json.dumps(
            {"outcomes": [{"relation": "satisfies", "target": "G001"}], "summary": "合并"}
        ),
    )

    graph = stores.stores["T001"].meta.graph
    node = graph.nodes["O001"]
    assert node.type == "aggregate_observation" and node.aggregate == "G001", node
    assert node.label == "合并", node
    assert [(e.from_, e.to, e.relation, e.evidence) for e in graph.edges] == [
        ("O001", "G001", "satisfies", "c2")
    ], graph.edges
    status = graph_with_status(graph)["nodes"]["G001"]["status"]  # type: ignore[index]
    assert status == "已完成", status


def test_an_update_that_no_longer_shows_it_done_puts_the_aggregate_back_to_pending() -> None:
    _, stores = _one_round(
        outcome=lambda prompt: '{"outcomes": [{"relation": "satisfies", "target": "G001"}]}',
    )

    graph = stores.stores["T001"].meta.graph
    assert graph.edges == [], graph.edges
    status = graph_with_status(graph)["nodes"]["G001"]["status"]  # type: ignore[index]
    assert status == "待完成", status


def test_calls_and_results_are_paired_outside_the_graph() -> None:
    _, stores = _one_round()

    meta = stores.stores["T001"].meta
    got = [(p.tool_use_id, p.result) for p in meta.tool_pairs["G001"]]
    assert got == [("c1", "结果c1"), ("c2", "结果c2")], got
    assert sum(n.type == "aggregate_observation" for n in meta.graph.nodes.values()) == 1


def test_an_aggregates_first_result_is_its_observation_as_summarized() -> None:
    titles = iter(["甲", "乙"])

    def aggregate(prompt: str) -> str:
        return json.dumps({"answers": [{"aggregate": "NEW", "title": next(titles)}]})

    _, stores = _one_round(aggregate=aggregate)

    nodes = stores.stores["T001"].meta.graph.nodes
    got = [(i, n.label) for i, n in nodes.items() if n.type == "aggregate_observation"]
    assert got == [("O001", "结果c1"), ("O002", "结果c2")], got


def test_call_c_gives_a_picked_pushed_observation_its_node() -> None:
    notice = Observation(
        source=_source("n1"), origin="system", text="API 错误", tool_use_id=None, is_error=False
    )
    items: list[Item] = [_user("u1", "第一条"), notice, _call("c1")]
    stores = Stores(merged_file="merged.jsonl", items=items)

    def relation(prompt: str) -> str:
        if "P1" not in prompt:
            return '{"edges": []}'
        edge = {"from": "P1", "to": "G001", "relation": "triggers", "evidence": "API 错误"}
        return json.dumps({"edges": [edge]})

    _assignments(items, ONE_SEGMENT, FULL, _new_then_first([]), stores=stores, relation=relation)

    node = stores.stores["T001"].meta.graph.nodes["O001"]
    assert node.type == "observation" and node.label == "概括", node
    assert _edges(stores) == [("O001", "G001", "triggers", "S002")], _edges(stores)
    assert list(stores.pending) == [], "a picked observation stops being pending"


def test_call_c_retries_a_trigger_into_the_aggregate_that_produced_the_observation() -> None:
    prompts: list[str] = []

    def relation(prompt: str) -> str:
        prompts.append(prompt)
        if "O001" not in prompt or len(prompts) > 2:
            return '{"edges": []}'
        edge = {"from": "O001", "to": "G001", "relation": "triggers", "evidence": "结果c1"}
        return json.dumps({"edges": [edge]})

    got, stores = _one_round(relation=relation)

    assert "自己的工具调用产生的结果" in prompts[2], prompts[2]
    assert not any(a.needs_review for a in got), got
    assert _edges(stores) == [], "the rejected edge must not be stored"


def test_call_c_retries_an_edge_against_the_direction_of_its_relation() -> None:
    prompts: list[str] = []

    def relation(prompt: str) -> str:
        prompts.append(prompt)
        if "O001" not in prompt or len(prompts) > 2:
            return '{"edges": []}'
        edge = {"from": "G001", "to": "O001", "relation": "triggers", "evidence": "结果c1"}
        return json.dumps({"edges": [edge]})

    got, stores = _one_round(relation=relation)

    assert "方向必须是" in prompts[2], prompts[2]
    assert not any(a.needs_review for a in got), got
    assert _edges(stores) == [], "the rejected edge must not be stored"


def test_an_incomplete_intent_serves_the_one_it_adds_to_without_asking_call_c() -> None:
    timeline: list[Item] = [_user("u1", "第一条"), _user("u2", "第二条")]
    said = Intent(text="做事", evidence="第", explicit=True, note="")
    segments = {
        u: [Segment(start=0, end=3, text="", intents=[said], attempts=1)] for u in ("u1", "u2")
    }
    stores = Stores(merged_file="merged.jsonl", items=timeline)

    def registry(prompt: str) -> str:
        return json.dumps({"answers": [{"match": "I001", "complete": False}]})

    _assignments(timeline, segments, FULL, _new_then_first([]), registry=registry, stores=stores)

    assert _edges(stores) == [("I002", "I001", "serves", "S002")], _edges(stores)


def test_a_failed_call_d_leaves_a_needs_review_node() -> None:
    _, stores = _one_round(
        outcome=lambda prompt: '{"outcomes": [{"relation": "x", "target": "y"}]}'
    )

    node = stores.stores["T001"].meta.graph.nodes["O001"]
    assert node.type == "observation" and node.needs_review is True, node


def test_replaying_the_assignments_rebuilds_the_same_stores() -> None:
    answers = iter(['{"outcomes": [{"relation": "blocks", "target": "G001"}]}'])

    def outcome(prompt: str) -> str:
        return next(answers, '{"outcomes": []}')

    got, stores = _one_round(outcome=outcome)
    replayed = Stores(merged_file="merged.jsonl", items=ONE_ROUND)
    assert _assignments(ONE_ROUND, ONE_SEGMENT, FULL, _no_round, done=got, stores=replayed) == [], (
        "every unit is done"
    )

    assert replayed.stores == stores.stores
    assert replayed.pending == stores.pending


def test_a_new_topic_does_not_hide_a_failure_from_the_log() -> None:
    unit = UnitRef(kind="user_segment", uuid="u1", chars=(0, 3))
    record = Assignment(
        unit=unit,
        topic="T001",
        new_title="标题",
        needs_review=True,
        review_reason="原因",
        attempts=1,
    )

    got = log_lines(record)

    assert len(got) == 2 and "new topic" in got[0] and "needs_review" in got[1], got


def test_a_message_without_intents_is_logged_as_a_placeholder() -> None:
    got = _assignments([_user("u1", "第一条")], ONE_SEGMENT, FULL, _new_then_first([]))

    assert got[0].placeholder, got
    assert any("placeholder" in line for line in log_lines(got[0])), got
