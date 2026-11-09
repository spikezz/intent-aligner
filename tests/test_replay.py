from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intent_aligner.corpus import TOOL_ROOT
from intent_aligner.extract import into_stores, walk
from intent_aligner.llm import LLM_HOME
from intent_aligner.replay import (
    RULE_SOURCES,
    Call,
    Conversation,
    Exchange,
    ask_rules,
    build_page,
    graph_changes,
    judge,
    literal_ranges,
    literal_texts,
    parse_run,
    read_exchange,
)
from intent_aligner.store import Stores
from intent_aligner.window import Digests

SOL_RUN = TOOL_ROOT / "corpus" / "results" / "claude-5a1f0c1a-sol-l1"
T0 = datetime(2026, 11, 8, 19, 0, tzinfo=UTC)


def _call(subject: str, prompt: str, minute: int, error: str | None = None) -> Call:
    exchange = Exchange(
        path=Path(f"/x/{minute}.jsonl"),
        model="m",
        system="s",
        prompt=prompt,
        schema=None,
        reply="{}",
        reasoning=None,
        usage={},
        finish="stop" if error is None else error,
        request={},
        ended=T0 + timedelta(minutes=minute),
    )
    return Call(
        stage="L1", ask="call", subject=subject, started=None, exchange=exchange, error=error
    )


def _empty() -> Conversation:
    return Conversation(
        items=[],
        digests=Digests(),
        segments={},
        assignments=[],
        units=[],
        unit_items=[],
        round_of={},
        window_rounds=2,
    )


def test_retry_rejects_the_attempt_before_it_with_its_feedback() -> None:
    calls = [_call("tool:a", "P", 0), _call("tool:a", "P\n\n你上一次的回答不符合要求", 1)]
    got = [(a.attempt, a.verdict, a.reason, a.feedback_at) for a in judge(calls, _empty())]
    expected = [(1, "rejected", "你上一次的回答不符合要求", None), (2, "accepted", "", 1)]
    assert got == expected, f"{got!r} vs {expected!r}"


def test_question_asked_again_from_scratch_fails_the_earlier_one() -> None:
    calls = [_call("tool:a", "P", 0), _call("tool:b", "P", 1), _call("tool:a", "P", 2)]
    got = [(a.subject, a.verdict) for a in judge(calls, _empty())]
    expected = [("tool:a", "failed"), ("tool:b", "accepted"), ("tool:a", "accepted")]
    assert got == expected, f"{got!r} vs {expected!r}"


def test_errored_call_fails() -> None:
    got = [a.verdict for a in judge([_call("tool:a", "P", 0, error="HTTP 500")], _empty())]
    assert got == ["failed"], got


needs_sol = pytest.mark.skipif(
    not (SOL_RUN / "run.sh").exists() or not (LLM_HOME / "codex").exists(),
    reason="needs the claude-5a1f0c1a-sol-l1 run and its codex transcripts",
)


@needs_sol
def test_run_sh_names_the_runs_files() -> None:
    run = parse_run(SOL_RUN)
    got = (run.merged.name, run.backend, run.model, run.digests, run.extract)
    expected = (
        "claude-5a1f0c1a.jsonl",
        "codex",
        "gpt-6.1-sol",
        SOL_RUN / "digests.jsonl",
        SOL_RUN / "extract",
    )
    assert got == expected, f"{got!r} vs {expected!r}"


@needs_sol
def test_old_run_matches_every_call_its_records_account_for() -> None:
    run = parse_run(SOL_RUN)
    conversation = Conversation.load(run)
    page = build_page(run, None, None)
    asks = Counter((c.stage, c.ask) for c in page.calls)
    digests = conversation.digests
    expected = {
        ("L1", "call"): len(digests.calls),
        ("L1", "result"): sum(len(r.summaries) for r in digests.results.values()),
        ("SEG", "segment"): sum(len(s) for s in conversation.segments.values()),
        ("LOOP", "topic"): sum(a.attempts for a in conversation.assignments),
        ("LOOP", "round"): len((run.extract / "tool_compressions.jsonl").read_text().splitlines()),
    }
    got = {key: asks[key] for key in expected}
    assert got == expected, f"{got!r} vs {expected!r}"
    assert page.unplaced == [], page.unplaced


def test_every_ask_has_retry_feedback_and_parsed_asks_have_checks() -> None:
    no_checks = {"round"}
    got = {
        ask: (
            any(r.kind == "feedback" for r in ask_rules(ask)),
            any(r.kind == "check" for r in ask_rules(ask)),
        )
        for ask in RULE_SOURCES
    }
    expected = {ask: (True, ask not in no_checks) for ask in RULE_SOURCES}
    assert got == expected, f"{got!r} vs {expected!r}"


@needs_sol
def test_window_reports_the_rounds_really_sent() -> None:
    page = build_page(parse_run(SOL_RUN), None, None)
    windowed = [c for c in page.calls if c.window_seen is not None]
    assert windowed, "no call of the run has a window"
    bad = [
        c.subject
        for c in windowed
        if not min(1, c.window_limit) <= len(c.window_seen or []) <= c.window_limit
    ]
    assert bad == [], bad


@needs_sol
def test_needs_review_units_have_a_route_with_their_unsent_steps() -> None:
    page = build_page(parse_run(SOL_RUN), None, None)
    units = [j for j in page.jobs if j.stage == "LOOP"]
    wrong = [
        j.id
        for j in units
        if bool(j.route) != bool((j.record["assignment"] or {}).get("needs_review"))
    ]
    assert wrong == [], wrong


def test_codex_reasoning_summaries_are_read_from_the_transcript(tmp_path: Path) -> None:
    events = [
        {"type": "request", "model": "m", "system_prompt": "s", "prompt": "p"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "甲"}},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "乙"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    exchange = read_exchange(path)
    assert exchange is not None and exchange.reasoning == "甲\n\n乙", exchange


def test_prompt_literals_cover_the_tags_of_a_prompt() -> None:
    prompt = "<此前的对话>\nhello\n</此前的对话>"
    got = [prompt[a:b] for a, b in literal_ranges(prompt, literal_texts())]
    assert "hello" not in "".join(got), got
    assert any("<此前的对话>" in part for part in got), got


@needs_sol
def test_graph_changes_add_up_to_the_final_graphs() -> None:
    run = parse_run(SOL_RUN)
    conversation = Conversation.load(run)
    nodes: dict[str, dict[str, object]] = {}
    edges: dict[str, list[object]] = {}
    for change in graph_changes(run, conversation):
        if change.topic is None:
            continue
        topic_nodes = nodes.setdefault(change.topic, {})
        for removed in change.removed:
            del topic_nodes[removed]
        topic_nodes.update(change.nodes)
        kept = [e for e in edges.get(change.topic, []) if e not in change.removed_edges]
        edges[change.topic] = kept + change.edges
    stores = Stores(merged_file=str(run.merged), items=conversation.items)
    for step, record in zip(
        walk(conversation.items, conversation.segments), conversation.assignments, strict=False
    ):
        into_stores(stores, step, record)
    graphs = {
        topic: store.meta.graph.model_dump(mode="json", by_alias=True, exclude_none=True)
        for topic, store in stores.stores.items()
    }
    got = {topic: (nodes[topic], edges[topic]) for topic in nodes}
    expected = {topic: (g["nodes"], g["edges"]) for topic, g in graphs.items()}
    assert got == expected, f"{got!r} vs {expected!r}"
