from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intent_aligner.corpus import TOOL_ROOT
from intent_aligner.llm import LLM_HOME
from intent_aligner.replay import (
    Call,
    Conversation,
    Exchange,
    build_page,
    judge,
    parse_run,
)
from intent_aligner.window import Digests

SOL_RUN = TOOL_ROOT / "corpus" / "results" / "claude-87eeb4bd-sol"
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
    reason="needs the claude-87eeb4bd-sol run and its codex transcripts",
)


@needs_sol
def test_run_sh_names_the_runs_files() -> None:
    run = parse_run(SOL_RUN)
    got = (run.merged.name, run.backend, run.model, run.digests, run.extract)
    expected = (
        "claude-87eeb4bd.jsonl",
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
        ("L2", "round"): len(digests.rounds),
        ("SEG", "segment"): sum(len(s) for s in conversation.segments.values()),
        ("LOOP", "topic"): sum(a.attempts for a in conversation.assignments),
        ("LOOP", "round"): len((run.extract / "tool_compressions.jsonl").read_text().splitlines()),
    }
    got = {key: asks[key] for key in expected}
    assert got == expected, f"{got!r} vs {expected!r}"
    assert page.unplaced == [], page.unplaced
