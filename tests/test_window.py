"""Window selection and rendering over a small hand-made timeline."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intent_aligner.intent import Intent
from intent_aligner.units import AIText, Item, Observation, Source, ToolCall, UserMessage
from intent_aligner.window import (
    FOLLOWING_BUDGET,
    FOLLOWING_ITEMS,
    CallDigest,
    Digests,
    RoundDigest,
    WindowError,
    render_following,
    render_items,
    window_text,
)


def _source(uuid: str) -> Source:
    return Source(session="s", uuid=uuid, timestamp=datetime(2026, 11, 6, tzinfo=UTC))


def _user(uuid: str, body: str) -> UserMessage:
    return UserMessage(source=_source(uuid), body=body, queued=False, images=())


def _call(tool_use_id: str) -> ToolCall:
    return ToolCall(
        source=_source(tool_use_id),
        tool_use_id=tool_use_id,
        name="Bash",
        input={"command": "ls"},
        text="Bash:\ncommand: ls",
    )


def _call_digest(tool_use_id: str, intent: str, targets: str = "t") -> CallDigest:
    return CallDigest(
        tool_use_id=tool_use_id,
        intents=[Intent(text=intent, evidence="Bash", explicit=False, note="从命令推断")],
        targets=targets,
    )


def _result(tool_use_id: str) -> Observation:
    return Observation(
        source=_source(f"r-{tool_use_id}"),
        origin="tool_result",
        text="x" * 5000,
        tool_use_id=tool_use_id,
        is_error=False,
    )


TIMELINE: list[Item] = [
    _user("u1", "第一条"),
    AIText(source=_source("a1"), text="好的"),
    _call("c1"),
    _result("c1"),
    _user("u2", "第二条"),
    _call("c2"),
    _result("c2"),
    _user("u3", "第三条"),
]
DIGESTS = Digests(
    rounds={u: RoundDigest(round_uuid=u, summary=f"概括{u}") for u in ("a1", "u1", "u2")},
)


def test_the_rounds_parameter_sets_how_far_back_the_window_reaches() -> None:
    three = window_text(TIMELINE, 7, DIGESTS, 10_000, len, 3)
    two = window_text(TIMELINE, 7, DIGESTS, 10_000, len, 2)

    assert three.startswith("【用户】\n第一条"), three
    assert two.startswith("【AI】\n好的"), "what the AI said opens a round of its own"
    assert "x" * 100 not in three, "tool results must enter only through the round digest"


def test_a_round_shows_its_tool_work_once() -> None:
    timeline = [TIMELINE[0], _call("c1"), _result("c1"), _call("c2"), _result("c2")]
    text = render_items(timeline, DIGESTS)

    assert text == "【用户】\n第一条\n\n【AI 的工具操作】\n概括u1", text


def test_a_result_in_a_later_round_opens_no_entry() -> None:
    timeline = [TIMELINE[0], _call("c1"), _user("u9", "排队的消息"), _result("c1")]

    assert render_items(timeline, DIGESTS).endswith("【用户】\n排队的消息"), (
        "no digest needed for u9"
    )


def test_falls_back_to_one_round() -> None:
    two = len(window_text(TIMELINE, 7, DIGESTS, 10_000, len, 2))
    text = window_text(TIMELINE, 7, DIGESTS, two - 1, len, 2)

    assert text.startswith("【用户】\n第二条"), text


def test_one_round_too_long_raises() -> None:
    with pytest.raises(WindowError, match="needs_review"):
        window_text(TIMELINE, 7, DIGESTS, 5, len, 2)


def test_missing_round_digest_names_the_fix() -> None:
    with pytest.raises(ValueError, match=r"run intent_aligner\.compress"):
        render_items([_user("u9", "新"), _call("c9")], DIGESTS)


def _many_calls(n: int) -> tuple[list[Item], Digests]:
    ids = [f"c{i}" for i in range(n)]
    return [_call(i) for i in ids], Digests(calls={i: _call_digest(i, f"意图{i}") for i in ids})


def test_following_starts_after_the_unit_and_stops_at_the_item_limit() -> None:
    items, digests = _many_calls(FOLLOWING_ITEMS + 3)

    text = render_following(items, 0, digests, len)

    assert "意图c0" not in text, text
    assert f"意图c{FOLLOWING_ITEMS} " in text, text
    assert f"意图c{FOLLOWING_ITEMS + 1} " not in text, text


def test_following_keeps_whole_entries_within_its_budget() -> None:
    items, digests = _many_calls(4)

    text = render_following(
        items, 0, digests, lambda entry: FOLLOWING_BUDGET // 2 + 1 if entry else 0
    )

    assert "意图c1 " in text and "意图c2 " not in text, text
