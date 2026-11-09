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
    ResultDigest,
    ToolCompression,
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
    calls={c: _call_digest(c, f"意图{c}", "t") for c in ("c1", "c2", "c3")},
    results={c: ResultDigest(tool_use_id=c, summaries=[f"结果{c}"]) for c in ("c1", "c2", "c3")},
)


def test_the_rounds_parameter_sets_how_far_back_the_window_reaches() -> None:
    three = window_text(TIMELINE, 7, 7, DIGESTS, [], 10_000, len, 3)
    two = window_text(TIMELINE, 7, 7, DIGESTS, [], 10_000, len, 2)

    assert three.startswith("【用户】\n第一条"), three
    assert two.startswith("【AI】\n好的"), "what the AI said opens a round of its own"
    assert "x" * 100 not in three, "tool results must enter only through their digests"


def test_tool_records_show_one_by_one_in_place() -> None:
    timeline: list[Item] = [
        TIMELINE[0],
        _call("c1"),
        _result("c1"),
        _user("u9", "排队"),
        _result("c2"),
    ]
    text = render_items(timeline, 0, len(timeline), DIGESTS, [])

    expected = (
        "【用户】\n第一条\n\n【AI 的工具调用】\nBash 意图：意图c1 对象：t\n\n"
        "【工具结果】\n结果c1\n\n【用户】\n排队\n\n【工具结果】\n结果c2"
    )
    assert text == expected, f"{text!r} vs {expected!r}"


def test_a_passage_shows_once_and_replaces_only_tool_records() -> None:
    timeline: list[Item] = [
        _user("u1", "第一条"),
        _call("c1"),
        _user("u9", "排队"),
        _result("c1"),
        _call("c2"),
    ]
    passage = ToolCompression(begin=1, through=4, summary="概括")
    text = render_items(timeline, 0, len(timeline), DIGESTS, [passage])

    expected = (
        "【用户】\n第一条\n\n【AI 此前的工具操作（已压缩）】\n概括\n\n【用户】\n排队\n\n"
        "【AI 的工具调用】\nBash 意图：意图c2 对象：t"
    )
    assert text == expected, f"{text!r} vs {expected!r}"


def test_a_window_over_its_room_raises() -> None:
    with pytest.raises(WindowError, match="needs_review"):
        window_text(TIMELINE, 7, 7, DIGESTS, [], 5, len, 2)


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
