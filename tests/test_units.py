"""The unit cutters: error paths offline, record coverage on the built corpus."""

from __future__ import annotations

import doctest
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from intent_aligner import units
from intent_aligner.corpus import TOOL_ROOT
from intent_aligner.units import (
    Observation,
    ToolCall,
    UserMessage,
    UserSignal,
    claude_units,
    codex_units,
    conversation_units,
)

MERGED = sorted((TOOL_ROOT / "corpus" / "merged").glob("claude-*.jsonl"))
CODEX_MERGED = sorted((TOOL_ROOT / "corpus" / "merged").glob("codex-*.jsonl"))
needs_corpus = pytest.mark.skipif(not MERGED, reason="corpus not built")


def test_docstring_examples_hold() -> None:
    result = doctest.testmod(units)

    assert result.failed == 0 and result.attempted > 0, result


def _write(tmp_path: Path, *records: dict[str, Any], name: str = "merged.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_unknown_record_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown record type 'brand-new'"):
        list(claude_units(_write(tmp_path, {"type": "brand-new"})))


def test_image_source_without_a_message_raises(tmp_path: Path) -> None:
    orphan = {
        "type": "user",
        "isMeta": True,
        "uuid": "u",
        "message": {"content": [{"type": "text", "text": "[Image: source: /a.png]"}]},
    }

    with pytest.raises(ValueError, match="follows no user message"):
        list(claude_units(_write(tmp_path, orphan)))


@pytest.fixture(scope="module", params=MERGED, ids=lambda p: p.name)
def cut(request: pytest.FixtureRequest) -> tuple[list[dict[str, Any]], list[units.Item]]:
    path: Path = request.param
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records, list(claude_units(path))


@needs_corpus
def test_every_human_message_and_tool_block_appears_once(
    cut: tuple[list[dict[str, Any]], list[units.Item]],
) -> None:
    records, items = cut
    human = sum((r.get("origin") or {}).get("kind") == "human" for r in records)
    queued = sum(
        r["type"] == "attachment"
        and r["attachment"]["type"] == "queued_command"
        and (r["attachment"].get("origin") or {}).get("kind") == "human"
        for r in records
    )
    blocks = Counter(
        b["type"]
        for r in records
        if r["type"] in ("user", "assistant") and isinstance(r["message"]["content"], list)
        for b in r["message"]["content"]
    )
    kinds = Counter(type(item).__name__ for item in items)
    rejections = sum(isinstance(x, UserSignal) and x.signal == "rejection" for x in items)
    results = sum(isinstance(x, Observation) and x.origin == "tool_result" for x in items)

    messages = [x for x in items if isinstance(x, UserMessage)]
    assert (len(messages), sum(x.queued for x in messages)) == (human + queued, queued), kinds
    assert kinds["ToolCall"] == blocks["tool_use"], (kinds, blocks)
    assert rejections + results == blocks["tool_result"], (rejections, results, blocks)


@pytest.mark.parametrize("path", MERGED + CODEX_MERGED, ids=lambda p: p.name)
def test_tool_results_follow_their_calls(path: Path) -> None:
    seen: set[str] = set()
    for item in conversation_units(path):
        if isinstance(item, ToolCall):
            seen.add(item.tool_use_id)
        elif isinstance(item, Observation | UserSignal) and item.tool_use_id is not None:
            assert item.tool_use_id in seen, f"result before its call: {item.source}"


def test_unknown_platform_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="names no platform"):
        conversation_units(_write(tmp_path, name="pi-1.jsonl"))


def _codex(kind: str, payload: dict[str, Any], line: int) -> dict[str, Any]:
    return {
        "timestamp": "2026-09-19T19:47:51.691Z",
        "ordinal": line,
        "type": kind,
        "payload": payload,
        "_merge": {"session": "s"},
    }


def _codex_user(text: str, line: int) -> dict[str, Any]:
    message = {"type": "UserMessage", "id": f"m{line}", "content": [{"type": "text", "text": text}]}
    return _codex("event_msg", {"type": "item_completed", "item": message}, line)


def test_codex_unknown_event_raises(tmp_path: Path) -> None:
    record = _codex("event_msg", {"type": "brand_new"}, 0)

    with pytest.raises(ValueError, match="unknown event type 'brand_new'"):
        list(codex_units(_write(tmp_path, record)))


def test_codex_question_answer_replaces_the_acknowledgement(tmp_path: Path) -> None:
    call = {
        "type": "function_call",
        "id": "fc",
        "call_id": "c1",
        "name": "request_user_input_async",
        "arguments": '{"questions": []}',
    }
    ack = {
        "type": "function_call_output",
        "id": "o",
        "call_id": "c1",
        "output": '{"accepted":true}',
    }
    answer = {
        "question": "推吗？",
        "answer": "推",
        "questionItemId": '["request_user_input_async","c1",0]',
    }
    reply = (
        "<send_user_message_question_reply>\n"
        + json.dumps([answer], ensure_ascii=False)
        + "\n</send_user_message_question_reply>"
    )
    path = _write(
        tmp_path,
        _codex("response_item", call, 0),
        _codex("response_item", ack, 1),
        _codex_user(reply, 2),
    )

    items = list(codex_units(path))

    assert [type(x).__name__ for x in items] == ["ToolCall", "Observation"], items
    result = items[1]
    assert isinstance(result, Observation), result
    assert (result.tool_use_id, result.text) == ("c1", "问题：推吗？\n回答：推"), result


def test_codex_goal_status_change_is_not_a_message(tmp_path: Path) -> None:
    def goal(status: str, line: int) -> dict[str, Any]:
        state = {"threadId": "t", "objective": "整理项目", "status": status}
        return _codex("event_msg", {"type": "thread_goal_updated", "goal": state}, line)

    items = list(
        codex_units(_write(tmp_path, goal("active", 0), goal("paused", 1), goal("active", 2)))
    )

    assert [(type(x).__name__, getattr(x, "body", None)) for x in items] == [
        ("UserMessage", "整理项目")
    ], items


def test_codex_message_after_ai_action_is_queued(tmp_path: Path) -> None:
    started = _codex("event_msg", {"type": "task_started"}, 0)
    said = {
        "type": "message",
        "id": "a",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "好"}],
    }
    path = _write(
        tmp_path,
        started,
        _codex_user("先看", 1),
        _codex("response_item", said, 2),
        _codex_user("顺便", 3),
    )

    queued = [x.queued for x in codex_units(path) if isinstance(x, UserMessage)]

    assert queued == [False, True], queued


def _codex_expected(path: Path) -> Counter[str]:
    """Item counts derived from the raw records, independently of the cutter's branches."""
    expected: Counter[str] = Counter()
    question_calls: set[str] = set()
    objectives: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            payload = record.get("payload", {})
            kind = payload.get("type")
            if kind == "item_completed" and payload["item"]["type"] == "UserMessage":
                text = next(b["text"] for b in payload["item"]["content"] if b["type"] == "text")
                reply = text.startswith("<send_user_message_question_reply>")
                expected["Observation" if reply else "UserMessage"] += 1
            elif kind == "thread_goal_updated":
                goal = payload["goal"]
                if objectives.get(goal["threadId"]) != goal["objective"]:
                    objectives[goal["threadId"]] = goal["objective"]
                    expected["UserMessage"] += 1
            elif kind == "turn_aborted":
                expected["UserSignal"] += 1
            elif record["type"] == "response_item" and kind in (
                "custom_tool_call",
                "function_call",
                "web_search_call",
            ):
                expected["ToolCall"] += 1
                if payload.get("name") == "request_user_input_async":
                    question_calls.add(payload["call_id"])
            elif kind in ("custom_tool_call_output", "function_call_output"):
                ack = (
                    payload["call_id"] in question_calls
                    and payload["output"] == '{"accepted":true}'
                )
                expected["Observation"] += not ack
            elif kind == "message" and payload["role"] == "assistant":
                expected["AIText"] += 1
            elif kind == "message" and payload["role"] == "user":
                shell = payload["content"][0].get("text", "").startswith("<user_shell_command>")
                expected["Observation"] += shell
    return expected


@pytest.mark.parametrize("path", CODEX_MERGED, ids=lambda p: p.name)
def test_codex_every_message_call_and_output_appears_once(path: Path) -> None:
    expected = _codex_expected(path)

    kinds = Counter(type(item).__name__ for item in codex_units(path))

    assert kinds == expected, f"{kinds!r} vs {expected!r}"
