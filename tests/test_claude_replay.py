"""Claude Code 2.1.318 native record shapes, including split blocks and prompt snapshots."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from intent_aligner import compress, replay
from intent_aligner.llm import CallRecord
from intent_aligner.replay import Conversation, Run, logged_calls, matched_calls, read_exchange
from intent_aligner.window import Digests


@pytest.fixture
def events() -> list[dict[str, Any]]:
    usage = {
        "input_tokens": 20,
        "cache_read_input_tokens": 30,
        "output_tokens": 12,
        "output_tokens_details": {"thinking_tokens": 10},
    }
    message = {"model": "claude-opus-5-5", "id": "msg_1", "stop_reason": "end_turn", "usage": usage}
    return [
        {"type": "queue-operation", "operation": "enqueue"},
        {"type": "user", "message": {"role": "user", "content": "What is 2 + 2?"}},
        {
            "type": "assistant",
            "effort": "high",
            "requestedModel": "opus",
            "message": {
                **message,
                "content": [
                    {"type": "thinking", "thinking": "Add two pairs.", "signature": "opaque"}
                ],
            },
        },
        {
            "type": "assistant",
            "effort": "high",
            "requestedModel": "opus",
            "message": {**message, "content": [{"type": "text", "text": "4"}]},
        },
        {
            "type": "attachment",
            "attachment": {
                "type": "prompt_snapshot",
                "systemPrompt": ["Return an integer."],
                "tools": [],
            },
        },
        {"type": "last-prompt", "lastPrompt": "What is 2 + 2?"},
    ]


def _write(path: Path, events: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return path


def _run(tmp_path: Path) -> Run:
    directory = tmp_path / "run"
    directory.mkdir()
    return Run(
        directory=directory,
        merged=directory / "merged.jsonl",
        backend="claude",
        model="opus",
        digests=directory / "digests.jsonl",
        segments=directory / "segments.jsonl",
        extract=directory / "extract",
        window_rounds=2,
    )


def test_native_thinking_text_request_and_usage(
    events: list[dict[str, Any]], tmp_path: Path
) -> None:
    path = _write(tmp_path / "session.jsonl", events)
    before = path.read_bytes()
    exchange = read_exchange(path)
    assert exchange is not None
    assert exchange.reasoning == "Add two pairs."
    assert exchange.reply == "4"
    assert exchange.system == "Return an integer."
    assert exchange.prompt == "What is 2 + 2?"
    assert exchange.request == {"backend": "claude", "model": "opus", "effort": "high"}
    assert exchange.usage == events[2]["message"]["usage"], (
        "split blocks must not double count usage"
    )
    assert exchange.finish == "stop"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "thinking",
    [
        [],
        [{"type": "thinking", "thinking": "", "signature": "opaque"}],
        [{"type": "redacted_thinking", "data": "opaque"}],
    ],
)
def test_missing_thinking_never_becomes_signature_text(
    thinking: list[dict[str, Any]], events: list[dict[str, Any]], tmp_path: Path
) -> None:
    events[2]["message"]["content"] = thinking
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None
    assert exchange.reasoning is None
    assert exchange.reply == "4"


def test_multiple_blocks_and_api_messages(events: list[dict[str, Any]], tmp_path: Path) -> None:
    events[2]["message"]["content"].append({"type": "thinking", "thinking": "Check the sum."})
    events[3]["message"]["id"] = "msg_2"
    events[3]["message"]["content"].append({"type": "text", "text": "Done."})
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None
    assert exchange.reasoning == "Add two pairs.\n\nCheck the sum."
    assert exchange.reply == "4\n\nDone."
    assert exchange.usage["input_tokens"] == 40
    assert exchange.usage["output_tokens_details"]["thinking_tokens"] == 20


def test_structured_output_overrides_commentary_and_ignores_tool_results(
    events: list[dict[str, Any]], tmp_path: Path
) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    events[4]["attachment"]["tools"] = [
        {"name": "StructuredOutput", "schema": {"input_schema": schema}}
    ]
    events[3]["message"]["content"] = [
        {"type": "text", "text": "Here is the result."},
        {"type": "tool_use", "name": "StructuredOutput", "input": {"answer": 4}},
    ]
    events[3]["message"]["stop_reason"] = "tool_use"
    events.append(
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}
    )
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None
    assert json.loads(exchange.reply or "") == {"answer": 4}
    assert exchange.schema == schema
    assert exchange.prompt == "What is 2 + 2?"
    assert exchange.finish == "stop"


def test_block_user_prompt_and_truncated_reply(
    events: list[dict[str, Any]], tmp_path: Path
) -> None:
    events[1]["message"]["content"] = [
        {"type": "text", "text": "First part."},
        {"type": "text", "text": "Second part."},
    ]
    events[3]["message"]["stop_reason"] = "max_tokens"
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None
    assert exchange.prompt == "First part.\n\nSecond part."
    assert exchange.finish == "length"


def test_sidechain_is_not_part_of_the_call(events: list[dict[str, Any]], tmp_path: Path) -> None:
    events.append(
        {
            "type": "assistant",
            "isSidechain": True,
            "message": {"content": [{"type": "thinking", "thinking": "unrelated"}]},
        }
    )
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None and exchange.reasoning == "Add two pairs."


def test_missing_snapshot_is_not_fabricated(events: list[dict[str, Any]], tmp_path: Path) -> None:
    events.pop(4)
    assert read_exchange(_write(tmp_path / "session.jsonl", events)) is None


def test_logged_claude_call_reaches_replay(events: list[dict[str, Any]], tmp_path: Path) -> None:
    path = _write(tmp_path / "claude" / "projects" / "cwd" / "session.jsonl", events)
    run = _run(tmp_path)
    now = datetime.now(UTC)
    record = CallRecord(ask="topic", subject="unit:0", started=now, ended=now, transcript=str(path))
    (run.directory / "calls.jsonl").write_text(record.model_dump_json() + "\n")
    calls = logged_calls(run)
    assert calls is not None and len(calls) == 1
    assert calls[0].exchange.reasoning == "Add two pairs."
    assert calls[0].subject == "unit:0"


def test_legacy_discovery_reads_claude_projects(
    events: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events[4]["attachment"]["systemPrompt"] = [compress.CALL_SYSTEM_PROMPT]
    path = _write(tmp_path / "claude" / "projects" / "cwd" / "session.jsonl", events)
    run = _run(tmp_path)
    monkeypatch.setattr(replay, "LLM_HOME", tmp_path)
    monkeypatch.setattr(replay._JobFinder, "jobs", lambda self, ask, prompt: ["tool:t"])
    conversation = Conversation(
        items=[],
        digests=Digests(),
        segments={},
        assignments=[],
        units=[],
        unit_items=[],
        round_of={},
        window_rounds=2,
    )
    calls, unplaced, _ = matched_calls(run, conversation, datetime(2020, 1, 1, tzinfo=UTC), None)
    assert len(calls) == 1 and calls[0].exchange.path == path
    assert unplaced == []


@pytest.mark.parametrize("thinking", [0, 10])
def test_claude_token_split_includes_cached_prompt(
    events: list[dict[str, Any]], tmp_path: Path, thinking: int
) -> None:
    events[2]["message"]["usage"]["cache_creation_input_tokens"] = 7
    events[2]["message"]["usage"]["output_tokens_details"]["thinking_tokens"] = thinking
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None
    tokens = replay._tokens(exchange)
    assert tokens is not None
    assert tokens.model_dump() == {"prompt": 57, "thinking": thinking, "answer": 12 - thinking}


def test_claude_unknown_thinking_count_remains_unknown(
    events: list[dict[str, Any]], tmp_path: Path
) -> None:
    del events[2]["message"]["usage"]["output_tokens_details"]
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None
    assert replay._tokens(exchange) is None
    assert exchange.usage["output_tokens"] == 12


@pytest.mark.parametrize("mixed", [False, True])
def test_claude_null_token_details_are_unknown(
    events: list[dict[str, Any]], tmp_path: Path, mixed: bool
) -> None:
    if mixed:
        events[3]["message"]["id"] = "msg_2"
        events[3]["message"]["usage"] = {
            **events[3]["message"]["usage"],
            "output_tokens_details": {"thinking_tokens": 3},
        }
    events[2]["message"]["usage"]["output_tokens_details"] = None
    exchange = read_exchange(_write(tmp_path / "session.jsonl", events))
    assert exchange is not None and exchange.reply == "4"
    assert "output_tokens_details" not in exchange.usage
    assert replay._tokens(exchange) is None
    assert exchange.usage["output_tokens"] == (24 if mixed else 12)
