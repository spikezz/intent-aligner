"""Isolation of the LLM wrapper.

The live tests call the real CLIs and cost tokens, so they only run with
INTENT_ALIGNER_LIVE=1; CI runs the offline ones.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from intent_aligner import llm
from intent_aligner.llm import (
    LLMCall,
    LLMError,
    build_invocation,
    complete,
    hive_client,
    hive_request_body,
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _call(backend: llm.Backend, model: str = "m", schema: bool = False) -> LLMCall:
    return LLMCall(
        backend=backend,
        model=model,
        effort="low",
        system_prompt="You are a terse calculator.",
        prompt="What is 2+2? Put it in answer.",
        json_schema=SCHEMA if schema else None,
    )


@pytest.fixture
def credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "credentials.json"
    expires = int((time.time() + 3600) * 1000)
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": "tok", "expiresAt": expires}}))
    monkeypatch.setattr(llm, "_CLAUDE_CREDENTIALS", path)
    return path


def test_claude_uses_isolated_config_and_borrowed_token(tmp_path: Path, credentials: Path) -> None:
    inv = build_invocation(_call("claude", schema=True), tmp_path)

    assert inv.env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude"), inv.env
    assert inv.env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok", inv.env
    assert inv.cwd == tmp_path / "work", inv.cwd
    assert json.loads(inv.argv[inv.argv.index("--json-schema") + 1]) == SCHEMA, inv.argv
    assert inv.argv[inv.argv.index("--effort") + 1] == "low", inv.argv
    assert inv.argv[inv.argv.index("--thinking-display") + 1] == "summarized", inv.argv
    assert not (tmp_path / "claude" / ".credentials.json").exists()


def test_claude_refuses_an_expired_token(tmp_path: Path, credentials: Path) -> None:
    credentials.write_text(json.dumps({"claudeAiOauth": {"accessToken": "t", "expiresAt": 0}}))

    with pytest.raises(LLMError, match="expired"):
        build_invocation(_call("claude"), tmp_path)


def test_codex_is_ephemeral_and_ignores_user_config(tmp_path: Path) -> None:
    inv = build_invocation(_call("codex", schema=True), tmp_path)

    for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules"):
        assert flag in inv.argv, inv.argv
    assert 'model_reasoning_effort="low"' in inv.argv, inv.argv
    schema_file = Path(inv.argv[inv.argv.index("--output-schema") + 1])
    assert schema_file.is_relative_to(tmp_path), schema_file
    assert json.loads(schema_file.read_text()) == SCHEMA
    assert inv.argv[-1] == "-", "prompt must come from stdin"


def test_codex_instructions_file_is_content_addressed(tmp_path: Path) -> None:
    first = build_invocation(_call("codex"), tmp_path).argv
    again = build_invocation(_call("codex"), tmp_path).argv

    assert first == again, (first, again)
    assert len(list((tmp_path / "codex" / "inputs").iterdir())) == 1


def test_pi_sessions_go_to_home(tmp_path: Path) -> None:
    inv = build_invocation(_call("pi"), tmp_path)

    assert inv.argv[inv.argv.index("--session-dir") + 1] == str(tmp_path / "pi"), inv.argv
    assert inv.argv[inv.argv.index("--thinking") + 1] == "low", inv.argv
    for flag in ("--no-tools", "--no-context-files", "--no-extensions", "--no-skills"):
        assert flag in inv.argv, inv.argv


def test_pi_rejects_a_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="structured-output"):
        complete(_call("pi", schema=True), tmp_path)


def test_empty_system_prompt_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMCall(backend="pi", model="m", effort="low", system_prompt="", prompt="p")


def test_main_sends_transcripts_to_the_home_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "system.txt"
    system.write_text("s")
    homes: list[Path] = []

    def fake_complete(call: LLMCall, home: Path) -> llm.LLMReply:
        homes.append(home)
        return llm.LLMReply(text="ok", session_id="id", transcript=home / "t.jsonl")

    monkeypatch.setattr(llm, "complete", fake_complete)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("p"))
    argv = ["llm", "--backend", "pi", "--model", "m", "--system", str(system), "--home", "rel"]
    monkeypatch.setattr("sys.argv", argv)
    llm.main()

    assert homes == [tmp_path / "rel"], "a relative --home must be resolved before the call"


QWEN_CLIENT = {
    "reasoning": True,
    "thinking_format": "qwen-chat-template",
    "thinking_token_budget_field": "reasoning_budget_tokens",
}
FLASH_CLIENT = {"reasoning": True, "thinking_format": "openai", "reasoning_effort": True}


def _hive_call(effort: llm.Effort = "low", schema: bool = False) -> LLMCall:
    return _call("hive", "deployment/d", schema).model_copy(update={"effort": effort})


def test_hive_body_is_only_system_and_user() -> None:
    call = _hive_call().model_copy(update={"stream": False})
    body = hive_request_body(call, {"reasoning": False})

    assert body == {
        "model": "deployment/d",
        "messages": [
            {"role": "system", "content": "You are a terse calculator."},
            {"role": "user", "content": "What is 2+2? Put it in answer."},
        ],
    }, body


def test_hive_streams_by_default() -> None:
    body = hive_request_body(_hive_call(), {"reasoning": False})

    assert body["stream"] is True, body
    assert body["stream_options"] == {"include_usage": True}, body


def test_hive_qwen_thinking_budget_is_the_same_at_every_effort() -> None:
    for effort in get_args(llm.Effort):
        body = hive_request_body(_hive_call(effort), QWEN_CLIENT)

        assert body["chat_template_kwargs"] == {"enable_thinking": True}, body
        assert body["reasoning_budget_tokens"] == llm.HIVE_THINKING_BUDGET, body


def test_hive_openai_thinking_sends_reasoning_effort() -> None:
    body = hive_request_body(_hive_call("xhigh"), FLASH_CLIENT)

    assert body["reasoning_effort"] == "xhigh", body
    assert "chat_template_kwargs" not in body, body


def test_hive_schema_becomes_response_format() -> None:
    body = hive_request_body(_hive_call(schema=True), QWEN_CLIENT)

    assert body["response_format"]["json_schema"]["schema"] == SCHEMA, body


def test_hive_schema_is_also_in_the_system_prompt() -> None:
    body = hive_request_body(_hive_call(schema=True), QWEN_CLIENT)
    system = body["messages"][0]["content"]

    assert system.startswith("You are a terse calculator."), system
    assert f"{llm.HIVE_SCHEMA_INTRO}\n{json.dumps(SCHEMA, ensure_ascii=False)}" in system, system


def _listing(*clients: dict[str, object]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {
        f"deployment/d{n}": {"id": f"deployment/d{n}", "hive": {"client": c}}
        for n, c in enumerate(clients)
    }
    members = [{"deployment": f"d{n}"} for n in range(len(clients))]
    entries["pool"] = {"id": "pool", "instances": members}
    return entries


def test_a_role_takes_its_members_shared_client() -> None:
    renamed = {**QWEN_CLIENT, "name": "another card"}

    got = hive_client(_listing(QWEN_CLIENT, renamed), "pool")

    assert {k: v for k, v in got.items() if k != "name"} == {
        k: v for k, v in QWEN_CLIENT.items() if k != "name"
    }, got


def test_a_role_whose_members_differ_in_protocol_is_refused() -> None:
    unbudgeted = {k: v for k, v in QWEN_CLIENT.items() if k != "thinking_token_budget_field"}

    with pytest.raises(LLMError, match="different client protocols"):
        hive_client(_listing(QWEN_CLIENT, unbudgeted), "pool")


class _Gateway:
    """A stand-in Hive gateway: one deployment, a canned chat reply, recorded requests."""

    def __init__(self) -> None:
        self.models: dict[str, Any] = {
            "data": [
                {"id": "chat"},
                {"id": "deployment/d", "hive": {"client": QWEN_CLIENT}},
            ]
        }
        self.reply: tuple[int, dict[str, Any]] = (
            200,
            {"choices": [{"finish_reason": "stop", "message": {"content": '{"answer": 4}'}}]},
        )
        self.requests: list[tuple[dict[str, str], dict[str, Any]]] = []


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Gateway]:
    state = _Gateway()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._answer(200, state.models)

        def _stream(self, body: dict[str, Any]) -> None:
            choice = body["choices"][0]
            chunk = {
                "choices": [
                    {"delta": choice["message"], "finish_reason": choice["finish_reason"]},
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((dict(self.headers), body))
            status, reply = state.reply
            if status == 200 and body.get("stream"):
                self._stream(reply)
            else:
                self._answer(status, reply)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm, "HIVE_FRAME_URL", f"http://127.0.0.1:{server.server_port}")
    yield state
    server.shutdown()


def test_hive_call_records_request_and_response(tmp_path: Path, gateway: _Gateway) -> None:
    reply = complete(_hive_call(schema=True), tmp_path)

    assert json.loads(reply.text) == {"answer": 4}, reply.text
    assert reply.transcript == tmp_path / "hive" / f"{reply.session_id}.jsonl", reply.transcript
    request, response = (json.loads(x) for x in reply.transcript.read_text().splitlines())
    headers, sent = gateway.requests[0]
    assert request["body"] == sent, (request, sent)
    assert response["body"] == gateway.reply[1], response
    assert "x-hive-lease" not in {k.lower() for k in headers}, "one-shot calls hold no lease"


def test_hive_block_reply_needs_no_assembly(tmp_path: Path, gateway: _Gateway) -> None:
    call = _hive_call(schema=True).model_copy(update={"stream": False})

    reply = complete(call, tmp_path)

    assert json.loads(reply.text) == {"answer": 4}, reply.text
    assert "stream" not in gateway.requests[0][1], gateway.requests[0][1]


def test_a_stream_is_assembled_into_one_completion() -> None:
    lines = [
        b'data: {"choices": [{"delta": {"reasoning_content": "hm"}, "finish_reason": null}]}\n',
        b"\n",
        b'data: {"choices": [{"delta": {"content": "{\\"a\\""}, "finish_reason": null}]}\n',
        b'data: {"choices": [{"delta": {"content": ": 1}"}, "finish_reason": "stop"}]}\n',
        b'data: {"choices": [], "usage": {"completion_tokens": 3}}\n',
        b"data: [DONE]\n",
    ]

    assert llm._assemble_stream(lines) == {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"a": 1}', "reasoning_content": "hm"},
            }
        ],
        "usage": {"completion_tokens": 3},
    }


def test_a_stream_cut_off_is_an_incomplete_reply() -> None:
    lines = [b'data: {"choices": [{"delta": {"content": "{"}, "finish_reason": null}]}\n']

    assert llm._assemble_stream(lines)["choices"][0]["finish_reason"] is None


def test_hive_rejects_a_name_that_is_neither_deployment_nor_role(
    tmp_path: Path, gateway: _Gateway
) -> None:
    with pytest.raises(LLMError, match=r"Running now: \['deployment/d'\]"):
        complete(_call("hive", "no-such-pool"), tmp_path)


def test_hive_truncated_reply_is_an_error_but_still_recorded(
    tmp_path: Path, gateway: _Gateway
) -> None:
    gateway.reply = (200, {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]})

    with pytest.raises(LLMError, match="'length'"):
        complete(_hive_call(), tmp_path)
    assert len(list((tmp_path / "hive").iterdir())) == 1


def test_hive_gateway_refusal_is_recorded(
    tmp_path: Path, gateway: _Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "_HIVE_CAPACITY_WAIT", 0)
    gateway.reply = (409, {"detail": "No verified compatible capacity is available"})

    with pytest.raises(LLMError, match="409"):
        complete(_hive_call(), tmp_path)
    (transcript,) = (tmp_path / "hive").iterdir()
    assert "No verified compatible capacity" in transcript.read_text(), transcript


def test_hive_waits_out_a_full_gateway(
    tmp_path: Path, gateway: _Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = gateway.reply
    gateway.reply = (409, {"error": "No verified compatible capacity is available"})

    def free_a_slot(_seconds: float) -> None:
        gateway.reply = answer

    monkeypatch.setattr("time.sleep", free_a_slot)
    reply = complete(_hive_call(), tmp_path)

    assert reply.text == '{"answer": 4}', reply
    assert len(gateway.requests) == 2, gateway.requests


LIVE_MODELS: dict[llm.Backend, str] = {
    "claude": "sonnet",
    "codex": "gpt-6-astra",
    "pi": "fireworks/accounts/fireworks/models/glm-5p3-flash",
    "hive": "deployment/pc-qwen38-np6",
}

# Where each CLI keeps transcripts in daily use; an isolated call must leave no trace there.
# hive has no CLI and so no daily transcripts.
DAILY_TRANSCRIPTS: dict[llm.Backend, Path] = {
    "claude": Path.home() / ".claude" / "projects",
    "codex": Path.home() / ".codex" / "sessions",
    "pi": Path.home() / ".pi" / "agent" / "sessions",
}


@pytest.mark.skipif(
    os.environ.get("INTENT_ALIGNER_LIVE") != "1", reason="set INTENT_ALIGNER_LIVE=1"
)
@pytest.mark.parametrize("backend", ["claude", "codex", "pi", "hive"])
def test_live_call_stays_isolated(backend: llm.Backend, tmp_path: Path) -> None:
    reply = complete(_call(backend, LIVE_MODELS[backend], schema=backend != "pi"), tmp_path)

    assert reply.transcript.is_relative_to(tmp_path), reply.transcript
    assert reply.session_id in reply.transcript.read_text(), reply.transcript
    if backend == "pi":
        assert "4" in reply.text, reply.text
    else:
        assert json.loads(reply.text) == {"answer": 4}, reply.text
    if backend == "hive":
        return
    leaked = [p for p in DAILY_TRANSCRIPTS[backend].rglob("*.jsonl") if reply.session_id in p.name]
    assert not leaked, f"{backend} wrote a daily transcript: {leaked}"


def test_asker_logs_every_call_with_its_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies: Iterator[llm.LLMReply | LLMError] = iter(
        [
            llm.LLMReply(text="{}", session_id="s1", transcript=tmp_path / "s1.jsonl"),
            LLMError("truncated", tmp_path / "s2.jsonl"),
        ]
    )

    def fake_complete(call: LLMCall, home: Path = llm.LLM_HOME) -> llm.LLMReply:
        reply = next(replies)
        if isinstance(reply, LLMError):
            raise reply
        return reply

    monkeypatch.setattr(llm, "complete", fake_complete)
    log = llm.CallLog(path=tmp_path / llm.CALLS_FILE)
    ask = llm.asker("hive", "m", "xhigh", "system", {}, True, log, "topic")
    llm.CALL_SUBJECT.set("unit:3")
    assert ask("p") == "{}"
    with pytest.raises(LLMError):
        ask("p")
    got = [llm.CallRecord.model_validate_json(line) for line in log.path.read_text().splitlines()]
    assert [(r.ask, r.subject, r.transcript, r.error) for r in got] == [
        ("topic", "unit:3", str(tmp_path / "s1.jsonl"), None),
        ("topic", "unit:3", str(tmp_path / "s2.jsonl"), "truncated"),
    ], got
