"""Isolation of the LLM wrapper.

The live tests call the real CLIs and cost tokens, so they only run with
INTENT_ALIGNER_LIVE=1; CI runs the offline ones.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from intent_aligner import llm
from intent_aligner.llm import LLMCall, LLMError, build_invocation, complete

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


LIVE_MODELS: dict[llm.Backend, str] = {
    "claude": "sonnet",
    "codex": "gpt-6-astra",
    "pi": "fireworks/accounts/fireworks/models/glm-5p3-flash",
}

# Where each CLI keeps transcripts in daily use; an isolated call must leave no trace there.
DAILY_TRANSCRIPTS: dict[llm.Backend, Path] = {
    "claude": Path.home() / ".claude" / "projects",
    "codex": Path.home() / ".codex" / "sessions",
    "pi": Path.home() / ".pi" / "agent" / "sessions",
}


@pytest.mark.skipif(
    os.environ.get("INTENT_ALIGNER_LIVE") != "1", reason="set INTENT_ALIGNER_LIVE=1"
)
@pytest.mark.parametrize("backend", ["claude", "codex", "pi"])
def test_live_call_stays_isolated(backend: llm.Backend, tmp_path: Path) -> None:
    reply = complete(_call(backend, LIVE_MODELS[backend], schema=backend != "pi"), tmp_path)

    assert reply.transcript.is_relative_to(tmp_path), reply.transcript
    assert reply.session_id in reply.transcript.read_text(), reply.transcript
    if backend == "pi":
        assert "4" in reply.text, reply.text
    else:
        assert json.loads(reply.text) == {"answer": 4}, reply.text
    leaked = [p for p in DAILY_TRANSCRIPTS[backend].rglob("*.jsonl") if reply.session_id in p.name]
    assert not leaked, f"{backend} wrote a daily transcript: {leaked}"
