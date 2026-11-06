"""One-shot LLM calls through the Claude Code, Codex and pi CLIs, isolated from daily use.

Usage: python -m intent_aligner.llm --backend B --model M --system FILE
           [--effort E] [--schema FILE] < prompt.txt

Every transcript a call produces lands under LLM_HOME and nowhere else:
    llm-home/claude/projects/<cwd slug>/<session>.jsonl   CLAUDE_CONFIG_DIR of the call
    llm-home/codex/<thread>.jsonl                         request + `codex exec --json` events
    llm-home/pi/<time>_<session>.jsonl                    `pi --session-dir`
    llm-home/work/                                        empty cwd shared by all calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import attrs
from pydantic import BaseModel, ConfigDict, Field

from intent_aligner.corpus import TOOL_ROOT

LLM_HOME = TOOL_ROOT / "llm-home"

Backend = Literal["claude", "codex", "pi"]
Effort = Literal["low", "medium", "high", "xhigh"]

_CLAUDE_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"


class LLMCall(BaseModel):
    """One stateless request; `model` is whatever the backend CLI accepts.

    `system_prompt` replaces the CLI's own. It must be non-empty: how each CLI treats
    an empty value (pi documents a coding-assistant default) is unverified.

    pi takes "provider/model-id". `json_schema` constrains the reply on claude and
    codex; pi has no structured-output mode, so it rejects a schema.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend
    model: str
    effort: Effort
    system_prompt: str = Field(min_length=1)
    prompt: str
    json_schema: dict[str, Any] | None = None


class LLMReply(BaseModel):
    """The final assistant text and the transcript file the call left in LLM_HOME.

    With a json_schema, `text` is the JSON document that satisfies it.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    session_id: str
    transcript: Path


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Invocation:
    """Everything subprocess.run needs; `env` is merged over os.environ."""

    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path


class LLMError(RuntimeError):
    """The backend CLI failed or returned no usable reply."""


def complete(call: LLMCall, home: Path = LLM_HOME) -> LLMReply:
    """Run `call` once and return its reply; raises LLMError on any backend failure."""
    if call.backend == "pi" and call.json_schema is not None:
        raise ValueError("pi has no structured-output mode; drop json_schema or use claude/codex.")
    (home / "work").mkdir(parents=True, exist_ok=True)
    invocation = build_invocation(call, home)
    proc = subprocess.run(
        invocation.argv,
        input=call.prompt,
        capture_output=True,
        text=True,
        cwd=invocation.cwd,
        env={**os.environ, **invocation.env},
        check=False,
    )
    if proc.returncode != 0:
        raise LLMError(
            f"{call.backend} exited with {proc.returncode}. stderr tail:\n{proc.stderr[-2000:]}"
        )
    if call.backend == "claude":
        return _claude_reply(proc.stdout, home)
    if call.backend == "codex":
        return _codex_reply(call, proc.stdout, home)
    return _pi_reply(proc.stdout, home)


def build_invocation(call: LLMCall, home: Path) -> Invocation:
    """Command line, environment and cwd that isolate `call` inside `home`."""
    work = home / "work"
    if call.backend == "claude":
        argv = [
            "claude", "-p",
            "--model", call.model,
            "--effort", call.effort,
            "--system-prompt", call.system_prompt,
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "user",
            "--output-format", "json",
        ]  # fmt: skip
        if call.json_schema is not None:
            argv += ["--json-schema", json.dumps(call.json_schema)]
        env = {
            "CLAUDE_CONFIG_DIR": str(home / "claude"),
            "CLAUDE_CODE_OAUTH_TOKEN": _claude_access_token(),
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        }
        return Invocation(argv=tuple(argv), env=env, cwd=work)

    if call.backend == "codex":
        argv = [
            "codex", "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--model", call.model,
            "-c", f"model_reasoning_effort={json.dumps(call.effort)}",
            "-c", f"model_instructions_file={json.dumps(str(_codex_file(home, call.system_prompt, 'md')))}",
            "--json",
        ]  # fmt: skip
        if call.json_schema is not None:
            schema = _codex_file(home, json.dumps(call.json_schema, sort_keys=True), "schema.json")
            argv += ["--output-schema", str(schema)]
        argv.append("-")
        return Invocation(argv=tuple(argv), env={}, cwd=work)

    argv = [
        "pi", "-p",
        "--mode", "json",
        "--model", call.model,
        "--thinking", call.effort,
        "--system-prompt", call.system_prompt,
        "--session-dir", str(home / "pi"),
        "--no-tools",
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--offline",
    ]  # fmt: skip
    return Invocation(argv=tuple(argv), env={}, cwd=work)


def _claude_access_token() -> str:
    """Read, never write, the daily login.

    The isolated config dir holds no credentials file: a copy there would refresh on
    its own and rotate the refresh token out from under the daily sessions. The call
    borrows the current access token instead; the daily sessions keep refreshing it.
    """
    oauth = json.loads(_CLAUDE_CREDENTIALS.read_text())["claudeAiOauth"]
    if oauth["expiresAt"] / 1000 < time.time() + 60:
        raise LLMError(
            f"The Claude access token in {_CLAUDE_CREDENTIALS} has expired. Run any "
            "interactive `claude` command once so the daily login refreshes it."
        )
    return str(oauth["accessToken"])


def _codex_file(home: Path, content: str, suffix: str) -> Path:
    """Content-addressed, so concurrent calls never overwrite each other's file."""
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    path = home / "codex" / "inputs" / f"{digest}.{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return path


def _claude_reply(stdout: str, home: Path) -> LLMReply:
    result: dict[str, Any] = json.loads(stdout)
    if result.get("is_error"):
        raise LLMError(f"claude reported an error: {result.get('result')!r}")
    session_id = str(result["session_id"])
    structured = result.get("structured_output")
    text = (
        json.dumps(structured, ensure_ascii=False) if structured is not None else result["result"]
    )
    return LLMReply(
        text=text,
        session_id=session_id,
        transcript=_single(home / "claude" / "projects", f"*/{session_id}.jsonl"),
    )


def _codex_reply(call: LLMCall, stdout: str, home: Path) -> LLMReply:
    """Codex runs --ephemeral, so this record is the only transcript of the call."""
    events = [cast(dict[str, Any], json.loads(line)) for line in stdout.splitlines() if line]
    thread_id = next(str(e["thread_id"]) for e in events if e["type"] == "thread.started")
    failures = [e for e in events if e["type"] in ("error", "turn.failed")]
    messages = [
        e["item"]["text"]
        for e in events
        if e["type"] == "item.completed" and e["item"]["type"] == "agent_message"
    ]
    transcript = home / "codex" / f"{thread_id}.jsonl"
    with transcript.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "request", **call.model_dump()}, ensure_ascii=False) + "\n")
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    if failures or not messages:
        raise LLMError(f"codex returned no reply; see {transcript}. Failures: {failures!r}")
    return LLMReply(text=str(messages[-1]), session_id=thread_id, transcript=transcript)


def _pi_reply(stdout: str, home: Path) -> LLMReply:
    events = [cast(dict[str, Any], json.loads(line)) for line in stdout.splitlines() if line]
    session_id = next(str(e["id"]) for e in events if e["type"] == "session")
    transcript = _single(home / "pi", f"*_{session_id}.jsonl")
    final = next(e for e in reversed(events) if e["type"] == "agent_end")["messages"][-1]
    if final["role"] != "assistant" or final["stopReason"] != "stop":
        raise LLMError(
            f"pi stopped with {final.get('stopReason')!r}: {final.get('errorMessage')!r}; "
            f"see {transcript}."
        )
    text = "".join(part["text"] for part in final["content"] if part["type"] == "text")
    return LLMReply(text=text, session_id=session_id, transcript=transcript)


def _single(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise LLMError(f"Expected one transcript {root / pattern}, found {matches}.")
    return matches[0]


def main() -> None:
    """CLI entry point: prompt on stdin, reply text on stdout, transcript path on stderr."""
    parser = argparse.ArgumentParser(description="One isolated LLM call.")
    parser.add_argument("--backend", choices=["claude", "codex", "pi"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="xhigh")
    parser.add_argument("--system", type=Path, required=True, help="file holding the system prompt")
    parser.add_argument("--schema", type=Path, help="file holding a JSON schema for the reply")
    args = parser.parse_args()
    call = LLMCall(
        backend=args.backend,
        model=args.model,
        effort=args.effort,
        system_prompt=args.system.read_text(),
        prompt=sys.stdin.read(),
        json_schema=json.loads(args.schema.read_text()) if args.schema else None,
    )
    reply = complete(call)
    print(reply.text)
    print(f"transcript: {reply.transcript}", file=sys.stderr)


if __name__ == "__main__":
    main()
