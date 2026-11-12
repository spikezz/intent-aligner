"""One-shot LLM calls, isolated from daily use.

The claude, codex and pi backends run those CLIs. The hive backend talks HTTP straight
to the Hive Frame gateway (HIVE_FRAME_URL, default http://127.0.0.1:8790) with no agent
harness in between: the request carries only the system prompt and the user prompt.

Usage: python -m intent_aligner.llm --backend B --model M --system FILE
           [--effort E] [--schema FILE] [--home DIR] [--no-stream] < prompt.txt

Every transcript a call produces lands under the home directory (--home, default
LLM_HOME) and nowhere else:
    llm-home/claude/projects/<cwd slug>/<session>.jsonl   CLAUDE_CONFIG_DIR of the call
    llm-home/codex/<thread>.jsonl                         request + `codex exec --json` events
    llm-home/pi/<time>_<session>.jsonl                    `pi --session-dir`
    llm-home/hive/<call id>.jsonl                         request body + gateway response
    llm-home/work/                                        empty cwd shared by all calls
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import attrs
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tokenizers import Tokenizer

from intent_aligner.corpus import TOOL_ROOT

LLM_HOME = TOOL_ROOT / "llm-home"

Backend = Literal["claude", "codex", "pi", "hive"]
Effort = Literal["low", "medium", "high", "xhigh"]

_CLAUDE_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"

HIVE_FRAME_URL = os.environ.get("HIVE_FRAME_URL", "http://127.0.0.1:8790").rstrip("/")

# The one thinking cap for every Hive call at every effort, so calls of one run are
# comparable; window.PROMPT_BUDGET is the np6 slot minus this cap.
HIVE_THINKING_BUDGET = 12_288
HIVE_SCHEMA_INTRO = "回答是符合下面这个 JSON schema 的 JSON："
# A capped reply's thinking, counted as completion minus the answer under TOKENIZER, came
# to 10,231-10,278 on 64 capped Qwen3.8 calls (2026-11-10) and to at most 10,124 on the
# others, so this margin absorbs the local count of the answer.
HIVE_CAPPED_MARGIN = 64
# The gateway answered 409 to a burst right after a stopped run and accepted the same
# requests about 40 s later (2026-11-06), so a 409 is waited out before it counts as failure.
_HIVE_CAPACITY_WAIT = 1800
_HIVE_CAPACITY_POLL = 10


# Qwen3-0.6B shares the Qwen tokenizer family; on six corpus texts its counts were
# within 7% of the qwen38 server's /tokenize. On the largest prompts of
# claude-87eeb4bd-qwen it counted the user prompt alone as high as the server counted
# system and user prompt together, so it errs on the side of fitting.
TOKENIZER = "Qwen/Qwen3-0.6B"


@cache
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(TOKENIZER)


def qwen_tokens(text: str) -> int:
    """Token count of `text` under the Qwen tokenizer (downloaded once into the HF cache)."""
    return len(_tokenizer().encode(text).ids)


class LLMCall(BaseModel):
    """One stateless request; `model` is whatever the backend CLI accepts.

    `system_prompt` replaces the CLI's own. It must be non-empty: how each CLI treats
    an empty value (pi documents a coding-assistant default) is unverified.

    pi takes "provider/model-id"; hive takes a gateway id "deployment/<name>". `json_schema`
    constrains the reply on claude, codex and hive; pi has no structured-output mode, so
    it rejects a schema.

    `stream` only matters on hive, the one backend this module reads over HTTP. Streaming
    is the default because the gateway never frees the slot of a non-streaming request
    that outlives its read timeout (2026-11-06: 6 of 8 np8 slots lost that way); a stream
    only times out when the engine goes silent, not when a long answer takes long.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend
    model: str
    effort: Effort
    system_prompt: str = Field(min_length=1)
    prompt: str
    json_schema: dict[str, Any] | None = None
    stream: bool = True


class LLMReply(BaseModel):
    """The final assistant text and the transcript file the call left in LLM_HOME.

    With a json_schema, `text` is the JSON document that satisfies it. `thinking_capped` is
    whether the thinking reached HIVE_THINKING_BUDGET, which only a hive reply can tell.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    session_id: str
    transcript: Path
    thinking_capped: bool = False


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Invocation:
    """Everything subprocess.run needs; `env` is merged over os.environ."""

    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path


class LLMError(RuntimeError):
    """The backend CLI failed or returned no usable reply.

    `transcript` is the file the failed call left, when the backend got far enough to write one.
    """

    def __init__(self, message: str, transcript: Path | None = None) -> None:
        super().__init__(message)
        self.transcript = transcript


class TruncatedReply(LLMError):
    """The model ran out of room before finishing; callers count it as one failed attempt."""


class ThinkingCapped(LLMError):
    """The thinking reached HIVE_THINKING_BUDGET; `text` is the answer written after the cut."""

    def __init__(self, message: str, text: str, transcript: Path) -> None:
        super().__init__(message, transcript)
        self.text = text


# Calls one prompt gets whose thinking reaches the cap; then the last capped answer is
# kept and marked for a human's review, so one hard prompt does not stop the run.
CAPPED_ATTEMPTS = 3


def capped_review(error: ThinkingCapped) -> str:
    """The review reason of a kept answer whose thinking reached the cap."""
    return (
        f"no call gave a valid answer within the {HIVE_THINKING_BUDGET:,}-token thinking cap; "
        f"the last capped answer is kept, see {error.transcript}"
    )


TRUNCATED_FEEDBACK = (
    "你上一次的思考和回答超出了长度上限，没有写完。请直接给出答案，不要反复核对同一处细节。"
)


def complete(call: LLMCall, home: Path = LLM_HOME) -> LLMReply:
    """Run `call` once and return its reply; raises LLMError on any backend failure."""
    if call.backend == "pi" and call.json_schema is not None:
        raise ValueError("pi has no structured-output mode; drop json_schema or use claude/codex.")
    if call.backend == "hive":
        return _hive_complete(call, home)
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
        # claude -p reports API and auth failures in the stdout result JSON, not on stderr.
        raise LLMError(
            f"{call.backend} exited with {proc.returncode}. stderr tail:\n{proc.stderr[-2000:]}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}"
        )
    if call.backend == "claude":
        return _claude_reply(proc.stdout, home)
    if call.backend == "codex":
        return _codex_reply(call, proc.stdout, home)
    return _pi_reply(proc.stdout, home)


Ask = Callable[[str], str]

# Each pipeline command logs its calls to this file next to its --out.
CALLS_FILE = "calls.jsonl"

# The job or unit the current call serves, e.g. "tool:<tool_use_id>" or "unit:<position>".
# It is a ContextVar and not an Ask argument because the job that knows it sits several
# calls above the Ask, in code that must not care whether calls are logged. Each job
# sets it in the thread that runs the job.
CALL_SUBJECT: ContextVar[str] = ContextVar("CALL_SUBJECT")


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Offer:
    """What the calls of one job may ask to see before they answer.

    `menu` is the sentence that tells the model what a name in `need` can be and what it
    gets for it. `fetch` gives the text behind one name and raises LookupError for a name
    it does not know. `room` is how many tokens the prompt has left for the instructions
    this adds and for everything fetched.
    """

    menu: str
    fetch: Callable[[str], str]
    room: int


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class _Offered:
    """An Offer in force: the rounds of names asked for so far and the text each name got.

    It outlives one Ask call, so a retry of the same prompt keeps what was already shown
    and does not get the rounds back.
    """

    offer: Offer
    rounds: list[list[str]] = attrs.field(factory=list)
    shown: dict[str, str] = attrs.field(factory=dict)

    def block(self) -> str:
        if not self.shown:
            return ""
        parts = "\n\n".join(f"【{name}】\n{text}" for name, text in self.shown.items())
        return f"\n\n<补充的内容>\n{parts}\n</补充的内容>"

    def take(self, names: list[str], room: int) -> None:
        """Fetch `names` in the order asked while the block stays within `room` tokens."""
        self.rounds.append(names)
        for name in dict.fromkeys(names):
            if name in self.shown:
                continue
            try:
                text = self.offer.fetch(name)
            except LookupError:
                text = OFFER_UNKNOWN
            self.shown[name] = text
            if qwen_tokens(self.block()) > room:
                self.shown[name] = OFFER_NO_ROOM


# Rounds of asking one job gets; the call after them has to answer. Two, so that the
# model can follow up once on what the first round showed it.
OFFER_ROUNDS = 2
OFFER_UNKNOWN = "没有这项内容。"
OFFER_NO_ROOM = "提示词放不下这项内容。"
_OFFER_SHOWN = "<补充的内容> 是你之前要求看的内容。"
_OFFER_EXIT = (
    "答案写在 reply 里。提示词里的内容不足以回答时，reply 先写成 "
    '{{"need": [...]}}，列出还想看的内容。{menu}它们会放进 <补充的内容>，然后再问你一次。'
)
_NEED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"need": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
    "required": ["need"],
    "additionalProperties": False,
}

_OFFERED: ContextVar[_Offered | None] = ContextVar("_OFFERED", default=None)


@contextmanager
def offering(offer: Offer) -> Iterator[None]:
    """Every Ask call made inside may ask to see more, as `offer` allows.

    It is a context and not an Ask argument for the same reason as CALL_SUBJECT: the job
    that knows what can be shown sits above the retry loops that call the Ask.
    """
    token = _OFFERED.set(_Offered(offer=offer))
    try:
        yield
    finally:
        _OFFERED.reset(token)


def offer_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` with the exit: `reply` is either an answer or the names the model needs.

    The root stays an object because the claude and codex CLIs reject a root `anyOf`.
    """
    return {
        "type": "object",
        "properties": {"reply": {"anyOf": [schema, _NEED_SCHEMA]}},
        "required": ["reply"],
        "additionalProperties": False,
    }


class _Need(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    need: list[str] = Field(min_length=1)


class _Wrapped(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reply: _Need | dict[str, Any] = Field(union_mode="left_to_right")


def _unwrap(text: str) -> tuple[list[str] | None, str]:
    """The names a reply under offer_schema asks for, or else its answer as JSON text.

    A reply that is not the wrapper is passed on as it is, so the caller's own check
    rejects it and sends its feedback.
    """
    try:
        reply = _Wrapped.model_validate_json(text).reply
    except ValidationError:
        return None, text
    if isinstance(reply, _Need):
        return reply.need, text
    return None, json.dumps(reply, ensure_ascii=False)


class CallRecord(BaseModel):
    """One endpoint call of a pipeline command, as a line of its calls.jsonl.

    `ask` is the asker's role in the command (e.g. "topic", "call"). A retry is a
    separate record; its prompt in the transcript is the first prompt plus the feedback.
    A call that asked to see more is a separate record too. `transcript` is None only
    when the backend failed before writing one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ask: str
    subject: str
    started: datetime
    ended: datetime
    transcript: str | None
    error: str | None = None


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class CallLog:
    """Appends CallRecords to one file; the asks of all worker threads share it."""

    path: Path
    lock: threading.Lock = attrs.field(factory=threading.Lock)

    def write(self, record: CallRecord) -> None:
        """Append one record as a line."""
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")


def asker(
    backend: Backend,
    model: str,
    effort: Effort,
    system_prompt: str,
    schema: dict[str, Any],
    stream: bool,
    log: CallLog,
    name: str,
    *,
    refuse_capped: bool = False,
) -> Ask:
    """An Ask: the answer to one prompt under a fixed system prompt and reply schema.

    Outside `offering` that is one isolated call. Inside it, a call may reply with the
    names of what it needs to see instead of an answer; their text goes after the prompt
    and the prompt is sent again, until the model answers or OFFER_ROUNDS are used up,
    after which the call has no exit. An offer whose room does not hold the exit's own
    instructions is not made.

    Every call, failed or not, is logged under `name` and the caller's CALL_SUBJECT. With
    `refuse_capped`, an answer whose thinking reached the cap raises ThinkingCapped; a
    capped reply that asks to see more is followed like any other.
    """

    def call_once(system: str, prompt: str, wrapped: bool) -> tuple[list[str] | None, str]:
        call = LLMCall(
            backend=backend,
            model=model,
            effort=effort,
            system_prompt=system,
            prompt=prompt,
            json_schema=offer_schema(schema) if wrapped else schema,
            stream=stream,
        )
        subject = CALL_SUBJECT.get()
        started = datetime.now(UTC)

        def record(transcript: Path | None, error: LLMError | None = None) -> None:
            log.write(
                CallRecord(
                    ask=name,
                    subject=subject,
                    started=started,
                    ended=datetime.now(UTC),
                    transcript=None if transcript is None else str(transcript),
                    error=None if error is None else str(error),
                )
            )

        try:
            reply = complete(call)
        except LLMError as error:
            record(error.transcript, error)
            raise
        need, text = _unwrap(reply.text) if wrapped else (None, reply.text)
        if need is None and refuse_capped and reply.thinking_capped:
            capped = ThinkingCapped(
                f"The thinking reached the {HIVE_THINKING_BUDGET:,}-token cap; "
                f"see {reply.transcript}.",
                text,
                reply.transcript,
            )
            record(reply.transcript, capped)
            raise capped
        record(reply.transcript)
        return need, text

    def ask(prompt: str) -> str:
        offered = _OFFERED.get()
        if offered is None:
            return call_once(system_prompt, prompt, False)[1]
        base = system_prompt.rstrip("\n")
        exit_text = _OFFER_EXIT.format(menu=offered.offer.menu)
        wrapping = json.dumps(offer_schema(schema), ensure_ascii=False)
        room = offered.offer.room - (
            qwen_tokens(f"\n\n{_OFFER_SHOWN}{exit_text}")
            + qwen_tokens(wrapping)
            - qwen_tokens(json.dumps(schema, ensure_ascii=False))
        )
        while True:
            shown = _OFFER_SHOWN if offered.shown else ""
            if room < 0 or len(offered.rounds) >= OFFER_ROUNDS:
                system = f"{base}\n\n{shown}\n" if shown else system_prompt
                return call_once(system, prompt + offered.block(), False)[1]
            need, text = call_once(
                f"{base}\n\n{shown}{exit_text}\n", prompt + offered.block(), True
            )
            if need is None:
                return text
            offered.take(need, room)

    return ask


def build_invocation(call: LLMCall, home: Path) -> Invocation:
    """Command line, environment and cwd that isolate a CLI-backed `call` inside `home`."""
    if call.backend == "hive":
        raise ValueError("The hive backend is HTTP, not a CLI; use complete().")
    work = home / "work"
    if call.backend == "claude":
        argv = [
            "claude", "-p",
            "--model", call.model,
            "--effort", call.effort,
            "--thinking-display", "summarized",
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
            "-c", 'model_reasoning_summary="detailed"',
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
        raise LLMError(
            f"codex returned no reply; see {transcript}. Failures: {failures!r}", transcript
        )
    return LLMReply(text=str(messages[-1]), session_id=thread_id, transcript=transcript)


def _pi_reply(stdout: str, home: Path) -> LLMReply:
    events = [cast(dict[str, Any], json.loads(line)) for line in stdout.splitlines() if line]
    session_id = next(str(e["id"]) for e in events if e["type"] == "session")
    transcript = _single(home / "pi", f"*_{session_id}.jsonl")
    final = next(e for e in reversed(events) if e["type"] == "agent_end")["messages"][-1]
    if final["role"] != "assistant" or final["stopReason"] != "stop":
        raise LLMError(
            f"pi stopped with {final.get('stopReason')!r}: {final.get('errorMessage')!r}; "
            f"see {transcript}.",
            transcript,
        )
    text = "".join(part["text"] for part in final["content"] if part["type"] == "text")
    return LLMReply(text=text, session_id=session_id, transcript=transcript)


def hive_request_body(call: LLMCall, client: dict[str, Any]) -> dict[str, Any]:
    """Chat-completions body for `call`, with thinking set the way `client` declares.

    `client` is the deployment's `hive.client` entry from the gateway's /v1/models;
    pi builds its Hive model menu from the same entry.

    A schema is also written into the system prompt: response_format only constrains the
    sampled tokens, so a model that cannot read the schema plans another shape while it thinks
    and the grammar then cuts its answer short (2026-11-10: 6 of 51 compressions on Qwen3.8
    wrote `targets` as "[", "..." or the word "targets").
    """
    system = call.system_prompt
    if call.json_schema is not None:
        system += f"\n{HIVE_SCHEMA_INTRO}\n{json.dumps(call.json_schema, ensure_ascii=False)}\n"
    body: dict[str, Any] = {
        "model": call.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": call.prompt},
        ],
    }
    if call.stream:
        body |= {"stream": True, "stream_options": {"include_usage": True}}
    if client.get("reasoning"):
        if client.get("thinking_format") == "qwen-chat-template":
            body["chat_template_kwargs"] = {"enable_thinking": True}
            field = client.get("thinking_token_budget_field")
            if field:
                body[field] = HIVE_THINKING_BUDGET
        elif client.get("reasoning_effort"):
            body["reasoning_effort"] = call.effort
    if call.json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": call.json_schema},
        }
    return body


def _hive_complete(call: LLMCall, home: Path) -> LLMReply:
    """No lease header: the gateway reserves a slot for this one request and frees it."""
    body = hive_request_body(call, _hive_client(call.model))
    call_id = uuid.uuid4().hex
    transcript = home / "hive" / f"{call_id}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open("w", encoding="utf-8") as f:
        record = {
            "type": "request",
            "id": call_id,
            **call.model_dump(),
            "url": HIVE_FRAME_URL,
            "body": body,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    status, response = _hive_post("/v1/chat/completions", body)
    waited = 0
    while status == 409 and waited < _HIVE_CAPACITY_WAIT:
        time.sleep(_HIVE_CAPACITY_POLL)
        waited += _HIVE_CAPACITY_POLL
        status, response = _hive_post("/v1/chat/completions", body)
    with transcript.open("a", encoding="utf-8") as f:
        record = {"type": "response", "status": status, "body": response}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if status != 200:
        raise LLMError(
            f"Hive gateway answered {status}: {response!r}; see {transcript}.", transcript
        )
    choice = response["choices"][0]
    if choice["finish_reason"] == "length":
        raise TruncatedReply(
            f"{call.model} stopped with 'length', so the reply is incomplete; see {transcript}.",
            transcript,
        )
    if choice["finish_reason"] != "stop":
        raise LLMError(
            f"{call.model} stopped with {choice['finish_reason']!r}, so the reply is "
            f"incomplete; see {transcript}.",
            transcript,
        )
    text = str(choice["message"]["content"])
    thinking = response.get("usage", {}).get("completion_tokens", 0) - qwen_tokens(text)
    return LLMReply(
        text=text,
        session_id=call_id,
        transcript=transcript,
        thinking_capped=thinking >= HIVE_THINKING_BUDGET - HIVE_CAPPED_MARGIN,
    )


def _hive_client(model: str) -> dict[str, Any]:
    _, listing = _hive_get("/v1/models")
    return hive_client({str(e["id"]): e for e in listing["data"]}, model)


def hive_client(entries: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    """The client declaration `model` is called with: a deployment's own, or a role's.

    The gateway picks a role's member only after the body is built, so every member
    must take the same body. Members may differ in display name, nothing else.
    """
    entry = entries.get(model)
    if entry is not None and "hive" in entry:
        return cast(dict[str, Any], entry["hive"]["client"])
    served = sorted(i for i, e in entries.items() if "hive" in e or i.startswith("deployment/"))
    if entry is None:
        raise LLMError(
            f"{model!r} is neither a running Hive deployment nor a role. Running now: "
            f"{served}. Start one with `hive-frame services apply` or `hive-frame setup apply`."
        )
    members = [f"deployment/{i['deployment']}" for i in entry["instances"]]
    clients = {m: entries[m]["hive"]["client"] for m in members}
    protocols = {
        json.dumps({k: v for k, v in c.items() if k != "name"}, sort_keys=True)
        for c in clients.values()
    }
    if len(protocols) != 1:
        raise LLMError(
            f"The members of {model!r} declare different client protocols, so one request "
            f"body cannot fit whichever member the gateway picks: {clients!r}. "
            "Align models.<model>.client in hive.json."
        )
    return cast(dict[str, Any], next(iter(clients.values())))


def _hive_get(path: str) -> tuple[int, dict[str, Any]]:
    return _hive_send(urllib.request.Request(HIVE_FRAME_URL + path), timeout=30)


def _hive_post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        HIVE_FRAME_URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-hive-client": "intent-aligner"},
    )
    # The gateway waits 1200 s for each read from the engine before it fails the request itself.
    return _hive_send(
        request, timeout=1260, read=_assemble_stream if body.get("stream") else json.load
    )


def _assemble_stream(lines: Iterable[bytes]) -> dict[str, Any]:
    """The server-sent chunks as one chat completion, so transcripts keep a single shape.

    A stream that ends without a finish_reason yields finish_reason None, which the
    caller reports as an incomplete reply.
    """
    response: dict[str, Any] = {}
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason = None
    for raw in lines:
        line = raw.decode()
        if not line.startswith("data: ") or line.startswith("data: [DONE]"):
            continue
        chunk = json.loads(line.removeprefix("data: "))
        for choice in chunk.get("choices", []):
            content.append(choice["delta"].get("content") or "")
            reasoning.append(choice["delta"].get("reasoning_content") or "")
            finish_reason = choice.get("finish_reason") or finish_reason
        response |= {k: v for k, v in chunk.items() if k != "choices" and v is not None}
    message = {"content": "".join(content)}
    if any(reasoning):
        message["reasoning_content"] = "".join(reasoning)
    return {"choices": [{"finish_reason": finish_reason, "message": message}], **response}


def _hive_send(
    request: urllib.request.Request,
    timeout: float,
    read: Callable[[Any], dict[str, Any]] = json.load,
) -> tuple[int, dict[str, Any]]:
    """Gateway refusals come back as (status, body) so they reach the transcript."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, read(response)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        try:
            return exc.code, cast(dict[str, Any], json.loads(text))
        except ValueError:
            return exc.code, {"error": text}
    except (OSError, http.client.HTTPException) as exc:
        raise LLMError(
            f"Hive gateway at {HIVE_FRAME_URL} is unreachable: {exc}. Check "
            "`systemctl --user status hive-frame-gateway` or set HIVE_FRAME_URL."
        ) from exc


def _single(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise LLMError(f"Expected one transcript {root / pattern}, found {matches}.")
    return matches[0]


def main() -> None:
    """CLI entry point: prompt on stdin, reply text on stdout, transcript path on stderr."""
    parser = argparse.ArgumentParser(description="One isolated LLM call.")
    parser.add_argument("--backend", choices=["claude", "codex", "pi", "hive"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="xhigh")
    parser.add_argument("--system", type=Path, required=True, help="file holding the system prompt")
    parser.add_argument("--schema", type=Path, help="file holding a JSON schema for the reply")
    parser.add_argument(
        "--home", type=Path, default=LLM_HOME, help=f"where transcripts go (default {LLM_HOME})"
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hive only: read the reply as a stream or as one block",
    )
    args = parser.parse_args()
    call = LLMCall(
        backend=args.backend,
        model=args.model,
        effort=args.effort,
        system_prompt=args.system.read_text(),
        prompt=sys.stdin.read(),
        json_schema=json.loads(args.schema.read_text()) if args.schema else None,
        stream=args.stream,
    )
    # Absolute because every backend runs with cwd inside home and gets paths under it.
    reply = complete(call, args.home.resolve())
    print(reply.text)
    print(f"transcript: {reply.transcript}", file=sys.stderr)


if __name__ == "__main__":
    main()
