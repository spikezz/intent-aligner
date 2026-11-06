"""Cut a merged Claude or Codex conversation into processing units, in timeline order.

The unit kinds and what is skipped are defined in docs/workflow.md ("处理单元").
A user_message is one unit here; intent_aligner.segment walks a pointer through
it. ai_text is context only: injected into input, never processed.

Usage: python -m intent_aligner.units MERGED.jsonl --out UNITS.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from intent_aligner.jsonl import DamagedLine, read_jsonl
from intent_aligner.seeds import BERLIN


class Source(BaseModel):
    """Where an item comes from: the record's session, uuid and Berlin-local time."""

    model_config = ConfigDict(frozen=True)

    session: str
    uuid: str
    timestamp: datetime


class UserMessage(BaseModel):
    """A message the user typed; `queued` ones were typed while the AI was working."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["user_message"] = "user_message"
    source: Source
    body: str
    queued: bool
    images: tuple[str, ...]


class UserSignal(BaseModel):
    """An ESC interrupt, or a rejected tool call (`tool_use_id` names the call)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["user_signal"] = "user_signal"
    source: Source
    signal: Literal["interrupt", "rejection"]
    text: str
    tool_use_id: str | None


class ToolCall(BaseModel):
    """One tool_use block; `text` is the rendering the extractor sees."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool_call"] = "tool_call"
    source: Source
    tool_use_id: str
    name: str
    input: dict[str, Any]
    text: str


class Observation(BaseModel):
    """What the environment reported; `tool_use_id` is set for tool results."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["observation"] = "observation"
    source: Source
    origin: Literal["tool_result", "task_notification", "system"]
    text: str
    tool_use_id: str | None
    is_error: bool


class AIText(BaseModel):
    """What the AI said to the user."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["ai_text"] = "ai_text"
    source: Source
    text: str


Item = Annotated[
    UserMessage | UserSignal | ToolCall | Observation | AIText,
    Field(discriminator="kind"),
]
ITEM_ADAPTER: TypeAdapter[Item] = TypeAdapter(Item)

INTERRUPT_MARKERS = (
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
)
# Claude Code's fixed openings for a tool call the user turned down. The safety-check
# denial ("denied by a built-in Claude Code safety check") is not the user and stays
# an observation.
REJECTION_PREFIXES = (
    "The user doesn't want to proceed with this tool use",
    "The user doesn't want to take this action right now",
)
_IMAGE_SOURCE = "[Image: source: "
# Records with conversation content are handled explicitly; everything else here
# is session state. An unlisted type raises so that a new CLI version is noticed.
_STATE_TYPES = frozenset(
    {
        "system",
        "last-prompt",
        "mode",
        "permission-mode",
        "atis-latch",
        "ai-title",
        "queue-operation",
        "agent-name",
        "file-history-snapshot",
        "file-history-delta",
        "cost-state",
        "continued-in",
    }
)
# Meta records with no conversation content: the size note after an image Read
# (the tool result already holds the image) and the caveat before local commands.
_SKIPPED_META_PREFIXES = ("[Image: original ", "<local-command-caveat>")
# Local slash commands (/login, /rate-limit-options) and their output never reach
# the model.
_LOCAL_COMMAND_PREFIXES = ("<command-name>", "<local-command-stdout>")


def render_tool_call(name: str, args: dict[str, Any]) -> str:
    """Render a call as "Name:" plus one "key: value" line per argument, values in full.

    Examples:
        >>> print(render_tool_call("Bash", {"command": "ls", "timeout": 5}))
        Bash:
        command: ls
        timeout: 5
    """
    lines = [f"{name}:"]
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {text}")
    return "\n".join(lines)


def claude_units(merged: Path) -> Iterator[Item]:
    """Yield the items of a merged Claude conversation in file (= timeline) order.

    A user message is held back until the next item so that the image source
    records following it can be attached.
    Raises ValueError on a record shape this cutter does not know.
    """
    pending: UserMessage | None = None
    for entry in read_jsonl(merged):
        if isinstance(entry, DamagedLine):
            raise ValueError(f"{merged}:{entry.line} is damaged; rebuild the merged corpus.")
        record = entry.data
        image_paths = _image_sources(record)
        if image_paths is not None:
            if pending is None:
                raise ValueError(
                    f"{merged}:{entry.line} image source record {record['uuid']} follows no "
                    "user message; check its parentUuid chain."
                )
            pending = pending.model_copy(update={"images": (*pending.images, *image_paths)})
            continue
        items = list(_record_items(record, merged, entry.line))
        if not items:
            continue
        if pending is not None:
            yield pending
            pending = None
        for item in items:
            if isinstance(item, UserMessage):
                pending = item
            else:
                yield item
    if pending is not None:
        yield pending


def _source(record: dict[str, Any]) -> Source:
    stamp = datetime.fromisoformat(record["timestamp"]).astimezone(BERLIN)
    return Source(session=record["sessionId"], uuid=record["uuid"], timestamp=stamp)


def _text_of(content: str | list[dict[str, Any]]) -> str:
    """Join the text blocks; images become "[image]" (their content is not read)."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if block["type"] == "text":
            parts.append(block["text"])
        elif block["type"] == "image":
            parts.append("[image]")
        elif block["type"] == "tool_reference":
            parts.append(f"[tool reference: {block['tool_name']}]")
        else:
            raise ValueError(f"Unknown content block type {block['type']!r}; extend _text_of.")
    return "\n".join(parts)


def _message_body(content: str | list[dict[str, Any]]) -> str:
    """The typed text of a message; pasted images sit beside it as image blocks.

    The text already holds an "[Image #N]" marker per pasted image, and segment
    offsets count from the start of this text.
    """
    if isinstance(content, str):
        return content
    texts: list[str] = [block["text"] for block in content if block["type"] == "text"]
    if len(texts) != 1:
        raise ValueError(f"Expected one text block in a user message, got {len(texts)}.")
    return texts[0]


def _image_sources(record: dict[str, Any]) -> list[str] | None:
    """Image paths of a "[Image: source: <path>]" meta record, None for any other record."""
    if record.get("type") != "user" or not record.get("isMeta"):
        return None
    text = _text_of(record["message"]["content"])
    if not text.startswith(_IMAGE_SOURCE):
        return None
    return [line.removeprefix(_IMAGE_SOURCE).removesuffix("]") for line in text.splitlines()]


def _record_items(record: dict[str, Any], merged: Path, line: int) -> Iterator[Item]:
    kind = record.get("type")
    if kind in _STATE_TYPES:
        return
    if kind == "attachment":
        yield from _attachment_items(record)
    elif kind == "assistant":
        yield from _assistant_items(record)
    elif kind == "user":
        yield from _user_items(record, merged, line)
    else:
        raise ValueError(f"{merged}:{line} has unknown record type {kind!r}; add it to units.py.")


def _attachment_items(record: dict[str, Any]) -> Iterator[Item]:
    attachment = record["attachment"]
    if attachment["type"] != "queued_command":
        return
    text = _text_of(attachment["prompt"])
    origin = (attachment.get("origin") or {}).get("kind")
    if origin == "human":
        body = _message_body(attachment["prompt"])
        yield UserMessage(source=_source(record), body=body, queued=True, images=())
    elif origin == "task-notification" or text.startswith("<task-notification>"):
        yield _observation(record, "task_notification", text)
    else:
        yield _observation(record, "system", text)


def _assistant_items(record: dict[str, Any]) -> Iterator[Item]:
    # Synthetic records (API errors, usage limits) are the CLI speaking, not the model.
    synthetic = record.get("isApiErrorMessage") or record["message"]["model"] == "<synthetic>"
    for block in record["message"]["content"]:
        if block["type"] in ("thinking", "redacted_thinking"):
            continue
        if block["type"] == "tool_use":
            yield ToolCall(
                source=_source(record),
                tool_use_id=block["id"],
                name=block["name"],
                input=block["input"],
                text=render_tool_call(block["name"], block["input"]),
            )
        elif block["type"] == "text" and synthetic:
            yield _observation(record, "system", block["text"])
        elif block["type"] == "text":
            yield AIText(source=_source(record), text=block["text"])
        else:
            raise ValueError(f"Unknown assistant block type {block['type']!r} in {record['uuid']}.")


def _user_items(record: dict[str, Any], merged: Path, line: int) -> Iterator[Item]:
    if record.get("isCompactSummary"):
        return
    content = record["message"]["content"]
    origin = (record.get("origin") or {}).get("kind")
    if origin == "human":
        body = _message_body(content)
        yield UserMessage(source=_source(record), body=body, queued=False, images=())
        return
    if isinstance(content, list) and any(b["type"] == "tool_result" for b in content):
        for block in content:
            yield _tool_result_item(record, block)
        return
    text = _text_of(content)
    if record.get("isMeta"):
        if not text.startswith(_SKIPPED_META_PREFIXES):
            notification = origin == "task-notification"
            yield _observation(record, "task_notification" if notification else "system", text)
    elif text in INTERRUPT_MARKERS:
        yield UserSignal(source=_source(record), signal="interrupt", text=text, tool_use_id=None)
    elif origin == "task-notification":
        yield _observation(record, "task_notification", text)
    elif not text.startswith(_LOCAL_COMMAND_PREFIXES):
        raise ValueError(
            f"{merged}:{line} user record {record['uuid']} (origin {origin!r}) is neither a "
            f"message, a signal nor an observation: {text[:200]!r}"
        )


def _tool_result_item(record: dict[str, Any], block: dict[str, Any]) -> Item:
    if block["type"] != "tool_result":
        raise ValueError(f"Tool result record {record['uuid']} also holds a {block['type']!r}.")
    text = _text_of(block.get("content") or "")
    if text.startswith(REJECTION_PREFIXES):
        return UserSignal(
            source=_source(record),
            signal="rejection",
            text=text,
            tool_use_id=block["tool_use_id"],
        )
    return Observation(
        source=_source(record),
        origin="tool_result",
        text=text,
        tool_use_id=block["tool_use_id"],
        is_error=bool(block.get("is_error")),
    )


def _observation(
    record: dict[str, Any], origin: Literal["task_notification", "system"], text: str
) -> Observation:
    return Observation(
        source=_source(record), origin=origin, text=text, tool_use_id=None, is_error=False
    )


# Codex keeps two copies of the conversation: `response_item` records are what the
# model saw, `item_completed` events are what the UI showed. User messages come from
# the UI copy, because the model copy mixes them with injected context (environment,
# goal continuations, skill files, images); everything else comes from the model copy.
_CODEX_STATE_TYPES = frozenset(
    {"session_meta", "turn_context", "world_state", "token_usage_record", "compacted"}
)
_CODEX_STATE_EVENTS = frozenset(
    {"task_complete", "token_count", "thread_settings_applied", "item_completed"}
)
# The UI copies of model-side items; ContextCompaction and Reasoning are not content.
_CODEX_SKIPPED_ITEMS = frozenset(
    {
        "AgentMessage",
        "CommandExecution",
        "ContextCompaction",
        "Extension",
        "FileChange",
        "ImageView",
        "McpToolCall",
        "Reasoning",
        "WebSearch",
    }
)
_CODEX_SKIPPED_RESPONSES = frozenset({"reasoning", "additional_tools", "configuration_update"})
_QUESTION_TOOL = "request_user_input_async"
# The question tool returns this at once; the answers arrive later as a user message.
_QUESTION_ACK = '{"accepted":true}'
_QUESTION_REPLY = re.compile(
    r"<send_user_message_question_reply>\n(.*)\n</send_user_message_question_reply>", re.DOTALL
)
_USER_SHELL = "<user_shell_command>"


class _QuestionAnswer(BaseModel):
    """One answer in a Codex question reply; questionItemId is a JSON [tool, call_id, n]."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    questionItemId: str

    @property
    def call_id(self) -> str:
        """The call_id of the request_user_input_async call this answers."""
        return str(json.loads(self.questionItemId)[1])


_ANSWERS_ADAPTER: TypeAdapter[list[_QuestionAnswer]] = TypeAdapter(list[_QuestionAnswer])


def codex_units(merged: Path) -> Iterator[Item]:
    """Yield the items of a merged Codex conversation in file (= timeline) order.

    A user message is `queued` when the AI already acted in the current turn. The
    answers to request_user_input_async become that call's tool result, as Claude's
    AskUserQuestion answers are. A goal set with /goal is a user message the first
    time its objective appears; status changes are state.
    Raises ValueError on a record shape this cutter does not know.
    """
    acted = False
    question_calls: set[str] = set()
    objectives: dict[str, str] = {}
    for entry in read_jsonl(merged):
        if isinstance(entry, DamagedLine):
            raise ValueError(f"{merged}:{entry.line} is damaged; rebuild the merged corpus.")
        record = entry.data
        kind, payload = record["type"], record.get("payload", {})
        where = f"{merged}:{entry.line}"
        if kind in _CODEX_STATE_TYPES:
            continue
        if kind == "event_msg":
            event = payload["type"]
            if event == "task_started":
                acted = False
            elif event == "item_completed" and payload["item"]["type"] == "UserMessage":
                yield _codex_user_item(record, payload["item"], acted)
            elif event == "item_completed" and payload["item"]["type"] not in _CODEX_SKIPPED_ITEMS:
                raise ValueError(f"{where} has unknown item type {payload['item']['type']!r}.")
            elif event == "turn_aborted":
                yield _codex_interrupt(record, payload, where)
            elif event == "thread_goal_updated":
                goal = payload["goal"]
                if objectives.get(goal["threadId"]) != goal["objective"]:
                    objectives[goal["threadId"]] = goal["objective"]
                    yield _codex_goal(record, goal["objective"], acted)
            elif event not in _CODEX_STATE_EVENTS:
                raise ValueError(f"{where} has unknown event type {event!r}; add it to units.py.")
        elif kind == "response_item":
            for item in _codex_response_items(record, payload, question_calls, where):
                acted = acted or isinstance(item, ToolCall | AIText)
                yield item
        else:
            raise ValueError(f"{where} has unknown record type {kind!r}; add it to units.py.")


def _codex_source(record: dict[str, Any], uuid: str) -> Source:
    stamp = datetime.fromisoformat(record["timestamp"]).astimezone(BERLIN)
    return Source(session=record["_merge"]["session"], uuid=uuid, timestamp=stamp)


def _codex_user_item(record: dict[str, Any], message: dict[str, Any], acted: bool) -> Item:
    source = _codex_source(record, message["id"])
    texts = [block["text"] for block in message["content"] if block["type"] == "text"]
    if len(texts) != 1:
        raise ValueError(
            f"Expected one text block in user message {message['id']}, got {len(texts)}."
        )
    reply = _QUESTION_REPLY.fullmatch(texts[0])
    if reply is not None:
        answers = _ANSWERS_ADAPTER.validate_json(reply.group(1))
        call_ids = {answer.call_id for answer in answers}
        if len(call_ids) != 1:
            raise ValueError(
                f"Question reply {message['id']} answers {len(call_ids)} calls; expected 1."
            )
        text = "\n\n".join(f"问题：{a.question}\n回答：{a.answer}" for a in answers)
        return Observation(
            source=source,
            origin="tool_result",
            text=text,
            tool_use_id=call_ids.pop(),
            is_error=False,
        )
    images = tuple(block["path"] for block in message["content"] if block["type"] == "local_image")
    return UserMessage(source=source, body=texts[0], queued=acted, images=images)


def _codex_interrupt(record: dict[str, Any], payload: dict[str, Any], where: str) -> UserSignal:
    if payload["reason"] != "interrupted":
        raise ValueError(f"{where} aborts a turn for {payload['reason']!r}; decide what it means.")
    # Codex writes no text for ESC; Claude's marker keeps both platforms' windows alike.
    return UserSignal(
        source=_codex_source(record, payload["turn_id"]),
        signal="interrupt",
        text=INTERRUPT_MARKERS[0],
        tool_use_id=None,
    )


def _codex_goal(record: dict[str, Any], objective: str, acted: bool) -> UserMessage:
    uuid = f"{record['_merge']['session']}:{record['ordinal']}"
    return UserMessage(source=_codex_source(record, uuid), body=objective, queued=acted, images=())


def _codex_response_items(
    record: dict[str, Any], payload: dict[str, Any], question_calls: set[str], where: str
) -> Iterator[Item]:
    response = payload["type"]
    if response in _CODEX_SKIPPED_RESPONSES:
        return
    source = _codex_source(record, payload["id"])
    if response == "message":
        yield from _codex_message_items(payload, source)
    elif response in ("custom_tool_call", "function_call"):
        name, args = _codex_call(payload, where)
        if name == _QUESTION_TOOL:
            question_calls.add(payload["call_id"])
        yield ToolCall(
            source=source,
            tool_use_id=payload["call_id"],
            name=name,
            input=args,
            text=render_tool_call(name, args),
        )
    elif response == "web_search_call":
        name, args = "web_search", payload["action"]
        yield ToolCall(
            source=source,
            tool_use_id=payload["id"],
            name=name,
            input=args,
            text=render_tool_call(name, args),
        )
    elif response in ("custom_tool_call_output", "function_call_output"):
        if payload["call_id"] in question_calls and payload["output"] == _QUESTION_ACK:
            return
        yield Observation(
            source=source,
            origin="tool_result",
            text=_codex_output_text(payload["output"]),
            tool_use_id=payload["call_id"],
            is_error=False,
        )
    else:
        raise ValueError(f"{where} has unknown response item {response!r}; add it to units.py.")


def _codex_message_items(payload: dict[str, Any], source: Source) -> Iterator[Item]:
    """Assistant text, and the output of a shell command the user ran with "!".

    Other user-role messages are the model's copy of user messages or injected
    context, and developer messages are injected instructions.
    """
    text = "\n".join(block["text"] for block in payload["content"] if "text" in block)
    if payload["role"] == "assistant":
        yield AIText(source=source, text=text)
    elif payload["role"] == "user" and text.startswith(_USER_SHELL):
        yield Observation(
            source=source, origin="system", text=text, tool_use_id=None, is_error=False
        )


def _codex_call(payload: dict[str, Any], where: str) -> tuple[str, dict[str, Any]]:
    """Name and arguments; a custom call's free-form input becomes the one argument "input"."""
    if payload["type"] == "custom_tool_call":
        return payload["name"], {"input": payload["input"]}
    name = (
        f"{payload['namespace']}.{payload['name']}" if payload.get("namespace") else payload["name"]
    )
    args = json.loads(payload["arguments"])
    if not isinstance(args, dict):
        raise ValueError(f"{where} call {payload['call_id']} has non-object arguments.")
    return name, args


def _codex_output_text(output: str | list[dict[str, Any]]) -> str:
    """Join the text blocks; images become "[image]" (their content is not read)."""
    if isinstance(output, str):
        return output
    parts: list[str] = []
    for block in output:
        if block["type"] == "input_text":
            parts.append(block["text"])
        elif block["type"] == "input_image":
            parts.append("[image]")
        else:
            raise ValueError(
                f"Unknown tool output block type {block['type']!r}; extend _codex_output_text."
            )
    return "\n".join(parts)


def conversation_units(merged: Path) -> Iterator[Item]:
    """The items of a merged conversation, cut by the platform its file name starts with."""
    if merged.name.startswith("claude-"):
        return claude_units(merged)
    if merged.name.startswith("codex-"):
        return codex_units(merged)
    raise ValueError(f"{merged.name} names no platform; merged files start with claude- or codex-.")


def main() -> None:
    """CLI entry point: write the items of MERGED as JSONL and print counts per kind."""
    parser = argparse.ArgumentParser(description="Cut a merged conversation into units.")
    parser.add_argument("merged", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8") as out:
        for item in conversation_units(args.merged):
            counts[item.kind] += 1
            out.write(item.model_dump_json() + "\n")
    for kind, n in counts.most_common():
        print(f"{n:7,}  {kind}")


if __name__ == "__main__":
    main()
