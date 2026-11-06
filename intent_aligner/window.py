"""The conversation window an LLM call sees before a user message.

The tool work of a round enters the window only as that round's one digest (see
intent_aligner.compress), so the window is mostly what the user and the AI said.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

import attrs
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from tokenizers import Tokenizer

from intent_aligner.units import AIText, Item, ToolCall, UserMessage, UserSignal

# hive-frame qwen38-np8 gives each slot 18,944 tokens; the rest of the slot is left
# for the 9,216-token thinking budget (llm._HIVE_THINKING_BUDGET) and the answer; the
# largest prompt plus completion on np8 so far is 15,947 tokens (2026-11-06).
PROMPT_BUDGET = 9_000
# Qwen3-0.6B shares the Qwen tokenizer family; on six corpus texts its counts were
# within 7% of the qwen38 server's /tokenize, which PROMPT_BUDGET's slack absorbs.
TOKENIZER = "Qwen/Qwen3-0.6B"
ROUNDS = (2, 1)

Count = Callable[[str], int]


class CallDigest(BaseModel):
    """A tool call compressed to what it is for and the concrete things it acts on."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["call"] = "call"
    tool_use_id: str
    intent: str
    targets: str


class ResultDigest(BaseModel):
    """A tool result compressed to one sentence per chunk it was cut into."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["result"] = "result"
    tool_use_id: str
    summaries: list[str]


class RoundDigest(BaseModel):
    """All tool work of one round in one passage; `round_uuid` is its opening user message."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["round"] = "round"
    round_uuid: str
    summary: str


Digest = Annotated[CallDigest | ResultDigest | RoundDigest, Field(discriminator="kind")]
DIGEST_ADAPTER: TypeAdapter[Digest] = TypeAdapter(Digest)


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Digests:
    """The digests of one conversation; calls and results by tool_use_id, rounds by round_uuid."""

    calls: dict[str, CallDigest] = attrs.field(factory=dict)
    results: dict[str, ResultDigest] = attrs.field(factory=dict)
    rounds: dict[str, RoundDigest] = attrs.field(factory=dict)

    def add(self, digest: Digest) -> None:
        """File `digest` under its kind."""
        if isinstance(digest, CallDigest):
            self.calls[digest.tool_use_id] = digest
        elif isinstance(digest, ResultDigest):
            self.results[digest.tool_use_id] = digest
        else:
            self.rounds[digest.round_uuid] = digest


def load_digests(path: Path) -> Digests:
    """Read a digest JSONL written by intent_aligner.compress."""
    digests = Digests()
    for line in path.read_text(encoding="utf-8").splitlines():
        digests.add(DIGEST_ADAPTER.validate_json(line))
    return digests


class WindowError(ValueError):
    """Even the one-round window does not fit; the unit needs review."""


@cache
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(TOKENIZER)


def qwen_tokens(text: str) -> int:
    """Token count of `text` under the Qwen tokenizer (downloaded once into the HF cache)."""
    return len(_tokenizer().encode(text).ids)


def render_items(items: Sequence[Item], digests: Digests) -> str:
    """The items as window text; a round's tool work appears once, at its first tool call.

    A tool result can land in a later round than its call (a message queued while
    the call ran); it is part of its call's round, so results never open an entry.
    Raises ValueError when a round with tool calls has no digest.
    """
    entries: list[str] = []
    round_uuid = ""
    round_shown = False
    for item in items:
        if isinstance(item, UserMessage):
            round_uuid, round_shown = item.source.uuid, False
            images = "".join(f"\n[图片] {path}" for path in item.images)
            entries.append(f"【用户】\n{item.body}{images}")
        elif isinstance(item, AIText):
            entries.append(f"【AI】\n{item.text}")
        elif isinstance(item, UserSignal):
            entries.append(f"【用户信号】\n{item.text}")
        elif isinstance(item, ToolCall):
            if not round_shown:
                entries.append(f"【AI 的工具操作】\n{_round_summary(digests, round_uuid)}")
                round_shown = True
        elif item.origin != "tool_result":
            label = "任务通知" if item.origin == "task_notification" else "系统"
            entries.append(f"【{label}】\n{item.text}")
    return "\n\n".join(entries)


def _round_summary(digests: Digests, round_uuid: str) -> str:
    if round_uuid not in digests.rounds:
        raise ValueError(
            f"No digest for round {round_uuid!r}; run intent_aligner.compress on this "
            "conversation first."
        )
    return digests.rounds[round_uuid].summary


def window_text(
    items: Sequence[Item], index: int, digests: Digests, room: int, count: Count
) -> str:
    """The rounds before the user message items[index], within `room` tokens.

    A round is one typed user message and everything up to the next one. Two
    earlier rounds are tried first, then one; raises WindowError when neither fits.
    """
    starts = [i for i in range(index) if isinstance(items[i], UserMessage)]
    for rounds in ROUNDS:
        low = starts[-rounds] if len(starts) >= rounds else 0
        text = render_items(items[low:index], digests)
        if count(text) <= room:
            return text
    raise WindowError(
        f"The {ROUNDS[-1]}-round window before item {index} ({items[index].source.uuid}) "
        f"needs {count(text):,} tokens but only {room:,} are left; mark it needs_review."
    )
