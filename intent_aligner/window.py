"""The conversation window an LLM call sees around the unit it judges.

The tool work of a round enters the window only as that round's one digest (see
intent_aligner.compress), so the window is mostly what the user and the AI said.
A round is one speech, by the user or the AI, and everything up to the next speech.
"""

from __future__ import annotations

from argparse import ArgumentTypeError
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

import attrs
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from tokenizers import Tokenizer

from intent_aligner.intent import Intent
from intent_aligner.units import (
    AIText,
    Item,
    Observation,
    Speech,
    ToolCall,
    UserMessage,
    UserSignal,
)

# hive-frame qwen38-np8 gives each slot 18,944 tokens; the rest of the slot is left
# for the 9,216-token thinking budget (llm._HIVE_THINKING_BUDGET) and the answer; the
# largest prompt plus completion on np8 so far is 15,947 tokens (2026-11-06).
PROMPT_BUDGET = 9_000
# Qwen3-0.6B shares the Qwen tokenizer family; on six corpus texts its counts were
# within 7% of the qwen38 server's /tokenize, which PROMPT_BUDGET's slack absorbs.
TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_WINDOW_ROUNDS = 2
# How far call G looks past the unit it files. Both aggregates misfiled before this
# existed were settled within two units; the limits keep room for the earlier rounds.
FOLLOWING_ITEMS = 8
FOLLOWING_BUDGET = 3_000

Count = Callable[[str], int]


class CallDigest(BaseModel):
    """A tool call compressed to what it is for and the concrete things it acts on.

    `intents` are the call's intents for the main loop as well, so each carries its
    evidence in the rendered call text.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["call"] = "call"
    tool_use_id: str
    intents: list[Intent] = Field(min_length=1)
    targets: str

    @property
    def purpose(self) -> str:
        """The intents as one line, for windows and trajectories."""
        return "；".join(i.text for i in self.intents)


class ResultDigest(BaseModel):
    """A tool result compressed to one sentence per chunk it was cut into."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["result"] = "result"
    tool_use_id: str
    summaries: list[str]


class RoundDigest(BaseModel):
    """All tool work of one round in one passage; `round_uuid` is its opening speech."""

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


def window_rounds(text: str) -> int:
    """argparse type of --window-rounds: how many earlier rounds a window may hold."""
    rounds = int(text)
    if rounds < 1:
        raise ArgumentTypeError(f"--window-rounds must be at least 1, got {rounds}.")
    return rounds


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
        if isinstance(item, Speech):
            round_uuid, round_shown = item.source.uuid, False
        if isinstance(item, ToolCall):
            if not round_shown:
                entries.append(f"【AI 的工具操作】\n{_round_summary(digests, round_uuid)}")
                round_shown = True
        elif not _is_tool_result(item):
            entries.append(_plain_entry(item))
    return "\n\n".join(entries)


class ToolCompression(BaseModel):
    """The tool calls and results among items[begin:through] of a round, as one level-2 passage.

    `start` is the timeline index where the round's items begin. A round can be
    compressed several times; each compression covers the records after the one
    before it, and every passage stays in the window.
    """

    model_config = ConfigDict(frozen=True)

    start: int
    begin: int
    through: int
    summary: str


def render_round_so_far(
    items: Sequence[Item],
    low: int,
    index: int,
    digests: Digests,
    compressions: Sequence[ToolCompression],
) -> str:
    """The current round items[low:index], each tool call and result as its level-1 digest.

    The round digest is not used here because it tells how the whole round ended,
    which the unit in hand must not see yet. The tool records a compression covers
    are replaced by its passage, put where the first of them was; everything else
    keeps its place.
    """
    entries: list[str] = []
    placed: set[int] = set()
    for position in range(low, index):
        item = items[position]
        covering = next((c for c in compressions if c.begin <= position < c.through), None)
        if covering is None or not is_tool_record(item):
            entries.append(round_entry(item, digests))
        elif covering.begin not in placed:
            entries.append(f"【AI 此前的工具操作（已压缩）】\n{covering.summary}")
            placed.add(covering.begin)
    return "\n\n".join(entries)


def round_entry(item: Item, digests: Digests) -> str:
    """One item of the current round: a tool call or result as its level-1 digest, the rest verbatim."""
    if isinstance(item, ToolCall):
        call = digests.calls[item.tool_use_id]
        return f"【AI 的工具调用】\n{item.name} 意图：{call.purpose} 对象：{call.targets}"
    if isinstance(item, Observation) and item.tool_use_id is not None:
        return f"【工具结果】\n{' '.join(digests.results[item.tool_use_id].summaries)}"
    return _plain_entry(item)


def render_following(items: Sequence[Item], index: int, digests: Digests, count: Count) -> str:
    """What comes after items[index], at most FOLLOWING_ITEMS entries in FOLLOWING_BUDGET tokens.

    Entries are whole and in order, so the first one over the budget ends the text.
    """
    entries: list[str] = []
    used = 0
    for item in items[index + 1 : index + 1 + FOLLOWING_ITEMS]:
        entry = round_entry(item, digests)
        used += count(entry)
        if used > FOLLOWING_BUDGET:
            break
        entries.append(entry)
    return "\n\n".join(entries)


def is_tool_record(item: Item) -> bool:
    """A tool call or its result; a rejection is the user's and is not one."""
    return isinstance(item, ToolCall) or _is_tool_result(item)


def _is_tool_result(item: Item) -> bool:
    return isinstance(item, Observation) and item.tool_use_id is not None


def _plain_entry(item: Item) -> str:
    """An item shown verbatim: anything but a tool call or a tool result."""
    if isinstance(item, UserMessage):
        images = "".join(f"\n[图片] {path}" for path in item.images)
        return f"【用户】\n{item.body}{images}"
    if isinstance(item, AIText):
        return f"【AI】\n{item.text}"
    if isinstance(item, UserSignal):
        return f"【用户信号】\n{item.text}"
    assert isinstance(item, Observation), item
    label = "任务通知" if item.origin == "task_notification" else "系统"
    return f"【{label}】\n{item.text}"


def _round_summary(digests: Digests, round_uuid: str) -> str:
    if round_uuid not in digests.rounds:
        raise ValueError(
            f"No digest for round {round_uuid!r}; run intent_aligner.compress on this "
            "conversation first."
        )
    return digests.rounds[round_uuid].summary


def window_text(
    items: Sequence[Item], index: int, digests: Digests, room: int, count: Count, rounds: int
) -> str:
    """The `rounds` rounds before the speech items[index], within `room` tokens.

    One round fewer is tried until the text fits; raises WindowError when even one
    round does not.
    """
    starts = [i for i in range(index) if isinstance(items[i], Speech)]
    text = ""
    for taken in range(rounds, 0, -1):
        low = starts[-taken] if len(starts) >= taken else 0
        text = render_items(items[low:index], digests)
        if count(text) <= room:
            return text
    raise WindowError(
        f"The 1-round window before item {index} ({items[index].source.uuid}) "
        f"needs {count(text):,} tokens but only {room:,} are left; mark it needs_review."
    )
