"""The conversation window an LLM call sees around the unit it judges.

Tool calls and results enter the window as their level-1 digests (intent_aligner.compress),
each in its place on the timeline. When a prompt would not fit, the main loop replaces
the oldest of them by a passage (ToolCompression), so the newer ones stay one by one.
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
from intent_aligner.llm import HIVE_THINKING_BUDGET
from intent_aligner.units import (
    AIText,
    Item,
    Observation,
    Speech,
    ToolCall,
    UserMessage,
    UserSignal,
)

# hive-frame qwen38-np6 gives each slot 27,136 tokens. The thinking budget is set
# apart and the prompt and the answer share what is left, so the prompt may take all
# of it; an answer left without room ends with finish_reason "length" (llm.TruncatedReply).
SLOT_TOKENS = 27_136
PROMPT_BUDGET = SLOT_TOKENS - HIVE_THINKING_BUDGET
# Qwen3-0.6B shares the Qwen tokenizer family; on six corpus texts its counts were
# within 7% of the qwen38 server's /tokenize. On the largest prompts of
# claude-87eeb4bd-qwen it counted the user prompt alone as high as the server counted
# system and user prompt together, so it errs on the side of fitting.
TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_WINDOW_ROUNDS = 2
# Call G alone sees what follows its unit; these bound that text.
FOLLOWING_ITEMS = 8
FOLLOWING_BUDGET = 3_000
# A passage comes to 60-160 tokens (claude-87eeb4bd), so one call frees most of this;
# a much smaller batch would call the model again at nearly every later unit.
COMPRESSION_BATCH = 2_000

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


Digest = Annotated[CallDigest | ResultDigest, Field(discriminator="kind")]
DIGEST_ADAPTER: TypeAdapter[Digest] = TypeAdapter(Digest)


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Digests:
    """The level-1 digests of one conversation, by tool_use_id."""

    calls: dict[str, CallDigest] = attrs.field(factory=dict)
    results: dict[str, ResultDigest] = attrs.field(factory=dict)

    def add(self, digest: Digest) -> None:
        """File `digest` under its kind."""
        if isinstance(digest, CallDigest):
            self.calls[digest.tool_use_id] = digest
        else:
            self.results[digest.tool_use_id] = digest


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


class ToolCompression(BaseModel):
    """The tool calls and results among items[begin:through], as one passage.

    Each passage covers records no earlier passage covers, and stays in the window.
    """

    model_config = ConfigDict(frozen=True)

    begin: int
    through: int
    summary: str


def covering(position: int, compressions: Sequence[ToolCompression]) -> ToolCompression | None:
    """The passage whose range holds the timeline index `position`."""
    return next((c for c in compressions if c.begin <= position < c.through), None)


def render_items(
    items: Sequence[Item],
    low: int,
    index: int,
    digests: Digests,
    compressions: Sequence[ToolCompression],
) -> str:
    """items[low:index] as window text, each tool call and result as its level-1 digest.

    The tool records a compression covers are replaced by its passage, put where the
    first of them shows; everything else keeps its place.
    """
    entries: list[str] = []
    placed: set[int] = set()
    for position in range(low, index):
        item = items[position]
        passage = covering(position, compressions)
        if passage is None or not is_tool_record(item):
            entries.append(round_entry(item, digests))
        elif passage.begin not in placed:
            entries.append(f"【AI 此前的工具操作（已压缩）】\n{passage.summary}")
            placed.add(passage.begin)
    return "\n\n".join(entries)


def round_entry(item: Item, digests: Digests) -> str:
    """One item of the window: a tool call or result as its level-1 digest, the rest verbatim."""
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


def window_start(items: Sequence[Item], low: int, rounds: int) -> int:
    """Where a window opens that holds `rounds` rounds before items[low]; 0 when fewer exist."""
    starts = [i for i in range(low) if isinstance(items[i], Speech)]
    return starts[-rounds] if len(starts) >= rounds else 0


def window_text(
    items: Sequence[Item],
    low: int,
    index: int,
    digests: Digests,
    compressions: Sequence[ToolCompression],
    room: int,
    count: Count,
    rounds: int,
) -> str:
    """`rounds` rounds before items[low], then items[low:index]; WindowError over `room` tokens."""
    start = window_start(items, low, rounds)
    text = render_items(items, start, index, digests, compressions)
    if count(text) > room:
        raise WindowError(
            f"The {rounds}-round window before item {index} ({items[index].source.uuid}) "
            f"needs {count(text):,} tokens but only {room:,} are left; mark it needs_review."
        )
    return text
