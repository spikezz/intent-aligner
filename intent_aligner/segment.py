"""Cut a speech (a user message or what the AI said to the user) into segments with an LLM-driven pointer.

The model never reports offsets: each call shows the whole message split at the
pointer and asks for the intents of the next segment plus the segment's closing
text, copied verbatim. Code locates that text in the rest of the message, so one
segment always ends where the next begins. Each intent carries its evidence, an
excerpt code checks against the located segment, and whether the speaker said it or
the model inferred it, so the main loop takes speech segments' intents from here and
does not ask call A for them.

The prompt also carries the window before the message (intent_aligner.window),
so the digests of its conversation must exist first (intent_aligner.compress).

A segment the model cannot place, or a message whose window does not fit, becomes
one `needs_review` segment running to the end of the body, so the segments of every
message still tile it.

Usage: python -m intent_aligner.segment MERGED.jsonl --digests DIGESTS.jsonl
           --backend codex --model gpt-6.1-sol --out SEGMENTS.jsonl [--workers 8]
           [--window-rounds N]
Every speech of the conversation is segmented; the output is appended to as
messages finish, so a stopped run resumes where it was.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from intent_aligner.intent import INTENT_SCHEMA, Intent, stray_evidence
from intent_aligner.llm import (
    CALL_SUBJECT,
    CALLS_FILE,
    TRUNCATED_FEEDBACK,
    Ask,
    CallLog,
    LLMError,
    TruncatedReply,
    asker,
)
from intent_aligner.units import SPEAKER_NAMES, Item, Speech, conversation_units
from intent_aligner.window import (
    DEFAULT_WINDOW_ROUNDS,
    PROMPT_BUDGET,
    Count,
    Digests,
    WindowError,
    load_digests,
    qwen_tokens,
    window_rounds,
    window_text,
)

MAX_ATTEMPTS = 4
CONTEXT_CHARS = 30
# Room kept free for retry feedback, which can quote every place an anchor occurs.
FEEDBACK_ROOM = 2_000

SYSTEM_PROMPT = """\
你在把对话里的一次发言按意图切成连续的段落。发言可能是用户说的，也可能是 AI 对用户说的，<说话的一方> 写明是谁。<此前的对话> 是这次发言之前的对话，其中每一轮的工具操作已经压缩成一段概括（【AI 的工具操作】）；它只用来理解说话的一方在回应什么、指的是什么，不切分它。要切的发言分成两部分：<已处理> 是前面已经切好的文本，<剩余> 是还没有切的文本。你只确定下一段：它从 <剩余> 的第一个字符开始，到哪里结束由你判断。

一段是一串连续的文本，其中表达的意图集合保持不变；意图集合一变，这一段就结束。一个句子可以被切成几段，几句话也可以是一段。本身没有意图的内容（客套话、粘贴进来的材料、日志、表格、图片标记 [Image #N]）归入用到它、说明它或回应它的那一段，不单独成段。

回答两项：
- intents：这一段里说话的一方的意图。说话的一方自己要做或者要对方做的事不同，才算不同的意图；告知的情况（理由、背景、条件、自己接下来的状态）并入它支撑的那个意图，不单独列出。一段里只有告知、没有要做的事时，告知的内容本身就是意图，比如 AI 报告的结论、回答对方的内容。每个意图回答四项：
  - text：意图，一句话。要脱离对话也能看懂：用 <此前的对话> 把“这个”“当前任务”“继续干”这类指代换成具体的事。
  - evidence：这个意图在这一段原文里的依据，原样摘录。可以用 … 省略中间的部分，但被 … 分开的每个片段都必须原样出现在这一段里，并且按原文的先后顺序排列。
  - explicit：说话的一方在这一段里明说了这个意图，就是 true。text 里有这一段原文没有说出、要靠 <此前的对话> 或常识推断的内容，就是 false，比如原文只有“继续”，要继续做什么来自此前的对话。只把“这个”“那里”换成它明确指向的东西，不算推断。
  - note：explicit 为 false 时写推断依据：推断出了什么，依据的是此前对话里的哪件事。explicit 为 true 时写空字符串。
- segment_end：从 <剩余> 里原样抄出这一段末尾的几个字符（不含末尾的空格和换行），一个字符都不能改，包括标点和空格。代码会在 <剩余> 里查找它来确定段落结尾，所以它在 <剩余> 里必须只出现一次；不够唯一就往前多抄几个字。
"""

FROM_START_RULE = (
    "这一次 segment_end 改成从 <剩余> 的第一个字符开始，原样抄到这一段的末尾，"
    "代码会检查 <剩余> 是否以它开头。"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "intents": {"type": "array", "items": INTENT_SCHEMA},
        "segment_end": {"type": "string", "minLength": 1},
    },
    "required": ["intents", "segment_end"],
    "additionalProperties": False,
}


class SegmentAnswer(BaseModel):
    """What the model returns for one segment."""

    model_config = ConfigDict(frozen=True)

    intents: list[Intent]
    segment_end: str


class Segment(BaseModel):
    """Characters [start, end) of the message body, with the intents the model gave."""

    model_config = ConfigDict(frozen=True)

    start: int
    end: int
    text: str
    intents: list[Intent]
    attempts: int
    needs_review: bool = False


class MessageSegments(BaseModel):
    """One line of the segments file: a user message and the segments that tile its body."""

    model_config = ConfigDict(frozen=True)

    uuid: str
    segments: list[Segment]


class AnchorError(ValueError):
    """The closing text does not pin down one position; the message goes back to the model."""

    def __init__(self, message: str, *, ambiguous: bool) -> None:
        super().__init__(message)
        self.ambiguous = ambiguous


def locate_end(body: str, start: int, anchor: str, *, from_start: bool) -> int:
    """Return the end of the segment that starts at `start` and closes with `anchor`.

    `anchor` must occur exactly once in body[start:] (overlapping occurrences
    count); with `from_start` it is instead the whole segment and must be a prefix
    of body[start:]. Whitespace right after the match joins the segment.

    Examples:
        >>> locate_end("好的，好的，就这样。\\n\\n下一句", 0, "就这样。", from_start=False)
        12
        >>> locate_end("好的，好的，就这样。", 3, "好的，", from_start=True)
        6
    """
    rest = body[start:]
    if from_start:
        if not rest.startswith(anchor):
            raise AnchorError(
                f"<剩余> 不是以你给的 segment_end 开头。<剩余> 的开头是："
                f"{rest[: len(anchor) + CONTEXT_CHARS]!r}",
                ambiguous=False,
            )
        hits = [0]
    else:
        hits = [i for i in range(len(rest)) if rest.startswith(anchor, i)]
    if not hits:
        raise AnchorError(
            "segment_end 在 <剩余> 里找不到。它必须是 <剩余> 里逐字照抄的一段文本。",
            ambiguous=False,
        )
    if len(hits) > 1:
        places = "\n".join(f"- …{_around(rest, i, len(anchor))}…" for i in hits)
        raise AnchorError(
            f"segment_end 在 <剩余> 里出现了 {len(hits)} 次，无法确定是哪一处：\n{places}",
            ambiguous=True,
        )
    end = start + hits[0] + len(anchor)
    while end < len(body) and body[end].isspace():
        end += 1
    return end


def _around(rest: str, i: int, length: int) -> str:
    left = max(0, i - CONTEXT_CHARS)
    return (
        f"{rest[left:i]}【{rest[i : i + length]}】{rest[i + length : i + length + CONTEXT_CHARS]}"
    )


def build_prompt(window: str, message: Speech, start: int, feedback: str) -> str:
    """The window, the speaker, the speech split at the pointer, its images, and any retry feedback."""
    parts = [
        f"<此前的对话>\n{window}\n</此前的对话>",
        f"<说话的一方>\n{SPEAKER_NAMES[message.speaker]}\n</说话的一方>",
        f"<已处理>\n{message.body[:start]}\n</已处理>",
        f"<剩余>\n{message.body[start:]}\n</剩余>",
    ]
    if message.images:
        parts.append("这条消息附带的图片（只有路径，内容不可见）：\n" + "\n".join(message.images))
    if feedback:
        parts.append(f"你上一次的回答有问题，请重新回答。\n{feedback}")
    return "\n\n".join(parts)


def segment_message(window: str, message: Speech, ask: Ask) -> list[Segment]:
    """Walk the pointer through the body; `ask` sends one prompt and returns the reply text.

    A segment still not located after MAX_ATTEMPTS calls takes the rest of the body
    as one `needs_review` segment, because the pointer cannot move past it.
    """
    segments: list[Segment] = []
    start = 0
    while start < len(message.body):
        segment = _next_segment(window, message, start, ask)
        segments.append(segment)
        start = segment.end
    return segments


def _next_segment(window: str, message: Speech, start: int, ask: Ask) -> Segment:
    feedback = ""
    from_start = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer = SegmentAnswer.model_validate_json(
                ask(build_prompt(window, message, start, feedback))
            )
            end = locate_end(message.body, start, answer.segment_end, from_start=from_start)
        except TruncatedReply:
            feedback = TRUNCATED_FEEDBACK
            continue
        except ValidationError as error:
            feedback = f"回答不符合要求：{error}"
            continue
        except AnchorError as error:
            from_start = from_start or error.ambiguous
            rule = FROM_START_RULE if from_start else ""
            feedback = f"你给的 segment_end 是 {answer.segment_end!r}。{error}\n{rule}"
            continue
        text = message.body[start:end]
        strays = stray_evidence(answer.intents, text)
        if strays:
            feedback = (
                f"这些 evidence 不是这一段的原文摘录：{strays!r}。这一段按你给的 segment_end "
                f"定位为：{text!r}。evidence 被 … 分开的每个片段都必须原样出现在这一段里，"
                "并且按原文的先后顺序排列。"
            )
            continue
        return Segment(start=start, end=end, text=text, intents=answer.intents, attempts=attempt)
    print(
        f"needs_review: segment at char {start} of message {message.source.uuid} was not "
        f"located in {MAX_ATTEMPTS} attempts; last feedback: {feedback}",
        file=sys.stderr,
        flush=True,
    )
    return _review_segment(message, start, MAX_ATTEMPTS)


def _review_segment(message: Speech, start: int, attempts: int) -> Segment:
    return Segment(
        start=start,
        end=len(message.body),
        text=message.body[start:],
        intents=[],
        attempts=attempts,
        needs_review=True,
    )


def message_window(
    items: Sequence[Item], index: int, digests: Digests, count: Count, rounds: int
) -> str:
    """The window for segmenting items[index], sized so the longest prompt still fits."""
    message = items[index]
    assert isinstance(message, Speech)
    rest = count(SYSTEM_PROMPT) + count(build_prompt("", message, 0, ""))
    return window_text(items, index, digests, PROMPT_BUDGET - rest - FEEDBACK_ROOM, count, rounds)


def segment_conversation(
    items: Sequence[Item],
    digests: Digests,
    done: set[str],
    ask: Ask,
    count: Count,
    workers: int,
    rounds: int,
) -> Iterator[MessageSegments]:
    """Yield the segments of every speech whose uuid is not in `done`, as each finishes.

    A backend failure does not stop the other messages, because a long run must keep
    what it can; LLMError is raised after the last one if any failed.
    """
    jobs = {
        index: item
        for index, item in enumerate(items)
        if isinstance(item, Speech) and item.source.uuid not in done
    }
    failures: list[LLMError] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_segment_at, items, index, message, digests, ask, count, rounds)
            for index, message in jobs.items()
        ]
        for future in as_completed(futures):
            try:
                yield future.result()
            except LLMError as error:
                failures.append(error)
                print(error, file=sys.stderr, flush=True)
    if failures:
        raise LLMError(
            f"{len(failures)} of {len(jobs)} messages failed, the rest are written; rerun the "
            f"same command to retry only those. First failure: {failures[0]}"
        )


def _segment_at(
    items: Sequence[Item],
    index: int,
    message: Speech,
    digests: Digests,
    ask: Ask,
    count: Count,
    rounds: int,
) -> MessageSegments:
    CALL_SUBJECT.set(f"message:{message.source.uuid}")
    try:
        window = message_window(items, index, digests, count, rounds)
    except WindowError as error:
        print(f"needs_review: {error}", file=sys.stderr, flush=True)
        segments = [_review_segment(message, 0, 0)]
    else:
        segments = segment_message(window, message, ask)
    return MessageSegments(uuid=message.source.uuid, segments=segments)


def load_done(path: Path) -> set[str]:
    """The uuids of the messages already in a segments file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {MessageSegments.model_validate_json(line).uuid for line in lines}


def main() -> None:
    """CLI entry point: segment every speech of one conversation, appending to --out."""
    parser = argparse.ArgumentParser(description="Segment speech with an LLM.")
    parser.add_argument("merged", type=Path)
    parser.add_argument("--digests", type=Path, required=True)
    parser.add_argument("--backend", choices=["claude", "codex", "hive"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="xhigh")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, help="np8 serves 8 slots")
    parser.add_argument("--window-rounds", type=window_rounds, default=DEFAULT_WINDOW_ROUNDS)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hive only: read the reply as a stream or as one block",
    )
    args = parser.parse_args()

    items = list(conversation_units(args.merged))
    done = load_done(args.out) if args.out.exists() else set()
    log = CallLog(path=args.out.parent / CALLS_FILE)
    ask = asker(
        args.backend, args.model, args.effort, SYSTEM_PROMPT, SCHEMA, args.stream, log, "segment"
    )
    results = segment_conversation(
        items, load_digests(args.digests), done, ask, qwen_tokens, args.workers, args.window_rounds
    )
    with args.out.open("a", encoding="utf-8") as out:
        for result in results:
            out.write(result.model_dump_json() + "\n")
            out.flush()
            print(f"{result.uuid}: {len(result.segments)} segments", flush=True)


if __name__ == "__main__":
    main()
