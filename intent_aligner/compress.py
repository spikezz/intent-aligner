"""Compress each tool call and its result, before the main loop runs (level 1).

The call becomes its intents and what it acts on, the result one sentence per chunk,
read against the call's intents and full text. The call's intents, each with evidence
in the call text, are also the intents the main loop records for the call, so it does
not ask for them again. A call and its result are one job; jobs read nothing of the
conversation around them and run in parallel.

summarize_trajectory is what the main loop calls when a prompt would not fit: it turns
the oldest tool records of the window into one passage (intent_aligner.window).

Usage: python -m intent_aligner.compress MERGED.jsonl --backend hive
           --model deployment/pc-qwen38-np6 --out DIGESTS.jsonl
           [--workers 8] [--until UUID]
The output is appended to as digests are made, so a stopped run resumes where it was.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

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
from intent_aligner.units import (
    SPEAKER_NAMES,
    Item,
    Observation,
    Speech,
    ToolCall,
    UserMessage,
    conversation_units,
)
from intent_aligner.window import (
    DIGEST_ADAPTER,
    PROMPT_BUDGET,
    CallDigest,
    Count,
    Digest,
    Digests,
    ResultDigest,
    WindowError,
    load_digests,
    qwen_tokens,
)

MAX_ATTEMPTS = 3
# Room kept free for the retry feedback a second attempt appends.
FEEDBACK_ROOM = 1_000
# Every chunk of a result repeats the call's full text, so a call must leave this much
# of the prompt for the result; the largest call in the corpus (6,116 tokens) leaves 1,650.
MIN_RESULT_ROOM = 1_000
# A round's opening speech is repeated in every batch of its trajectory, so it is cut to
# this many tokens; the corpus has single messages (38cb90a8) larger than a whole prompt.
MESSAGE_ROOM = 2_000
# Room for the "（第 i/n 段）" label on a chunked result.
LABEL_ROOM = 50

CALL_SYSTEM_PROMPT = """\
你在提炼 AI 助手一次工具调用的意图，让后面读对话的人不看原文也知道这次调用是做什么的。<调用> 是这次调用的完整原文。

回答两项：
- intents：这次调用想达成什么。只写意图，不写具体怎么做。一次调用通常只有一个意图；同时做了几件意图不同的事时，每个意图各写一项。每项回答四项：
  - text：意图，一句话。
  - evidence：这个意图在 <调用> 原文里的依据，原样摘录。可以用 … 省略中间的部分，但被 … 分开的每个片段都必须原样出现在 <调用> 里，并且按原文的先后顺序排列。
  - explicit：调用里写明了这个意图就是 true，比如 Bash 调用的 description 字段、代码里说明用途的注释；意图是从命令、路径和参数推断出来的，就是 false。
  - note：explicit 为 false 时写推断依据：从调用里的什么推断出这个意图。explicit 为 true 时写空字符串。
- targets：这次调用针对的具体对象：路径、文件名、关键参数和数值。
"""

RESULT_SYSTEM_PROMPT = """\
你在压缩一次工具调用返回的结果，让后面读对话的人不看原文也知道结果说了什么。<调用> 是产生这个结果的调用：它的意图和原文；<结果> 是返回的原文，结果太长时只是其中一段，会注明是第几段。

回答 summary：<结果> 说了什么，一句话。对照调用的意图写出结论：意图达成了、没达成（为什么），或者得到了什么。只保留这个结论直接依据的具体信息，例如决定结论的数值、状态、报错的关键原文；其余细节归纳进结论，不逐项列出。
"""

ROUND_SYSTEM_PROMPT = """\
你在概括 AI 助手连续做的一段工具操作。这段概括会放进对话记录，代替这些操作的逐条记录，供人判断后面的意图时阅读：读的人要靠它知道 AI 这段时间在为什么事做操作、做到了哪一步、发现了什么。<发言> 是紧挨在这段操作前面的那次发言，开头注明是用户说的还是 AI 说的，太长时只保留开头，会注明；注明了“AI 工作时发出”的用户消息，AI 也可能还在接着做之前的请求。对话开头没有发言时，没有这一项。<轨迹> 按时间顺序列出每次工具调用的意图、对象和结果，或者更细的轨迹已经写成的几段概括；太长时 <轨迹> 只是其中一段，会注明是第几段，这时只概括这一段。

回答 summary：一小段话，写出 AI 在为什么事做这些操作、最后的状态和结论、还没解决的问题。只保留和最后结论直接相关的对象和数值，例如结论依据的关键数值、改动或产出的文件；单次操作的细节和其余数值归纳进主线，不逐项列出。只供 AI 自己中转的东西不写，例如后台任务 ID 和临时输出文件名。
"""

CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "intents": {"type": "array", "items": INTENT_SCHEMA, "minItems": 1},
        "targets": {"type": "string"},
    },
    "required": ["intents", "targets"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


class _CallAnswer(BaseModel):
    """Validated with the rendered call as context, which every evidence must quote."""

    model_config = ConfigDict(frozen=True)

    intents: list[Intent] = Field(min_length=1)
    targets: str

    @model_validator(mode="after")
    def _evidence_quotes_the_call(self, info: ValidationInfo) -> _CallAnswer:
        assert isinstance(info.context, str), "validate with the call text as context"
        strays = stray_evidence(self.intents, info.context)
        if strays:
            raise ValueError(
                f"这些 evidence 不是 <调用> 原文的摘录：{strays!r}。evidence 被 … 分开的每个片段"
                "都必须原样出现在 <调用> 里，并且按原文的先后顺序排列。"
            )
        return self


class _SummaryAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str


def compress_conversation(
    items: Sequence[Item],
    done: Digests,
    ask_call: Ask,
    ask_result: Ask,
    count: Count,
    workers: int,
) -> Iterator[Digest]:
    """Yield the digests missing from `done`, adding them there.

    A failed job does not stop the others, because a long run must keep what it can;
    LLMError is raised after the last job when any failed.
    """
    results = {
        item.tool_use_id: item
        for item in items
        if isinstance(item, Observation) and item.origin == "tool_result"
    }
    pending = results.keys() - done.results.keys()
    jobs = [
        partial(
            _compress_pair, item, results.get(item.tool_use_id), done, ask_call, ask_result, count
        )
        for item in items
        if isinstance(item, ToolCall)
        and (item.tool_use_id not in done.calls or item.tool_use_id in pending)
    ]
    yield from _run_jobs(jobs, done, workers)


def _run_jobs(
    jobs: Sequence[Callable[[], list[Digest]]], done: Digests, workers: int
) -> Iterator[Digest]:
    failures: list[LLMError] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(job) for job in jobs]):
            try:
                made = future.result()
            except LLMError as error:
                failures.append(error)
                print(error, file=sys.stderr, flush=True)
                continue
            for digest in made:
                done.add(digest)
                yield digest
    if failures:
        raise LLMError(
            f"{len(failures)} of {len(jobs)} jobs failed, the rest are written; rerun the same "
            f"command to retry only those. First failure: {failures[0]}"
        )


def _compress_pair(
    call: ToolCall,
    result: Observation | None,
    done: Digests,
    ask_call: Ask,
    ask_result: Ask,
    count: Count,
) -> list[Digest]:
    """A call (unless done) then its result; a rejected or unanswered call has no result."""
    CALL_SUBJECT.set(f"tool:{call.tool_use_id}")
    made: list[Digest] = []
    digest = done.calls.get(call.tool_use_id)
    if digest is None:
        digest = _compress_call(call, ask_call, count)
        made.append(digest)
    if result is not None and call.tool_use_id not in done.results:
        made.append(_compress_result(result, call, digest, ask_result, count))
    return made


def _compress_call(item: ToolCall, ask: Ask, count: Count) -> CallDigest:
    prompt = f"<调用>\n{item.text}\n</调用>"
    room = PROMPT_BUDGET - count(CALL_SYSTEM_PROMPT) - FEEDBACK_ROOM
    if count(prompt) > room:
        raise LLMError(
            f"Tool call {item.tool_use_id} needs {count(prompt):,} tokens but the prompt budget "
            f"leaves {room:,}; the largest call in the corpus was 6,116, so check its input."
        )
    answer = _ask_until_valid(ask, prompt, _CallAnswer, item.tool_use_id, context=item.text)
    return CallDigest(tool_use_id=item.tool_use_id, intents=answer.intents, targets=answer.targets)


def _compress_result(
    item: Observation, call: ToolCall, digest: CallDigest, ask: Ask, count: Count
) -> ResultDigest:
    head = f"<调用>\n意图：{digest.purpose}\n原文：\n{call.text}\n</调用>\n\n"
    room = PROMPT_BUDGET - count(RESULT_SYSTEM_PROMPT) - count(head) - FEEDBACK_ROOM - LABEL_ROOM
    if room < MIN_RESULT_ROOM:
        raise LLMError(
            f"Tool call {call.tool_use_id} leaves {room:,} tokens for each chunk of its result, "
            f"below {MIN_RESULT_ROOM:,}; check the call's input."
        )
    parts = chunks(item.text, room, count)
    summaries = []
    for number, part in enumerate(parts, 1):
        label = f"（第 {number}/{len(parts)} 段）" if len(parts) > 1 else ""
        prompt = f"{head}<结果>{label}\n{part}\n</结果>"
        summaries.append(_ask_until_valid(ask, prompt, _SummaryAnswer, call.tool_use_id).summary)
    return ResultDigest(tool_use_id=call.tool_use_id, summaries=summaries)


def summarize_trajectory(
    speech: Speech | None, trajectory: list[str], ask: Ask, count: Count, where: str
) -> str:
    """One passage for `trajectory`, the level-one lines of consecutive tool records.

    `ask` must carry ROUND_SYSTEM_PROMPT and SUMMARY_SCHEMA; `where` names the records
    in the errors raised. Raises WindowError when the speech leaves no room for them.
    """
    head = _speech_block(speech, count)
    room = PROMPT_BUDGET - count(ROUND_SYSTEM_PROMPT) - count(head) - FEEDBACK_ROOM - LABEL_ROOM
    if room <= 0:
        raise WindowError(
            f"{where}: the compression prompt has no room left for the trajectory "
            f"({room} tokens after the system prompt and the speech)"
        )
    lines = trajectory
    while True:
        parts = chunks("\n".join(lines), room, count)
        summaries = []
        for number, part in enumerate(parts, 1):
            label = f"（第 {number}/{len(parts)} 段）" if len(parts) > 1 else ""
            prompt = f"{head}<轨迹>{label}\n{part}\n</轨迹>"
            summaries.append(_ask_until_valid(ask, prompt, _SummaryAnswer, where).summary)
        if len(summaries) == 1:
            return summaries[0]
        lines = [" ".join(summary.split()) for summary in summaries]


def _speech_block(speech: Speech | None, count: Count) -> str:
    """The `<发言>` block a round prompt starts with; empty for tool work before any speech."""
    if speech is None:
        return ""
    text = speech.body + "".join(f"\n[图片] {path}" for path in speech.images)
    queued = isinstance(speech, UserMessage) and speech.queued
    label = f"（{SPEAKER_NAMES[speech.speaker]}）" + ("（AI 工作时发出）" if queued else "")
    if count(text) > MESSAGE_ROOM:
        text, label = chunks(text, MESSAGE_ROOM, count)[0], f"{label}（太长，只保留开头）"
    return f"<发言>{label}\n{text}\n</发言>\n"


def chunks(text: str, room: int, count: Count) -> list[str]:
    """Cut `text` at line ends into pieces of at most `room` tokens; longer lines are cut inside.

    Examples:
        >>> chunks("ab\\ncd\\nef", 6, len)
        ['ab\\ncd\\n', 'ef']
        >>> chunks("abcdefgh", 3, len)
        ['abc', 'def', 'gh']
    """
    pieces: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if current and count(current + line) > room:
            pieces.append(current)
            current = ""
        while count(line) > room:
            cut = _longest_prefix(line, room, count)
            pieces.append(line[:cut])
            line = line[cut:]
        current += line
    if current:
        pieces.append(current)
    return pieces


def _longest_prefix(line: str, room: int, count: Count) -> int:
    """Binary search, because token counts do not grow in step with characters."""
    low, high = 1, len(line)
    while low < high:
        middle = (low + high + 1) // 2
        if count(line[:middle]) <= room:
            low = middle
        else:
            high = middle - 1
    return low


def _ask_until_valid[Answer: BaseModel](
    ask: Ask, prompt: str, schema: type[Answer], tool_use_id: str, context: str | None = None
) -> Answer:
    feedback = ""
    for _ in range(MAX_ATTEMPTS):
        try:
            return schema.model_validate_json(ask(prompt + feedback), context=context)
        except TruncatedReply:
            feedback = f"\n\n{TRUNCATED_FEEDBACK}"
        except ValidationError as error:
            feedback = f"\n\n你上一次的回答不符合要求，请重新回答：{error}"
    raise LLMError(f"No valid digest for {tool_use_id} in {MAX_ATTEMPTS} attempts: {feedback}")


def _until(items: list[Item], uuid: str | None) -> list[Item]:
    """Items before the user message `uuid`, or all of them."""
    if uuid is None:
        return items
    for index, item in enumerate(items):
        if isinstance(item, UserMessage) and item.source.uuid == uuid:
            return items[:index]
    raise ValueError(
        f"No user message {uuid}; the units file that `python -m intent_aligner.units` writes "
        "for this conversation lists every uuid."
    )


def main() -> None:
    """CLI entry point: compress one merged conversation, appending digests to --out."""
    parser = argparse.ArgumentParser(description="Compress tool calls and results with an LLM.")
    parser.add_argument("merged", type=Path)
    parser.add_argument("--backend", choices=["claude", "codex", "hive"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="xhigh")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6, help="np6 serves 6 slots")
    parser.add_argument("--until", help="stop before this user message uuid")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hive only: read the reply as a stream or as one block",
    )
    args = parser.parse_args()

    items = _until(list(conversation_units(args.merged)), args.until)
    done = load_digests(args.out) if args.out.exists() else Digests()
    llm_args = (args.backend, args.model, args.effort)
    log = CallLog(path=args.out.parent / CALLS_FILE)
    ask_call = asker(*llm_args, CALL_SYSTEM_PROMPT, CALL_SCHEMA, args.stream, log, "call")
    ask_result = asker(*llm_args, RESULT_SYSTEM_PROMPT, SUMMARY_SCHEMA, args.stream, log, "result")
    digests = compress_conversation(items, done, ask_call, ask_result, qwen_tokens, args.workers)
    with args.out.open("a", encoding="utf-8") as out:
        for digest in digests:
            line = DIGEST_ADAPTER.dump_json(digest).decode()
            out.write(line + "\n")
            out.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
