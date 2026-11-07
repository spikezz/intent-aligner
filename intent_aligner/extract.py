"""The main loop of trajectory extraction: the pointer and call T (topic assignment).

The pointer walks the units of a conversation in timeline order, a user message
replaced by its segments (intent_aligner.segment); AI text is context only and is
never a unit. A tool result takes the topic of the call that produced it; every
other unit goes to call T, which picks a registered topic or opens a new one.

The current round enters call T's window as level-1 digests, one per tool call and
result. When the prompt would not fit the budget even with one earlier round, the
round's tool records not yet compressed become one more level-2 passage, which stands
in for them from then on; the passages of earlier compressions stay as they are.

Calls A to D are not built yet, so a run writes these files under --out:

    assignments.jsonl      one line per unit in pointer order; a rerun resumes after the last
    tool_compressions.jsonl  one line per compression of a current round's tool records
    topics.yaml            the topic list (format in docs/trajectory-extractor.md), rebuilt
                           from assignments.jsonl, which is the only record of new topics
    topics/<id>.yaml       one per topic, with an empty graph until calls A to D fill it
    run.log                new topics and needs_review units, for a human to look over

Usage: python -m intent_aligner.extract MERGED.jsonl --digests DIGESTS.jsonl
           --segments SEGMENTS.jsonl --backend codex --model gpt-6.1-sol --out DIR
The digests and segments must cover the whole conversation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import attrs
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from intent_aligner.compress import ROUND_SYSTEM_PROMPT, SUMMARY_SCHEMA, summarize_trajectory
from intent_aligner.llm import Ask, asker
from intent_aligner.segment import MessageSegments, Segment
from intent_aligner.units import (
    AIText,
    Item,
    Observation,
    ToolCall,
    UserMessage,
    UserSignal,
    conversation_units,
)
from intent_aligner.window import (
    PROMPT_BUDGET,
    Count,
    Digests,
    ToolCompression,
    WindowError,
    is_tool_record,
    load_digests,
    qwen_tokens,
    render_round_so_far,
    round_entry,
    window_text,
)

MAX_ATTEMPTS = 2
# Room kept free for retry feedback, which quotes a pydantic error at most.
FEEDBACK_ROOM = 500
NEW = "NEW"
STORE_HEADER = "# 结构和填写规则见 docs/trajectory-extractor.md（路径相对于工具根目录）\n"

T_SYSTEM_PROMPT = """\
你在判断对话里的一个单元属于哪个 topic。topic 是对话里一件独立的事情：一个 session 里可以穿插几件互不相干的事，一件事也可以延续好几个 session。

<header> 是结构化的信息：topics 是已经登记的 topic 列表（编号和标题）；unit 是当前单元的类型、发出者和时间，用户段落还有它在消息正文里的字符区间 chars；当前单元是工具调用时，round_topic_hint 是这一轮前一个工具调用的 topic，AI 在一轮里通常做同一件事，但它只是参考。

<此前的对话> 是当前单元之前的对话。更早的轮次里，AI 的工具操作压成了每轮一段概括（【AI 的工具操作】）；当前这一轮已经发生的工具调用和工具结果逐条列出，每条是一句概括（【AI 的工具调用】【工具结果】）。

<当前单元> 是要判断的单元。用户段落给出整条消息，用 <当前段落> 和 </当前段落> 标出要判断的那一段；工具调用给出它的意图和操作对象的概括。

回答两项：
- topic：当前单元做的事属于哪个已有 topic，就回答它的编号；和已有的 topic 都不是同一件事时，回答 NEW。
- title：topic 为 NEW 时，写新 topic 的标题，一句完整的话，写清楚这件事是什么；topic 不是 NEW 时为空字符串。
"""

T_SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}, "title": {"type": "string"}},
    "required": ["topic", "title"],
    "additionalProperties": False,
}

UnitKind = Literal["user_segment", "user_signal", "tool_call", "observation"]


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class UserSegment:
    """One segment of a user message; the pointer visits it in place of the whole message."""

    message: UserMessage
    segment: Segment


Unit = UserSegment | UserSignal | ToolCall | Observation


class UnitRef(BaseModel):
    """Where a unit sits in the merged conversation; a resumed run checks it still matches."""

    model_config = ConfigDict(frozen=True)

    kind: UnitKind
    uuid: str
    tool_use_id: str | None = None
    chars: tuple[int, int] | None = None


class Assignment(BaseModel):
    """One line of assignments.jsonl; `topic` is None when the unit is unassigned.

    `new_title` is set on the unit that opened its topic. `review_reason` says why a
    unit needs review; `attempts` is 0 when no call T was made.
    """

    model_config = ConfigDict(frozen=True)

    unit: UnitRef
    topic: str | None
    new_title: str | None = None
    needs_review: bool = False
    review_reason: str | None = None
    attempts: int


class TopicEntry(BaseModel):
    """One topic of topics.yaml."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    file: str


class _TAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    topic: str
    title: str


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class _Step:
    """A unit, the timeline index of its item, and where that item's round opens.

    `round_start` is None for tool work before the first user message.
    """

    unit: Unit
    index: int
    round_start: int | None


def walk(items: Sequence[Item], segments: dict[str, list[Segment]]) -> Iterator[_Step]:
    """The units in timeline order."""
    round_start: int | None = None
    for index, item in enumerate(items):
        if isinstance(item, UserMessage):
            round_start = index
            if item.source.uuid not in segments:
                raise ValueError(
                    f"User message {item.source.uuid} has no segments; run "
                    "intent_aligner.segment on the whole conversation first."
                )
            for segment in segments[item.source.uuid]:
                unit = UserSegment(message=item, segment=segment)
                yield _Step(unit=unit, index=index, round_start=round_start)
        elif not isinstance(item, AIText):
            yield _Step(unit=item, index=index, round_start=round_start)


def unit_ref(unit: Unit) -> UnitRef:
    """The UnitRef of `unit`."""
    if isinstance(unit, UserSegment):
        chars = (unit.segment.start, unit.segment.end)
        return UnitRef(kind="user_segment", uuid=unit.message.source.uuid, chars=chars)
    kind: UnitKind = (
        "user_signal"
        if isinstance(unit, UserSignal)
        else "tool_call"
        if isinstance(unit, ToolCall)
        else "observation"
    )
    return UnitRef(kind=kind, uuid=unit.source.uuid, tool_use_id=unit.tool_use_id)


def topics_from(assignments: Sequence[Assignment], out: Path) -> list[TopicEntry]:
    """The topics the assignments opened, in opening order, with their files under `out`."""
    return [
        TopicEntry(id=a.topic, title=a.new_title, file=str(out / "topics" / f"{a.topic}.yaml"))
        for a in assignments
        if a.topic is not None and a.new_title is not None
    ]


def assign_topics(
    items: Sequence[Item],
    segments: dict[str, list[Segment]],
    digests: Digests,
    done: Sequence[Assignment],
    compressed: Sequence[ToolCompression],
    ask: Ask,
    ask_round: Ask,
    count: Count,
) -> Iterator[Assignment | ToolCompression]:
    """Yield an assignment for every unit after the `done` ones, and each new compression.

    The units run one at a time because each call T sees the topics opened before it.
    A compression is yielded before the assignment whose prompt needed it. `ask_round`
    must carry the level-2 prompt of intent_aligner.compress.
    Raises ValueError when `done` no longer lines up with the conversation's units.
    """
    compressions: dict[int, list[ToolCompression]] = {}
    for earlier in compressed:
        compressions.setdefault(earlier.start, []).append(earlier)
    topics = [(a.topic, a.new_title) for a in done if a.topic and a.new_title]
    context = _Context(items=items, digests=digests, topics=topics, count=count)
    call_topics: dict[str, str | None] = {}
    round_hint: str | None = None
    for position, step in enumerate(walk(items, segments)):
        ref = unit_ref(step.unit)
        if position < len(done):
            assignment = done[position]
            if assignment.unit != ref:
                raise ValueError(
                    f"Unit {position} is {ref} but assignments.jsonl has {assignment.unit}; "
                    "the conversation or its segments changed, so start a new --out."
                )
        else:
            hint = round_hint if isinstance(step.unit, ToolCall) else None
            in_force = compressions.setdefault(_round_low(step), [])
            compression = _fit_round(step, context, hint, in_force, ask_round)
            if compression is not None:
                in_force.append(compression)
                yield compression
            assignment = _assign(step, ref, context, hint, in_force, call_topics, ask)
            if assignment.new_title is not None:
                assert assignment.topic is not None
                topics.append((assignment.topic, assignment.new_title))
            yield assignment
        if isinstance(step.unit, UserSegment):
            round_hint = None
        elif isinstance(step.unit, ToolCall):
            call_topics[step.unit.tool_use_id] = assignment.topic
            round_hint = assignment.topic


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class _Context:
    """What every prompt of one run is built from; `topics` grows as topics open."""

    items: Sequence[Item]
    digests: Digests
    topics: Sequence[tuple[str, str]]
    count: Count


def _round_low(step: _Step) -> int:
    """Where the round shown before `step` starts; a user segment opens its round, so none is."""
    if isinstance(step.unit, UserSegment):
        return step.index
    return step.round_start or 0


def _fit_round(
    step: _Step,
    context: _Context,
    hint: str | None,
    in_force: Sequence[ToolCompression],
    ask_round: Ask,
) -> ToolCompression | None:
    """A new compression of `step`'s round if its prompt needs one, else None.

    The detector is the prompt itself: call T's prompt for this unit, with one earlier
    round, must fit the budget. A tool result never calls T, so records it adds are
    checked at the next unit that does; the round is compressed only when there is no
    other way to fit, and the next prompt is built at once from the new passage.
    """
    if not _needs_tool_call_t(step.unit):
        return None
    try:
        t_prompt(step, context, hint, in_force)
    except WindowError:
        pass
    else:
        return None
    low = _round_low(step)
    begin = in_force[-1].through if in_force else low
    pending = [item for item in context.items[begin : step.index] if is_tool_record(item)]
    if not pending:
        return None
    lines = [" ".join(round_entry(item, context.digests).split()) for item in pending]
    opening = context.items[low]
    message = opening if isinstance(opening, UserMessage) else None
    summary = summarize_trajectory(
        message, lines, ask_round, context.count, f"round at item {low} up to item {step.index}"
    )
    return ToolCompression(start=low, begin=begin, through=step.index, summary=summary)


def _needs_tool_call_t(unit: Unit) -> bool:
    return not (isinstance(unit, Observation) and unit.tool_use_id is not None)


def _assign(
    step: _Step,
    ref: UnitRef,
    context: _Context,
    hint: str | None,
    in_force: Sequence[ToolCompression],
    call_topics: dict[str, str | None],
    ask: Ask,
) -> Assignment:
    unit = step.unit
    if isinstance(unit, Observation) and unit.tool_use_id is not None:
        return Assignment(unit=ref, topic=call_topics[unit.tool_use_id], attempts=0)
    try:
        prompt = t_prompt(step, context, hint, in_force)
    except WindowError as error:
        return Assignment(
            unit=ref, topic=None, needs_review=True, review_reason=str(error), attempts=0
        )
    topics = context.topics
    ids = {topic_id for topic_id, _ in topics}
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer = _TAnswer.model_validate_json(ask(prompt + feedback))
        except ValidationError as error:
            feedback = f"\n\n你上一次的回答不是要求的 JSON，请重新回答：{error}"
            continue
        title = answer.title.strip()
        if answer.topic == NEW and title:
            topic_id = f"T{len(topics) + 1:03d}"
            return Assignment(unit=ref, topic=topic_id, new_title=title, attempts=attempt)
        if answer.topic in ids:
            return Assignment(unit=ref, topic=answer.topic, attempts=attempt)
        problem = (
            "topic 为 NEW 时 title 不能为空。"
            if answer.topic == NEW
            else f"{answer.topic!r} 不在 topics 列表里，只能回答列表里的编号或 NEW。"
        )
        feedback = f"\n\n你上一次的回答有问题，请重新回答：{problem}"
    return Assignment(
        unit=ref,
        topic=None,
        needs_review=True,
        review_reason=f"call T gave no valid answer in {MAX_ATTEMPTS} attempts:{feedback}",
        attempts=MAX_ATTEMPTS,
    )


def t_prompt(
    step: _Step, context: _Context, hint: str | None, in_force: Sequence[ToolCompression]
) -> str:
    """The call T prompt for `step`; raises WindowError when even one earlier round does not fit.

    The window is the earlier rounds (as in segmentation) followed by the current
    round up to this unit, so the unit sees what led to it but not how the round ended.
    """
    items, digests, count = context.items, context.digests, context.count
    round_start = step.round_start
    low = _round_low(step)
    so_far = render_round_so_far(items, low, step.index, digests, in_force)
    head = _header(step.unit, context.topics, hint)
    current = f"<当前单元>\n{_render_unit(step.unit, digests)}\n</当前单元>"
    rest = count(T_SYSTEM_PROMPT) + count(head) + count(so_far) + count(current)
    room = PROMPT_BUDGET - rest - FEEDBACK_ROOM
    if room < 0:
        raise WindowError(
            f"The current round up to item {step.index} ({items[step.index].source.uuid}) "
            f"needs {-room:,} tokens more than the budget; mark it needs_review."
        )
    earlier = "" if round_start is None else window_text(items, round_start, digests, room, count)
    window = "\n\n".join(part for part in (earlier, so_far) if part)
    return f"{head}\n\n<此前的对话>\n{window}\n</此前的对话>\n\n{current}"


def _header(unit: Unit, topics: Sequence[tuple[str, str]], hint: str | None) -> str:
    ref = unit_ref(unit)
    source = unit.message.source if isinstance(unit, UserSegment) else unit.source
    actor = {"user_segment": "user", "user_signal": "user", "tool_call": "ai"}.get(
        ref.kind, "environment"
    )
    described: dict[str, object] = {
        "kind": ref.kind,
        "actor": actor,
        "session": source.session,
        "timestamp": source.timestamp.isoformat(),
    }
    if ref.chars is not None:
        described["chars"] = list(ref.chars)
    header: dict[str, object] = {
        "topics": [{"id": topic_id, "title": title} for topic_id, title in topics],
        "unit": described,
    }
    if hint is not None:
        header["round_topic_hint"] = hint
    return f"<header>\n{json.dumps(header, ensure_ascii=False, indent=2)}\n</header>"


def _render_unit(unit: Unit, digests: Digests) -> str:
    if not isinstance(unit, UserSegment):
        return round_entry(unit, digests)
    body, segment = unit.message.body, unit.segment
    marked = (
        f"{body[: segment.start]}<当前段落>{body[segment.start : segment.end]}"
        f"</当前段落>{body[segment.end :]}"
    )
    images = "".join(f"\n[图片] {path}" for path in unit.message.images)
    return f"【用户】\n{marked}{images}"


def write_topic_files(out: Path, topics: Sequence[TopicEntry]) -> None:
    """Rewrite topics.yaml and create the store file of every topic that has none yet."""
    (out / "topics").mkdir(parents=True, exist_ok=True)
    listing = {"topics": [topic.model_dump() for topic in topics]}
    (out / "topics.yaml").write_text(STORE_HEADER + _yaml(listing), encoding="utf-8")
    for topic in topics:
        path = Path(topic.file)
        if not path.exists():
            store = {"meta": {"topic": topic.id, "graph": {"nodes": {}, "edges": []}}, "spans": []}
            path.write_text(STORE_HEADER + _yaml(store), encoding="utf-8")


def _yaml(data: object) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def load_assignments(path: Path) -> list[Assignment]:
    """The lines of an assignments.jsonl, or none when it does not exist yet."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Assignment.model_validate_json(line) for line in lines]


def load_compressions(path: Path) -> list[ToolCompression]:
    """The lines of a tool_compressions.jsonl, or none when it does not exist yet."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ToolCompression.model_validate_json(line) for line in lines]


def load_segments(path: Path) -> dict[str, list[Segment]]:
    """The segments file of intent_aligner.segment, by message uuid."""
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed = [MessageSegments.model_validate_json(line) for line in lines]
    return {message.uuid: message.segments for message in parsed}


def log_line(assignment: Assignment) -> str | None:
    """The run.log line for `assignment`, or None when it is not worth a human's look."""
    unit = assignment.unit
    where = f"{unit.kind} {unit.uuid}" + (f" chars {list(unit.chars)}" if unit.chars else "")
    if unit.tool_use_id:
        where += f" tool_use_id {unit.tool_use_id}"
    if assignment.new_title is not None:
        return f"new topic {assignment.topic}: {assignment.new_title} (opened by {where})"
    if assignment.review_reason is not None:
        return f"needs_review, unassigned: {where}: {assignment.review_reason}"
    return None


def main() -> None:
    """CLI entry point: assign every unit of one conversation to a topic, appending to --out."""
    parser = argparse.ArgumentParser(description="Assign conversation units to topics.")
    parser.add_argument("merged", type=Path)
    parser.add_argument("--digests", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--backend", choices=["claude", "codex", "hive"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="xhigh")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hive only: read the reply as a stream or as one block",
    )
    args = parser.parse_args()

    items = list(conversation_units(args.merged))
    assignments_path = args.out / "assignments.jsonl"
    done = load_assignments(assignments_path)
    topics = topics_from(done, args.out)
    write_topic_files(args.out, topics)
    compressions_path = args.out / "tool_compressions.jsonl"
    llm = (args.backend, args.model, args.effort)
    results = assign_topics(
        items,
        load_segments(args.segments),
        load_digests(args.digests),
        done,
        load_compressions(compressions_path),
        asker(*llm, T_SYSTEM_PROMPT, T_SCHEMA, args.stream),
        asker(*llm, ROUND_SYSTEM_PROMPT, SUMMARY_SCHEMA, args.stream),
        qwen_tokens,
    )
    position = len(done)
    with (
        assignments_path.open("a", encoding="utf-8") as out,
        compressions_path.open("a", encoding="utf-8") as compressions,
        (args.out / "run.log").open("a", encoding="utf-8") as log,
    ):
        for assignment in results:
            if isinstance(assignment, ToolCompression):
                compressions.write(assignment.model_dump_json() + "\n")
                compressions.flush()
                print(
                    f"compressed the tool records of items {assignment.start}-"
                    f"{assignment.through - 1}",
                    flush=True,
                )
                continue
            out.write(assignment.model_dump_json() + "\n")
            out.flush()
            if assignment.new_title is not None:
                topics = [*topics, *topics_from([assignment], args.out)]
                write_topic_files(args.out, topics)
            line = log_line(assignment)
            if line is not None:
                log.write(line + "\n")
                log.flush()
                print(line, file=sys.stderr, flush=True)
            print(f"{position}: {assignment.unit.kind} -> {assignment.topic}", flush=True)
            position += 1


if __name__ == "__main__":
    main()
