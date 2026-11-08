"""Replay a pipeline run as one self-contained HTML page, from its records and its calls' transcripts.

Usage: python -m intent_aligner.replay RUN_DIR --out PAGE.html [--since T] [--until T]

RUN_DIR holds a run.sh that runs intent_aligner.extract (and usually .compress and
.segment before it); run.sh names the conversation, the model and every file the run
wrote. Every endpoint call appears with its request and reply exactly as its
transcript in llm-home holds them.

A run whose commands wrote calls.jsonl names its own transcripts. An older run did not,
so its transcripts are found in llm-home by model, by system prompt and by the
conversation text in their prompts. Nothing in a transcript tells two runs of one
conversation on one model apart, so when another run directory has the same
conversation and model, --since/--until must bound this run's time.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import attrs
from pydantic import BaseModel, ConfigDict

from intent_aligner import compress, extract, segment
from intent_aligner.corpus import TOOL_ROOT
from intent_aligner.extract import (
    Assignment,
    Unit,
    load_assignments,
    load_compressions,
    load_segments,
    render_unit,
    walk,
)
from intent_aligner.llm import CALLS_FILE, LLM_HOME, CallRecord
from intent_aligner.segment import Segment, message_window
from intent_aligner.units import Item, ToolCall, UserMessage, conversation_units
from intent_aligner.window import Digests, load_digests, qwen_tokens

TEMPLATE = Path(__file__).with_name("replay.html")

# The extract command's inline "round" asks share compress's system prompt; which
# command made one is known from calls.jsonl, or else from when it ran (see _allocate).
STEPS: dict[str, tuple[str, str]] = {
    compress.CALL_SYSTEM_PROMPT: ("L1", "call"),
    compress.RESULT_SYSTEM_PROMPT: ("L1", "result"),
    compress.ROUND_SYSTEM_PROMPT: ("L2", "round"),
    segment.SYSTEM_PROMPT: ("SEG", "segment"),
    extract.T_SYSTEM_PROMPT: ("LOOP", "topic"),
    extract.A_SYSTEM_PROMPT: ("LOOP", "signal"),
    extract.B_SYSTEM_PROMPT: ("LOOP", "registry"),
    extract.G_SYSTEM_PROMPT: ("LOOP", "aggregate"),
    extract.L_SYSTEM_PROMPT: ("LOOP", "tool_intent"),
    extract.C_SYSTEM_PROMPT: ("LOOP", "relation"),
    extract.D_SYSTEM_PROMPT: ("LOOP", "outcome"),
    extract.D_SUMMARY_SYSTEM_PROMPT: ("LOOP", "outcome_summary"),
}
# The letter each extract ask goes by in review_reason ("call T gave no valid answer ...").
LETTERS = {
    "topic": "T",
    "signal": "A",
    "registry": "B",
    "aggregate": "G",
    "tool_intent": "L",
    "relation": "C",
    "outcome": "D",
    "outcome_summary": "D",
}


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Run:
    """What run.sh says: the conversation, the model and the files the commands wrote."""

    directory: Path
    merged: Path
    backend: str
    model: str
    digests: Path
    segments: Path
    extract: Path


def parse_run(directory: Path) -> Run:
    """Read run.sh: its variables, then the arguments of its intent_aligner.extract command."""
    script = directory / "run.sh"
    if not script.exists():
        raise ValueError(f"{script} not found; a replay reads the run's commands from its run.sh.")
    variables: dict[str, str] = {}
    base = TOOL_ROOT
    commands: dict[str, dict[str, str]] = {}
    for raw in script.read_text(encoding="utf-8").replace("\\\n", " ").splitlines():
        line = re.sub(r"\$\{?(\w+)\}?", lambda m: variables.get(m.group(1), m.group(0)), raw)
        assignment = re.fullmatch(r"\s*(\w+)=(.*)", line)
        if assignment:
            variables[assignment.group(1)] = " ".join(shlex.split(assignment.group(2)))
        elif line.strip().startswith("cd "):
            base = Path(shlex.split(line)[1])
        module = re.search(r"-m intent_aligner\.(\w+)\s+(.*?)\s*(?:>|\|\||$)", line)
        if module:
            commands[module.group(1)] = _options(shlex.split(module.group(2)))
    if "extract" not in commands:
        raise ValueError(f"{script} does not run intent_aligner.extract.")
    options = commands["extract"]
    return Run(
        directory=directory,
        merged=base / options[""],
        backend=options["--backend"],
        model=options["--model"],
        digests=base / options["--digests"],
        segments=base / options["--segments"],
        extract=base / options["--out"],
    )


def _options(tokens: Sequence[str]) -> dict[str, str]:
    """`--flag value` pairs; the positional argument is under ""."""
    options: dict[str, str] = {}
    rest = list(tokens)
    while rest:
        token = rest.pop(0)
        if token.startswith("--"):
            options[token] = rest.pop(0) if rest and not rest[0].startswith("--") else ""
        else:
            options[""] = token
    return options


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Exchange:
    """One transcript: the request as sent and the reply as received.

    `ended` is the file's mtime: the backend writes the transcript's last line when the
    reply arrives, and nothing touches it afterwards.
    """

    path: Path
    model: str
    system: str
    prompt: str
    schema: dict[str, Any] | None
    reply: str | None
    reasoning: str | None
    usage: dict[str, Any]
    finish: str
    ended: datetime


def read_exchange(path: Path) -> Exchange | None:
    """A codex or hive transcript; None for the backends whose transcripts lack the request."""
    # cast: transcripts are written by intent_aligner.llm, one JSON object per line.
    lines = [cast(dict[str, Any], json.loads(line)) for line in path.read_text().splitlines()]
    request = lines[0] if lines else {}
    if request.get("type") != "request":
        return None
    reply: str | None = None
    reasoning: str | None = None
    usage: dict[str, Any] = {}
    finish = "no reply"
    for line in lines[1:]:
        kind = line.get("type")
        if kind == "item.completed" and line["item"]["type"] == "agent_message":
            reply, finish = str(line["item"]["text"]), "stop"
        elif kind == "turn.completed":
            usage = line["usage"]
        elif kind == "response" and line["status"] != 200:
            finish = f"HTTP {line['status']}"
        elif kind == "response":
            choice = line["body"]["choices"][0]
            reply, finish = choice["message"]["content"], choice["finish_reason"]
            reasoning = choice["message"].get("reasoning_content")
            usage = line["body"].get("usage", {})
    return Exchange(
        path=path,
        model=request["model"],
        system=request["system_prompt"],
        prompt=request["prompt"],
        schema=request.get("json_schema"),
        reply=reply,
        reasoning=reasoning,
        usage=usage,
        finish=finish,
        ended=datetime.fromtimestamp(path.stat().st_mtime, UTC),
    )


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Conversation:
    """The run's conversation and what the run wrote about it.

    `units` are the extract units' <当前单元> blocks ("" while their digests are
    missing) and `unit_items` their timeline indexes. `round_of` maps a tool call to
    the user message that opened its round, "" before the first one.
    """

    items: Sequence[Item]
    digests: Digests
    segments: dict[str, list[Segment]]
    assignments: list[Assignment]
    units: list[str]
    unit_items: list[int]
    round_of: dict[str, str]

    @classmethod
    def load(cls, run: Run) -> Conversation:
        """Everything the run wrote so far."""
        items = list(conversation_units(run.merged))
        digests = load_digests(run.digests) if run.digests.exists() else Digests()
        segments = load_segments(run.segments) if run.segments.exists() else {}
        done = run.extract / "assignments.jsonl"
        steps = list(walk(items, segments)) if segments else []
        round_of: dict[str, str] = {}
        opening = ""
        for item in items:
            if isinstance(item, UserMessage):
                opening = item.source.uuid
            elif isinstance(item, ToolCall):
                round_of[item.tool_use_id] = opening
        return cls(
            items=items,
            digests=digests,
            segments=segments,
            assignments=load_assignments(done) if done.exists() else [],
            units=[_unit_block(step.unit, digests) for step in steps],
            unit_items=[step.index for step in steps],
            round_of=round_of,
        )


def _unit_block(unit: Unit, digests: Digests) -> str:
    try:
        return f"<当前单元>\n{render_unit(unit, digests)}\n</当前单元>"
    except KeyError:
        return ""


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Call:
    """One endpoint call placed in the run.

    `subject` is the job it served: "tool:<id>", "round:<uuid>",
    "message:<uuid>:<start>" or "unit:<position>". `started` is known only from calls.jsonl.
    """

    stage: str
    ask: str
    subject: str
    started: datetime | None
    exchange: Exchange
    error: str | None

    @property
    def at(self) -> datetime:
        """When the call began, or when it ended if that is all the transcript says."""
        return self.started or self.exchange.ended


def logged_calls(run: Run) -> list[Call] | None:
    """The calls named by the run's calls.jsonl files, or None for a run older than them."""
    logs = [(run.digests.parent / CALLS_FILE, False), (run.extract / CALLS_FILE, True)]
    if not any(path.exists() for path, _ in logs):
        return None
    calls = []
    for path, in_extract in logs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = CallRecord.model_validate_json(line)
            exchange = None if record.transcript is None else read_exchange(Path(record.transcript))
            if exchange is None:
                continue
            subject = record.subject
            if record.ask == "segment":
                subject = f"{subject}:{len(_split(exchange.prompt)[0])}"
            calls.append(
                Call(
                    stage="LOOP" if in_extract else STEPS[exchange.system][0],
                    ask=record.ask,
                    subject=subject,
                    started=record.started,
                    exchange=exchange,
                    error=record.error,
                )
            )
    return calls


def _split(prompt: str) -> tuple[str, str]:
    """A segment prompt's <已处理> and <剩余> texts."""
    done, rest = prompt.split("<已处理>\n", 1)[1].split("\n</已处理>\n\n<剩余>\n", 1)
    return done, rest.rsplit("\n</剩余>", 1)[0]


def matched_calls(
    run: Run, conversation: Conversation, since: datetime | None, until: datetime | None
) -> tuple[list[Call], list[Path], datetime | None]:
    """The run's calls found in llm-home, the transcripts in its span left unplaced, and its start.

    A call is the run's when its model is run.model, its system prompt is one the
    current source sends, and its prompt names a job of this conversation. Earlier runs
    of the same conversation match too, since a user segment's prompt can be the same
    in every run; without --since the run starts after the last pause of RUN_GAP
    between matched calls before the run's last output. Unplaced transcripts are this
    model's calls inside the run's span that matched no job: the code that sent them
    has changed since, or they belong to a run this replay cannot see.
    """
    found: list[tuple[Exchange, str, str, list[str]]] = []
    others: list[Exchange] = []
    finder = _JobFinder(conversation=conversation)
    for path in sorted((LLM_HOME / run.backend).glob("*.jsonl")):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if (since and mtime < since) or (until and mtime > until):
            continue
        exchange = read_exchange(path)
        if exchange is None or exchange.model != run.model:
            continue
        stage, ask = STEPS.get(exchange.system, ("", ""))
        jobs = finder.jobs(ask, exchange.prompt) if ask else []
        if jobs:
            found.append((exchange, stage, ask, jobs))
        else:
            others.append(exchange)
    found.sort(key=lambda f: f[0].ended)
    if since is None:
        since = _run_start([f[0].ended for f in found], _last_output(run))
        found = [f for f in found if since is None or f[0].ended >= since]
    calls = _allocate(found, finder)
    if not calls:
        return [], [], since
    first, last = calls[0].exchange.ended, calls[-1].exchange.ended
    return calls, [e.path for e in others if first <= e.ended <= last], since


# The longest pause between two calls of one run: extract waits on one call at a time
# and the slowest sol calls took about 10 minutes, while separate runs of a
# conversation were hours apart.
RUN_GAP = timedelta(minutes=30)


def _run_start(times: Sequence[datetime], last_output: datetime) -> datetime | None:
    """The first call time after the last RUN_GAP pause before `last_output`."""
    inside = [t for t in times if t <= last_output]
    if not inside:
        return None
    start = inside[-1]
    for earlier in reversed(inside[:-1]):
        if start - earlier > RUN_GAP:
            break
        start = earlier
    return start


def _last_output(run: Run) -> datetime:
    """When the run last wrote a record, plus a minute for the file system's clock."""
    files = [run.digests, run.segments, run.extract / "assignments.jsonl"]
    latest = max(f.stat().st_mtime for f in files if f.exists())
    return datetime.fromtimestamp(latest, UTC) + timedelta(minutes=1)


@attrs.define(slots=True, kw_only=True)
class _JobFinder:
    """Which jobs of the conversation a prompt could serve, from the text it quotes.

    `unit_pointer` is the unit of the last extract call placed: extract walks the
    units one at a time, so each call serves that unit or a later one.
    """

    conversation: Conversation
    tools_by_text: dict[str, list[str]] = attrs.field(factory=lambda: defaultdict(list))
    tools_by_line: dict[str, list[str]] = attrs.field(factory=lambda: defaultdict(list))
    messages_by_body: dict[str, list[int]] = attrs.field(factory=lambda: defaultdict(list))
    unit_pointer: int = 0

    def __attrs_post_init__(self) -> None:
        digests = self.conversation.digests
        for index, item in enumerate(self.conversation.items):
            if isinstance(item, ToolCall):
                self.tools_by_text[item.text].append(item.tool_use_id)
                digest = digests.calls.get(item.tool_use_id)
                if digest is not None:
                    line = f"{item.name} 意图：{digest.purpose} 对象：{digest.targets}"
                    self.tools_by_line[" ".join(line.split())].append(item.tool_use_id)
            elif isinstance(item, UserMessage):
                self.messages_by_body[item.body].append(index)

    def jobs(self, ask: str, prompt: str) -> list[str]:
        """Candidate subjects in timeline order."""
        if ask == "call":
            text = prompt.removeprefix("<调用>\n").split("\n</调用>", 1)[0]
            return [f"tool:{t}" for t in self.tools_by_text.get(text, [])]
        if ask == "result":
            text = prompt.split("\n原文：\n", 1)[-1].split("\n</调用>\n\n<结果>", 1)[0]
            return [f"tool:{t}" for t in self.tools_by_text.get(text, [])]
        if ask == "round":
            return self._rounds(prompt)
        if ask == "segment":
            return self._segments(prompt)
        units = self.conversation.units
        return [f"unit:{n}" for n, block in enumerate(units) if block and block in prompt]

    def _rounds(self, prompt: str) -> list[str]:
        """The round of the first trajectory line naming a known call.

        compress writes lines as "name 意图：… 对象：… 结果：…"; extract's inline
        compression writes each round entry on one line, headed 【AI 的工具调用】.
        """
        for line in prompt.split("<轨迹>", 1)[-1].splitlines():
            entry = " ".join(line.split()).removeprefix("【AI 的工具调用】 ").split(" 结果：", 1)[0]
            if entry in self.tools_by_line:
                return [f"round:{self.conversation.round_of[t]}" for t in self.tools_by_line[entry]]
        return []

    def _segments(self, prompt: str) -> list[str]:
        """The message being split; identical messages are told apart by their window."""
        if "<已处理>\n" not in prompt:
            return []
        done, rest = _split(prompt)
        indexes = self.messages_by_body.get(done + rest, [])
        items, digests = self.conversation.items, self.conversation.digests
        if len(indexes) > 1:
            indexes = [
                i
                for i in indexes
                if prompt.startswith(
                    f"<此前的对话>\n{message_window(items, i, digests, qwen_tokens)}\n</此前的对话>"
                )
            ] or indexes
        return [f"message:{items[i].source.uuid}:{len(done)}" for i in indexes]

    def unit(self, candidates: list[str], prompt: str) -> str:
        """The unit an extract call served among identical-looking candidates.

        Units whose blocks are identical (the same tool call twice) differ in the
        header's timestamp, which every extract prompt carries.
        """
        positions = [int(c.split(":")[1]) for c in candidates]
        ahead = [p for p in positions if p >= self.unit_pointer] or positions
        items, unit_items = self.conversation.items, self.conversation.unit_items
        stamped = [
            p
            for p in ahead
            if f'"timestamp": "{items[unit_items[p]].source.timestamp.isoformat()}"' in prompt
        ]
        self.unit_pointer = (stamped or ahead)[0]
        return f"unit:{self.unit_pointer}"


def _allocate(found: list[tuple[Exchange, str, str, list[str]]], finder: _JobFinder) -> list[Call]:
    """Give each transcript one subject; `found` is in the order the transcripts ended.

    A retry carries its first attempt's prompt plus feedback, so it takes that
    attempt's subject. Identical fresh prompts (two tool calls with the same text) go
    to their candidates in turn. A "round" ask after the first extract call is the
    extract command's inline compression for the unit whose call comes next.
    """
    loop_times = [e.ended for e, stage, _, _ in found if stage == "LOOP"]
    asked: dict[tuple[str, str], int] = defaultdict(int)
    subjects: dict[str, str] = {}
    calls: list[Call] = []
    for exchange, stage, ask, jobs in found:
        inline = ask == "round" and bool(loop_times) and exchange.ended > loop_times[0]
        retry_of = next(
            (s for base, s in subjects.items() if _extends(exchange.prompt, base)), None
        )
        if stage == "LOOP":
            subject = finder.unit(jobs, exchange.prompt)
        elif retry_of is not None:
            subject = retry_of
        else:
            n = asked[(ask, exchange.prompt)]
            asked[(ask, exchange.prompt)] += 1
            subject = jobs[min(n, len(jobs) - 1)]
        subjects[exchange.prompt] = subject
        calls.append(
            Call(
                stage="LOOP" if inline else stage,
                ask=ask,
                subject=subject,
                started=None,
                exchange=exchange,
                error=None if exchange.finish == "stop" else exchange.finish,
            )
        )
    for n, call in enumerate(calls):
        if call.stage == "LOOP" and call.ask == "round":
            after = next((c for c in calls[n + 1 :] if c.ask in LETTERS), None)
            if after is not None:
                calls[n] = attrs.evolve(call, subject=after.subject)
    return calls


def _extends(prompt: str, base: str) -> bool:
    """Whether `prompt` is `base` with retry feedback appended (always after a blank line)."""
    return len(prompt) > len(base) and prompt.startswith(base + "\n\n")


class Attempt(BaseModel):
    """One call as the page shows it, judged among the calls of its question.

    A question is a first prompt and the retries that extend it with feedback;
    `feedback_at` is where a retry's feedback starts in its prompt. `verdict` is
    "accepted", "rejected" (the pipeline asked again, or gave up and marked the
    record) or "failed" (the call errored, or the same question was asked again from
    scratch later, so this answer was never kept).
    """

    model_config = ConfigDict(frozen=True)

    stage: str
    ask: str
    subject: str
    attempt: int
    verdict: str
    reason: str
    started: datetime | None
    ended: datetime
    transcript: str
    prompt: str
    feedback_at: int | None
    reply: str | None
    reasoning: str | None
    usage: dict[str, Any]
    finish: str


def judge(calls: Sequence[Call], conversation: Conversation) -> list[Attempt]:
    """Number the attempts of each question and give every call its verdict."""
    ordered = sorted(calls, key=lambda c: (c.at, c.exchange.ended))
    questions: list[list[Call]] = []
    open_question: dict[tuple[str, str], list[Call]] = {}
    for call in ordered:
        key = (call.ask, call.subject)
        question = open_question.get(key)
        if question and _extends(call.exchange.prompt, question[0].exchange.prompt):
            question.append(call)
        else:
            question = [call]
            questions.append(question)
            open_question[key] = question
    last_asked = {
        (q[0].ask, q[0].subject, q[0].exchange.prompt): n for n, q in enumerate(questions)
    }
    judged: list[Attempt] = []
    for n, question in enumerate(questions):
        base = question[0].exchange.prompt
        rerun = last_asked[(question[0].ask, question[0].subject, base)] != n
        for k, call in enumerate(question):
            verdict, reason = "accepted", ""
            gave_up = _gave_up(call, conversation)
            if k + 1 < len(question):
                verdict, reason = "rejected", question[k + 1].exchange.prompt[len(base) :].strip()
            elif call.error is not None:
                verdict, reason = "failed", call.error
            elif rerun:
                verdict, reason = "failed", "后来同一个问题又从头问了一次，这次的回答没有被保留。"
            elif gave_up is not None:
                verdict, reason = "rejected", gave_up
            judged.append(_attempt(call, k + 1, verdict, reason, len(base) if k else None))
    return sorted(judged, key=lambda a: (a.started or a.ended, a.ended))


def _gave_up(call: Call, conversation: Conversation) -> str | None:
    """The record's review reason when it says this call's ask never got a valid answer."""
    if call.ask not in LETTERS:
        return None
    position = int(call.subject.split(":")[1])
    if position >= len(conversation.assignments):
        return None
    reason = conversation.assignments[position].review_reason or ""
    return reason if reason.startswith(f"call {LETTERS[call.ask]} gave no valid answer") else None


def _attempt(
    call: Call, attempt: int, verdict: str, reason: str, feedback_at: int | None
) -> Attempt:
    exchange = call.exchange
    return Attempt(
        stage=call.stage,
        ask=call.ask,
        subject=call.subject,
        attempt=attempt,
        verdict=verdict,
        reason=reason,
        started=call.started,
        ended=exchange.ended,
        transcript=str(exchange.path),
        prompt=exchange.prompt,
        feedback_at=feedback_at,
        reply=exchange.reply,
        reasoning=exchange.reasoning,
        usage=exchange.usage,
        finish=exchange.finish,
    )


class Job(BaseModel):
    """A node of the page's DAG: one job of the run and the record it wrote.

    `column` is the round it belongs to (0 before the first user message); `after`
    lists the jobs whose output it reads.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    stage: str
    label: str
    column: int
    after: list[str]
    record: Any


def run_jobs(conversation: Conversation, compressions_path: Path) -> list[Job]:
    """Every job the run had to do, whether or not it called the model.

    A message's segments run one after another and read the round summaries of the
    window before it; extract units run in a single chain. The barriers between
    commands (all L1, then all L2, then segmentation, then extraction) are drawn by
    the page from the stages themselves.
    """
    items, digests = conversation.items, conversation.digests
    made: list[Job] = []
    rounds: list[str] = []
    columns: list[int] = []
    for item in items:
        if isinstance(item, UserMessage):
            rounds.append(item.source.uuid)
            made.append(_round_job(item.source.uuid, item.body[:40], len(rounds), digests))
        elif isinstance(item, ToolCall):
            if not rounds and not made:
                made.append(_round_job("", "首条用户消息之前", 0, digests))
            record = {
                "call": _dump(digests.calls.get(item.tool_use_id)),
                "result": _dump(digests.results.get(item.tool_use_id)),
            }
            made.append(
                Job(
                    id=f"tool:{item.tool_use_id}",
                    stage="L1",
                    label=item.name,
                    column=len(rounds),
                    after=[],
                    record=record,
                )
            )
        columns.append(len(rounds))
    for n, uuid in enumerate(rounds):
        window = [f"round:{r}" for r in rounds[max(0, n - 2) : n]]
        previous: list[str] = []
        for part in conversation.segments.get(uuid, []):
            job_id = f"message:{uuid}:{part.start}"
            made.append(
                Job(
                    id=job_id,
                    stage="SEG",
                    label=f"{part.start}–{part.end}",
                    column=n + 1,
                    after=window + previous,
                    record=part.model_dump(mode="json"),
                )
            )
            previous = [job_id]
    compressions = load_compressions(compressions_path) if compressions_path.exists() else []
    for position, block in enumerate(conversation.units):
        index = conversation.unit_items[position]
        assignment = (
            conversation.assignments[position] if position < len(conversation.assignments) else None
        )
        made.append(
            Job(
                id=f"unit:{position}",
                stage="LOOP",
                label=block.split("\n", 2)[-1][:40],
                column=columns[index],
                after=[f"unit:{position - 1}"] if position else [],
                record={
                    "assignment": _dump(assignment),
                    "compressions": [
                        c.model_dump(mode="json") for c in compressions if c.through == index
                    ],
                },
            )
        )
    return made


def _round_job(uuid: str, label: str, column: int, digests: Digests) -> Job:
    return Job(
        id=f"round:{uuid}",
        stage="L2",
        label=label,
        column=column,
        after=[],
        record=_dump(digests.rounds.get(uuid)),
    )


def _dump(model: BaseModel | None) -> Any:
    return None if model is None else model.model_dump(mode="json")


class Page(BaseModel):
    """What the page's script reads; systems and schemas are stored once per ask.

    `since` is the start of the time span transcripts were matched in, None for a
    logged run.
    """

    model_config = ConfigDict(frozen=True)

    directory: str
    merged: str
    backend: str
    model: str
    logged: bool
    since: datetime | None
    systems: dict[str, str]
    schemas: dict[str, Any]
    calls: list[Attempt]
    jobs: list[Job]
    unplaced: list[str]
    attempts: dict[str, int]


def build_page(run: Run, since: datetime | None, until: datetime | None) -> Page:
    """Load the run, place its calls and judge them."""
    conversation = Conversation.load(run)
    calls = logged_calls(run)
    logged = calls is not None
    unplaced: list[Path] = []
    if calls is None:
        _check_alone(run, since, until)
        calls, unplaced, since = matched_calls(run, conversation, since, until)
    return Page(
        directory=str(run.directory),
        merged=str(run.merged),
        backend=run.backend,
        model=run.model,
        logged=logged,
        since=since,
        systems={c.ask: c.exchange.system for c in calls},
        schemas={c.ask: c.exchange.schema for c in calls},
        calls=judge(calls, conversation),
        jobs=run_jobs(conversation, run.extract / "tool_compressions.jsonl"),
        unplaced=[str(p) for p in unplaced],
        attempts={
            "L1": compress.MAX_ATTEMPTS,
            "L2": compress.MAX_ATTEMPTS,
            "SEG": segment.MAX_ATTEMPTS,
            "topic": extract.T_ATTEMPTS,
            "LOOP": extract.CALL_ATTEMPTS,
        },
    )


def _check_alone(run: Run, since: datetime | None, until: datetime | None) -> None:
    if since is not None or until is not None:
        return
    for script in sorted(run.directory.parent.glob("*/run.sh")):
        if script.parent == run.directory:
            continue
        try:
            other = parse_run(script.parent)
        except (ValueError, KeyError):
            continue
        if other.merged == run.merged and other.model == run.model:
            raise ValueError(
                f"{other.directory} ran the same conversation on {run.model}, and its calls "
                "look the same in llm-home; pass --since/--until to bound this run's time."
            )


def render(page: Page) -> str:
    """The template with the page data inlined."""
    data = page.model_dump_json().replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*PAGE_DATA*/null", data)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Replay a pipeline run as a local HTML page.")
    parser.add_argument("run", type=Path, help="run directory with its run.sh")
    parser.add_argument("--out", type=Path, required=True, help="HTML page to write")
    parser.add_argument("--since", type=_time, help="ignore transcripts before this time")
    parser.add_argument("--until", type=_time, help="ignore transcripts after this time")
    args = parser.parse_args()
    page = build_page(parse_run(args.run.resolve()), args.since, args.until)
    args.out.write_text(render(page), encoding="utf-8")
    stages: dict[str, int] = defaultdict(int)
    for call in page.calls:
        stages[call.stage] += 1
    print(f"{len(page.calls)} calls {dict(stages)}, {len(page.unplaced)} unplaced -> {args.out}")


def _time(text: str) -> datetime:
    """ISO time; a naive one is local time."""
    return datetime.fromisoformat(text).astimezone(UTC)


if __name__ == "__main__":
    main()
