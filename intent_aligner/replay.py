"""Replay a pipeline run as one self-contained HTML page, from its records and its calls' transcripts.

Usage: python -m intent_aligner.replay RUN_DIR [--out PAGE.html] [--since T] [--until T] [--watch S]

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
import ast
import importlib
import inspect
import json
import re
import shlex
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast, get_args

import attrs
from pydantic import BaseModel, ConfigDict

from intent_aligner import compress, extract, segment, window
from intent_aligner.corpus import TOOL_ROOT
from intent_aligner.extract import (
    Assignment,
    SpeechSegment,
    Unit,
    into_stores,
    load_assignments,
    load_compressions,
    load_segments,
    render_unit,
    walk_so_far,
)
from intent_aligner.llm import (
    CALLS_FILE,
    LLM_HOME,
    TRUNCATED_FEEDBACK,
    CallRecord,
    qwen_tokens,
)
from intent_aligner.segment import Segment
from intent_aligner.store import IntentNode, Stores
from intent_aligner.units import (
    SPEAKER_NAMES,
    Item,
    Observation,
    Speech,
    ToolCall,
    conversation_units,
)
from intent_aligner.window import Digests, load_digests, round_entry

TEMPLATE = Path(__file__).with_name("replay.html")
# Inlined in this order so the page works offline; cytoscape-dagre needs both libraries before it.
GRAPH_LIBRARIES = [
    Path(__file__).with_name("vendor") / name
    for name in ("cytoscape.min.js", "dagre.min.js", "cytoscape-dagre.js")
]

# The extract command's inline "round" asks share compress's system prompt; which
# command made one is known from calls.jsonl, or else from when it ran (see _allocate).
STEPS: dict[str, tuple[str, str]] = {
    compress.CALL_SYSTEM_PROMPT: ("L1", "call"),
    compress.RESULT_SYSTEM_PROMPT: ("L1", "result"),
    compress.ROUND_SYSTEM_PROMPT: ("LOOP", "round"),
    segment.SYSTEM_PROMPT: ("SEG", "segment"),
    extract.T_SYSTEM_PROMPT: ("LOOP", "topic"),
    extract.A_SYSTEM_PROMPT: ("LOOP", "signal"),
    extract.B_SYSTEM_PROMPT: ("LOOP", "registry"),
    extract.G_SYSTEM_PROMPT: ("LOOP", "aggregate"),
    extract.L_SYSTEM_PROMPT: ("LOOP", "tool_intent"),
    extract.C_SYSTEM_PROMPT: ("LOOP", "relation"),
    extract.C_STEP_SYSTEM_PROMPT: ("LOOP", "step_relation"),
    extract.D_SYSTEM_PROMPT: ("LOOP", "outcome"),
    extract.D_SUMMARY_SYSTEM_PROMPT: ("LOOP", "outcome_summary"),
    extract.D_UPDATE_SYSTEM_PROMPT: ("LOOP", "outcome_update"),
}
# The letter each extract ask goes by in review_reason ("call T gave no valid answer ...").
LETTERS = {
    "topic": "T",
    "signal": "A",
    "registry": "B",
    "aggregate": "G",
    "tool_intent": "L",
    "relation": "C",
    "step_relation": "C",
    "outcome": "D",
    "outcome_summary": "D",
    "outcome_update": "D",
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
    window_rounds: int


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
        window_rounds=int(options.get("--window-rounds", window.DEFAULT_WINDOW_ROUNDS)),
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

    `request` is the transcript's request line without the texts shown on their own
    (system, prompt, schema, and a hive body's messages). `ended` is the file's mtime:
    the backend writes the transcript's last line when the reply arrives, and nothing
    touches it afterwards.
    """

    path: Path
    request: dict[str, Any]
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
    """A Codex/Hive request log or a native Claude one-shot transcript with a prompt snapshot."""
    # cast: transcripts are written by intent_aligner.llm, one JSON object per line.
    lines = [cast(dict[str, Any], json.loads(line)) for line in path.read_text().splitlines()]
    request = lines[0] if lines else {}
    if request.get("type") != "request":
        return _claude_exchange(path, lines)
    reply: str | None = None
    reasoning: str | None = None
    summaries: list[str] = []
    usage: dict[str, Any] = {}
    finish = "no reply"
    for line in lines[1:]:
        kind = line.get("type")
        if kind == "item.completed" and line["item"]["type"] == "agent_message":
            reply, finish = str(line["item"]["text"]), "stop"
        elif kind == "item.completed" and line["item"]["type"] == "reasoning":
            summaries.append(str(line["item"]["text"]))
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
        request=_request_settings(request),
        model=request["model"],
        # A hive body carries the system prompt the model read, schema included (llm.hive_request_body).
        system=request["body"]["messages"][0]["content"]
        if "body" in request
        else request["system_prompt"],
        prompt=request["prompt"],
        schema=request.get("json_schema"),
        reply=reply,
        reasoning="\n\n".join(summaries) if summaries else reasoning,
        usage=usage,
        finish=finish,
        ended=datetime.fromtimestamp(path.stat().st_mtime, UTC),
    )


def _claude_exchange(path: Path, lines: list[dict[str, Any]]) -> Exchange | None:
    """Claude splits one API message into block records with the same usage and message ID.

    The prompt snapshot records the supplied system prompt and StructuredOutput schema;
    tool-result user messages are not new prompts. Empty or redacted thinking has no
    readable summary to recover. Signatures are never used as thinking text.
    """
    prompt: str | None = None
    system: str | None = None
    schema: dict[str, Any] | None = None
    settings: dict[str, Any] = {"backend": "claude"}
    texts: list[str] = []
    summaries: list[str] = []
    structured: dict[str, Any] | None = None
    messages: dict[str, dict[str, Any]] = {}
    finish = "no reply"
    for line in lines:
        if line.get("isSidechain"):
            continue
        snapshot = line.get("attachment", {})
        if snapshot.get("type") == "prompt_snapshot":
            system = "\n\n".join(snapshot["systemPrompt"])
            for tool in snapshot.get("tools", []):
                if tool.get("name") == "StructuredOutput":
                    schema = tool["schema"]["input_schema"]
        kind = line.get("type")
        message = line.get("message", {})
        content = message.get("content", [])
        if kind == "user" and prompt is None:
            if isinstance(content, str):
                prompt = content
            else:
                parts = [block["text"] for block in content if block["type"] == "text"]
                if parts:
                    prompt = "\n\n".join(parts)
        if kind != "assistant":
            continue
        settings["model"] = line.get("requestedModel", message["model"])
        if "effort" in line:
            settings["effort"] = line["effort"]
        if "usage" in message:
            messages[message["id"]] = message["usage"]
        if message.get("stop_reason"):
            finish = message["stop_reason"]
        for block in content:
            if block["type"] == "thinking" and block.get("thinking"):
                summaries.append(block["thinking"])
            elif block["type"] == "text":
                texts.append(block["text"])
            elif block["type"] == "tool_use" and block["name"] == "StructuredOutput":
                structured = block["input"]
    if prompt is None or system is None or "model" not in settings:
        return None
    usage: dict[str, Any] = {}
    for counts in messages.values():
        for key, value in counts.items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
    # Claude transcripts write "output_tokens_details": null when the API sent no split.
    if messages and all(
        "thinking_tokens" in (m.get("output_tokens_details") or {}) for m in messages.values()
    ):
        usage["output_tokens_details"] = {
            "thinking_tokens": sum(
                m["output_tokens_details"]["thinking_tokens"] for m in messages.values()
            )
        }
    if finish == "end_turn" or (finish == "tool_use" and structured is not None):
        finish = "stop"
    elif finish == "max_tokens":
        finish = "length"
    reply = "\n\n".join(texts) if texts else None
    if structured is not None:
        reply = json.dumps(structured, ensure_ascii=False)
    return Exchange(
        path=path,
        request=settings,
        model=settings["model"],
        system=system,
        prompt=prompt,
        schema=schema,
        reply=reply,
        reasoning="\n\n".join(summaries) if summaries else None,
        usage=usage,
        finish=finish,
        ended=datetime.fromtimestamp(path.stat().st_mtime, UTC),
    )


def _request_settings(request: dict[str, Any]) -> dict[str, Any]:
    shown = {"system_prompt", "prompt", "json_schema", "type"}
    settings = {key: value for key, value in request.items() if key not in shown}
    if isinstance(settings.get("body"), dict):
        body = {key: value for key, value in settings["body"].items() if key != "messages"}
        body["messages"] = "[system, user]：内容见 system 和提示词"
        settings["body"] = body
    return settings


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Conversation:
    """The run's conversation and what the run wrote about it.

    `units` are the extract units' <当前单元> blocks ("" while their digests are
    missing) and `unit_items` their timeline indexes. `round_of` maps a tool call to
    the speech that opened its round, "" before the first one.
    """

    items: Sequence[Item]
    digests: Digests
    segments: dict[str, list[Segment]]
    assignments: list[Assignment]
    units: list[str]
    unit_items: list[int]
    round_of: dict[str, str]
    window_rounds: int

    @classmethod
    def load(cls, run: Run) -> Conversation:
        """Everything the run wrote so far."""
        items = list(conversation_units(run.merged))
        digests = load_digests(run.digests) if run.digests.exists() else Digests()
        segments = load_segments(run.segments) if run.segments.exists() else {}
        done = run.extract / "assignments.jsonl"
        steps = walk_so_far(items, segments, digests)
        round_of: dict[str, str] = {}
        opening = ""
        for item in items:
            if isinstance(item, Speech):
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
            window_rounds=run.window_rounds,
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
    `origin` is the run directory that made the call when it is another run's: results
    copied from there come with their calls.jsonl lines.
    """

    stage: str
    ask: str
    subject: str
    started: datetime | None
    exchange: Exchange
    error: str | None
    origin: str | None = None

    @property
    def at(self) -> datetime:
        """When the call began, or when it ended if that is all the transcript says."""
        return self.started or self.exchange.ended


def _stage(ask: str) -> str:
    """Stage from the log record: a run's system prompts may be older than this code's."""
    if ask in ("call", "result"):
        return "L1"
    if ask == "segment":
        return "SEG"
    return "LOOP"


def logged_calls(run: Run) -> list[Call] | None:
    """The calls named by the run's calls.jsonl files, or None for a run older than them.

    A run that reads another run's segments.jsonl gets that run's segment calls with it,
    from the extract/calls.jsonl next to the file; digests come with their own directory's
    calls.jsonl the same way.
    """
    logs = [run.digests.parent / CALLS_FILE, run.extract / CALLS_FILE]
    segments_log = run.segments.parent / "extract" / CALLS_FILE
    reused_segments = segments_log not in logs and segments_log.exists()
    if not any(path.exists() for path in logs):
        return None if _has_output(run) else []
    origins = _origins(run)
    calls = []
    for path in [*logs, segments_log] if reused_segments else logs:
        if not path.exists():
            continue
        owner = path.parent.parent if path.parent.name == "extract" else path.parent
        foreign = None if owner == run.directory else str(owner)
        for line in path.read_text(encoding="utf-8").splitlines():
            record = CallRecord.model_validate_json(line)
            if path == segments_log and reused_segments and record.ask != "segment":
                continue
            exchange = None if record.transcript is None else read_exchange(Path(record.transcript))
            if exchange is None:
                continue
            subject = record.subject
            if record.ask == "segment":
                subject = f"{subject}:{len(_split(exchange.prompt)[0])}"
            calls.append(
                Call(
                    stage=_stage(record.ask),
                    ask=record.ask,
                    subject=subject,
                    started=record.started,
                    exchange=exchange,
                    error=record.error,
                    origin=foreign or origins.get(record.transcript or ""),
                )
            )
    return calls


def _has_output(run: Run) -> bool:
    """A run that wrote nothing yet has made no call, whatever its age."""
    files = [run.digests, run.segments, run.extract / "assignments.jsonl"]
    return any(f.exists() and f.stat().st_size > 0 for f in files)


def _origins(run: Run) -> dict[str, str]:
    """Transcript -> the other run directory whose calls.jsonl logged it first."""
    logs = [
        path
        for directory in run.directory.parent.iterdir()
        if directory.is_dir() and directory != run.directory
        for path in (directory / CALLS_FILE, directory / "extract" / CALLS_FILE)
        if path.exists()
    ]
    origins: dict[str, str] = {}
    for path in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True):
        directory = str(path.parent.parent if path.parent.name == "extract" else path.parent)
        for line in path.read_text(encoding="utf-8").splitlines():
            transcript = CallRecord.model_validate_json(line).transcript
            if transcript is not None:
                origins[transcript] = directory
    return origins


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
    of the same conversation match too, since a speech segment's prompt can be the same
    in every run; without --since the run starts after the last pause of RUN_GAP
    between matched calls before the run's last output. Unplaced transcripts are this
    model's calls inside the run's span that matched no job: the code that sent them
    has changed since, or they belong to a run this replay cannot see.
    """
    found: list[tuple[Exchange, str, str, list[str]]] = []
    others: list[Exchange] = []
    finder = _JobFinder(conversation=conversation)
    pattern = "projects/*/*.jsonl" if run.backend == "claude" else "*.jsonl"
    for path in sorted((LLM_HOME / run.backend).glob(pattern)):
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
            elif isinstance(item, Speech):
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
        items = self.conversation.items
        return [f"message:{items[i].source.uuid}:{len(done)}" for i in indexes]

    def unit(self, candidates: list[str], prompt: str) -> str:
        """The unit an extract call served among identical-looking candidates.

        Units whose blocks are identical (the same tool call twice) differ in the
        header's timestamp, which every extract prompt carries.
        """
        positions = [int(c.split(":")[1]) for c in candidates]
        ahead = [p for p in positions if p >= self.unit_pointer] or positions
        items, unit_items = self.conversation.items, self.conversation.unit_items
        header = _header_of(prompt)
        stamped = [
            p
            for p in ahead
            if header["unit"]["timestamp"] == items[unit_items[p]].source.timestamp.isoformat()
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


class TokenUse(BaseModel):
    """Token usage; Claude answer includes all non-thinking output, including tool JSON."""

    model_config = ConfigDict(frozen=True)

    prompt: int
    thinking: int
    answer: int


class Attempt(BaseModel):
    """One call as the page shows it, judged among the calls of its question.

    A question is a first prompt and the retries that extend it with feedback;
    `feedback_at` is where a retry's feedback starts in its prompt. `verdict` is
    "accepted", "rejected" (the pipeline asked again, or gave up and marked the
    record) or "failed" (the call errored, or the same question was asked again from
    scratch later, so this answer was never kept). `fixed` are the [start, end)
    ranges of the prompt that are string literals of the source (see literal_texts).
    `window_seen` are the opening uuids of the earlier rounds the prompt's window really
    holds, oldest first, or None for a prompt without a window; `window_limit` is how
    many it could have held, the run's window size or the rounds that exist if fewer.
    `origin` is the other run directory that made the call (see Call). `tokens` is
    None when the reply carries no usage; `thinking_budget` is the cap its request set,
    None for a backend without one.
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
    request: dict[str, Any]
    prompt: str
    fixed: list[tuple[int, int]]
    feedback_at: int | None
    window_seen: list[str] | None
    window_limit: int
    reply: str | None
    reasoning: str | None
    usage: dict[str, Any]
    finish: str
    origin: str | None
    tokens: TokenUse | None
    thinking_budget: int | None


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
    literals = literal_texts()
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
            exchange = call.exchange
            window_seen, window_limit = _window_seen(call, conversation)
            judged.append(
                Attempt(
                    stage=call.stage,
                    ask=call.ask,
                    subject=call.subject,
                    attempt=k + 1,
                    verdict=verdict,
                    reason=reason,
                    started=call.started,
                    ended=exchange.ended,
                    transcript=str(exchange.path),
                    request=exchange.request,
                    prompt=readable(exchange.prompt),
                    fixed=literal_ranges(readable(exchange.prompt), literals),
                    feedback_at=len(readable(base)) if k else None,
                    window_seen=window_seen,
                    window_limit=window_limit,
                    reply=exchange.reply,
                    reasoning=exchange.reasoning,
                    usage=exchange.usage,
                    finish=exchange.finish,
                    origin=call.origin,
                    tokens=_tokens(exchange),
                    thinking_budget=_thinking_budget(exchange.request),
                )
            )
    return sorted(judged, key=lambda a: (a.started or a.ended, a.ended))


def _tokens(exchange: Exchange) -> TokenUse | None:
    """Codex and Claude report reasoning counts; Hive reports only completion_tokens.

    Claude input includes cache reads and writes. Its answer count includes all
    non-thinking output, not just the final StructuredOutput payload.

    For Hive the answer is counted under the Qwen tokenizer and the rest of the
    completion is thinking, so only the short answer carries the local tokenizer's
    error (within ~7% of the server's).
    """
    usage = exchange.usage
    if exchange.request.get("backend") == "claude":
        details = usage.get("output_tokens_details", {})
        if "thinking_tokens" not in details:
            return None
        thinking = int(details["thinking_tokens"])
        return TokenUse(
            prompt=sum(
                int(usage.get(key, 0))
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            ),
            thinking=thinking,
            answer=int(usage["output_tokens"]) - thinking,
        )
    if "reasoning_output_tokens" in usage:
        output, thinking = int(usage["output_tokens"]), int(usage["reasoning_output_tokens"])
        return TokenUse(
            prompt=int(usage["input_tokens"]), thinking=thinking, answer=output - thinking
        )
    if "completion_tokens" in usage:
        answer = qwen_tokens(exchange.reply or "")
        return TokenUse(
            prompt=int(usage["prompt_tokens"]),
            thinking=int(usage["completion_tokens"]) - answer,
            answer=answer,
        )
    return None


def _thinking_budget(request: dict[str, Any]) -> int | None:
    """The budget the request body carried, so a run keeps the budget it was sent with.

    `reasoning_budget_tokens` is the field hive-frame's config names for every Qwen client.
    """
    body = request.get("body")
    return body.get("reasoning_budget_tokens") if isinstance(body, dict) else None


HEADER = re.compile(r"<header>\n(.*?)\n</header>", re.S)


def _header_of(prompt: str) -> Any:
    match = HEADER.search(prompt)
    if match is None:
        raise ValueError(f"an extract prompt without a <header> block: {prompt[:200]!r}")
    return json.loads(match.group(1))


def readable(prompt: str) -> str:
    """The prompt with its header indented for people; the model got the compact layout."""
    return HEADER.sub(
        lambda m: (
            f"<header>\n{json.dumps(json.loads(m.group(1)), ensure_ascii=False, indent=2)}\n</header>"
        ),
        prompt,
        count=1,
    )


def _window_seen(call: Call, conversation: Conversation) -> tuple[list[str] | None, int]:
    """Which earlier rounds the call's window holds, and how many it could have held.

    The window drops whole rounds when it does not fit, so the earlier speeches still
    quoted in it, counted back from the current round, are the rounds it kept.
    """
    opened = call.exchange.prompt.split("<此前的对话>\n", 1)
    kind, _, rest = call.subject.partition(":")
    if len(opened) < 2 or kind not in ("unit", "message"):
        return None, 0
    items = conversation.items
    speeches = [i for i, item in enumerate(items) if isinstance(item, Speech)]
    if kind == "unit":
        index = conversation.unit_items[int(rest)]
    else:
        uuid = rest.rsplit(":", 1)[0]
        index = next(i for i in speeches if items[i].source.uuid == uuid)
    opening = max((i for i in speeches if i <= index), default=-1)
    earlier = [i for i in speeches if i < opening][-conversation.window_rounds :]
    block = opened[1].split("\n</此前的对话>", 1)[0]
    seen: list[str] = []
    for i in reversed(earlier):
        if round_entry(items[i], conversation.digests) not in block:
            break
        seen.insert(0, items[i].source.uuid)
    return seen, len(earlier)


def _gave_up(call: Call, conversation: Conversation) -> str | None:
    """The record's review reason when it says this call's ask never got a valid answer."""
    if call.ask not in LETTERS:
        return None
    position = int(call.subject.split(":")[1])
    if position >= len(conversation.assignments):
        return None
    reason = conversation.assignments[position].review_reason or ""
    return reason if reason.startswith(f"call {LETTERS[call.ask]} gave no valid answer") else None


# The modules whose string literals build prompts; units.py is left out because the
# tool call texts it renders are the conversation's content.
PROMPT_MODULES = (compress, segment, extract, window)
# A literal counts as prompt text only if it carries a tag, a label or a full-width
# punctuation mark; shorter or plainer literals ("\n", "topics") would also match
# inside the conversation's own text.
_PROMPT_MARKS = frozenset("<>【】：（）")


def literal_texts() -> list[str]:
    """Prompt-building string literals of the source, longest first, docstrings left out."""
    found: set[str] = set()
    for module in PROMPT_MODULES:
        tree = ast.parse(inspect.getsource(module))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and len(node.value.strip()) >= 2
                and _PROMPT_MARKS & set(node.value)
            ):
                found.add(node.value)
    return sorted(found, key=len, reverse=True)


def literal_ranges(prompt: str, literals: Sequence[str]) -> list[tuple[int, int]]:
    """Where `literals` occur in `prompt`, longest first and without overlaps, merged.

    Examples:
        >>> literal_ranges("<a>\\nx\\n</a>", ["\\n</a>", "<a>\\n"])
        [(0, 4), (5, 10)]
    """
    taken = bytearray(len(prompt))
    for literal in literals:
        at = prompt.find(literal)
        while at >= 0:
            end = at + len(literal)
            if not any(taken[at:end]):
                taken[at:end] = b"\x01" * len(literal)
            at = prompt.find(literal, at + 1)
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate([*taken, 0]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ranges.append((start, i))
            start = None
    return ranges


class Rule(BaseModel):
    """A check the code applies to an ask's reply, or feedback it sends with a retry.

    `text` is the message as written in the source, its expressions shown as {source};
    `where` is the definition it is in.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["check", "feedback"]
    text: str
    where: str


# The definitions that judge each ask's reply. Their messages in Chinese are the ones
# the model is sent back; the reply models they use are searched too (see ask_rules).
RULE_SOURCES: dict[str, tuple[str, ...]] = {
    "call": ("compress._compress_call", "compress._ask_until_valid"),
    "result": ("compress._compress_result", "compress._ask_until_valid"),
    "round": ("compress.summarize_trajectory", "compress._ask_until_valid"),
    "segment": ("segment.next_segment", "segment.locate_end"),
    "topic": ("extract._assign.parse", "extract._ask_until_valid"),
    "signal": ("extract._with_intents.parse_a", "extract._ask_until_valid"),
    "registry": ("extract._match_intents.parse_b", "extract._ask_until_valid"),
    "aggregate": ("extract._file_under_aggregates.parse", "extract._ask_until_valid"),
    "tool_intent": ("extract._match_tool_intents.parse", "extract._ask_until_valid"),
    "relation": ("extract._relate.parse", "extract._ask_until_valid"),
    "step_relation": ("extract._relate.parse", "extract._ask_until_valid"),
    "outcome": ("extract._parse_outcomes", "extract._ask_until_valid"),
    "outcome_summary": ("extract._parse_outcomes", "extract._ask_until_valid"),
    "outcome_update": ("extract._parse_outcomes", "extract._ask_until_valid"),
}


def ask_rules(ask: str) -> list[Rule]:
    """The checks and retry feedback of `ask`, read from the source of RULE_SOURCES[ask].

    Each source also brings the checks of the reply models it names and of the models
    nested in their fields.
    """
    rules: list[Rule] = []
    models: list[type[BaseModel]] = []
    for where in RULE_SOURCES[ask]:
        module_name, *path = where.split(".")
        module = importlib.import_module(f"intent_aligner.{module_name}")
        source = inspect.getsource(module)
        node = _definition(ast.parse(source), path, where)
        rules += _messages(source, node, where)
        for name in {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}:
            value = getattr(module, name, None)
            if isinstance(value, type) and issubclass(value, BaseModel):
                _collect_models(value, models)
    for model in models:
        source = inspect.getsource(model)
        where = f"{model.__module__.rsplit('.', 1)[-1]}.{model.__name__}"
        rules += _messages(source, ast.parse(source), where)
    unique: dict[str, Rule] = {}
    for rule in rules:
        unique.setdefault(rule.text, rule)
    return list(unique.values())


def _definition(tree: ast.AST, path: Sequence[str], where: str) -> ast.AST:
    node = tree
    for name in path:
        found = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.FunctionDef | ast.ClassDef) and child.name == name
        ]
        if not found:
            raise ValueError(f"{where} is not in the source; update replay.RULE_SOURCES.")
        node = found[0]
    return node


def _collect_models(model: type[BaseModel], into: list[type[BaseModel]]) -> None:
    if model in into or not model.__module__.startswith("intent_aligner."):
        return
    into.append(model)
    for field in model.model_fields.values():
        for arg in _type_args(field.annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                _collect_models(arg, into)


def _type_args(annotation: object) -> list[object]:
    args = list(get_args(annotation))
    return [annotation, *[a for arg in args for a in _type_args(arg)]]


def _messages(source: str, node: ast.AST, where: str) -> list[Rule]:
    """Chinese raise messages (checks) and assignments to `feedback` (retry feedback)."""
    rules: list[Rule] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call) and child.exc.args:
            text = _shown(source, child.exc.args[0])
            if text is not None:
                rules.append(Rule(kind="check", text=text, where=where))
        elif isinstance(child, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "feedback" for t in child.targets
        ):
            value = child.value
            if isinstance(value, ast.Name) and value.id == "TRUNCATED_FEEDBACK":
                rules.append(Rule(kind="feedback", text=TRUNCATED_FEEDBACK, where=where))
                continue
            if isinstance(value, ast.JoinedStr) and _is_truncation(value):
                rules.append(Rule(kind="feedback", text=TRUNCATED_FEEDBACK, where=where))
                continue
            text = _shown(source, value)
            if text is not None:
                rules.append(Rule(kind="feedback", text=text.strip(), where=where))
    return rules


def _is_truncation(value: ast.JoinedStr) -> bool:
    return any(
        isinstance(part, ast.FormattedValue)
        and isinstance(part.value, ast.Name)
        and part.value.id == "TRUNCATED_FEEDBACK"
        for part in value.values
    )


def _shown(source: str, expression: ast.expr) -> str | None:
    """A string or f-string with its expressions as {source}; None unless it has Chinese."""
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        text = expression.value
    elif isinstance(expression, ast.JoinedStr):
        text = "".join(
            str(part.value)
            if isinstance(part, ast.Constant)
            # The parts of an f-string are only its literal pieces and FormattedValues.
            else "{"
            + (ast.get_source_segment(source, cast(ast.FormattedValue, part).value) or "")
            + "}"
            for part in expression.values
        )
    else:
        return None
    return text if re.search(r"[\u4e00-\u9fff]", text) else None


def window_rules(window_rounds: int) -> list[str]:
    """The constants and docstrings that decide what a prompt's windows hold."""
    constants = [
        f"window.PROMPT_BUDGET = {window.PROMPT_BUDGET:,} token（单槽 {window.SLOT_TOKENS:,} 扣掉思考预算"
        "后，提示词和回答共用剩下的部分；header 里的图谱全部计入）",
        f"--window-rounds = {window_rounds}（这次运行的值。放不下时先压缩最旧的工具记录，"
        "全部压缩过仍然放不下才逐轮减少，1 轮也放不下时这个单元标记 needs_review）",
        f"window.COMPRESSION_BATCH = {window.COMPRESSION_BATCH:,}（一次压缩至少累积这么多 token 的工具记录）",
        f"compress.MESSAGE_ROOM = {compress.MESSAGE_ROOM:,}（压缩提示词里的发言超过这么多 token 时只给开头）",
        f"window.FOLLOWING_ITEMS = {window.FOLLOWING_ITEMS}，window.FOLLOWING_BUDGET = "
        f"{window.FOLLOWING_BUDGET:,}（调用 G 的 <此后的对话>）",
        f"FEEDBACK_ROOM：compress {compress.FEEDBACK_ROOM:,}，segment {segment.FEEDBACK_ROOM:,}，"
        f"extract {extract.FEEDBACK_ROOM:,}",
    ]
    documented = (
        extract._prompt,
        extract._fitted_prompt,
        extract._within_budget,
        extract._compress_oldest,
        window.render_items,
        window.render_following,
    )
    return constants + [
        f"{f.__module__.rsplit('.', 1)[-1]}.{f.__name__}：{inspect.getdoc(f)}" for f in documented
    ]


class Job(BaseModel):
    """A node of the page's DAG: one job of the run and the record it wrote.

    `column` is the round it belongs to (0 before the first speech); `after`
    lists the jobs whose output it reads. `speaker` names who made the speech a
    SEG job works from. `route` is set on an extract unit marked needs_review: the letters
    of the calls its path goes through, so the page can show the ones that were never sent;
    a letter ending in ? runs only if a model answer the unit did not get asks for it.
    `text` is what the job works on when its
    prompt does not show it alone: an extract unit's block or a message's body. `log`
    are the run.log lines about an extract unit.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    stage: str
    label: str
    column: int
    after: list[str]
    speaker: str = ""
    route: list[str] = []
    text: str = ""
    log: list[str] = []
    record: Any


def run_jobs(conversation: Conversation, extract_dir: Path, stores: Stores) -> list[Job]:
    """Every job the run had to do, whether or not it called the model.

    `stores` starts empty and has each unit folded in, because whether a unit needs a
    call depends on the graph as it stood when the unit ran.

    A speech's segments run one after another. Extract units run in a single chain. The
    barrier between the commands (all L1, then the main loop) is drawn by the page from
    the stages themselves.
    """
    items, digests = conversation.items, conversation.digests
    made: list[Job] = []
    rounds: list[str] = []
    columns: list[int] = []
    bodies: dict[str, str] = {}
    speakers: dict[str, str] = {}
    for item in items:
        if isinstance(item, Speech):
            rounds.append(item.source.uuid)
            bodies[item.source.uuid] = item.body
            speakers[item.source.uuid] = SPEAKER_NAMES[item.speaker]
        elif isinstance(item, ToolCall):
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
        previous: list[str] = []
        for part in conversation.segments.get(uuid, []):
            job_id = f"message:{uuid}:{part.start}"
            made.append(
                Job(
                    id=job_id,
                    stage="SEG",
                    label=f"{part.start}–{part.end}",
                    column=n + 1,
                    after=previous,
                    speaker=speakers[uuid],
                    text=bodies[uuid],
                    record=part.model_dump(mode="json"),
                )
            )
            previous = [job_id]
    compressions_path = extract_dir / "tool_compressions.jsonl"
    compressions = load_compressions(compressions_path) if compressions_path.exists() else []
    log_path = extract_dir / "run.log"
    log = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    steps = walk_so_far(items, conversation.segments, conversation.digests)
    for position, (block, step) in enumerate(zip(conversation.units, steps, strict=True)):
        index = conversation.unit_items[position]
        assignment = (
            conversation.assignments[position] if position < len(conversation.assignments) else None
        )
        route = (
            _route(step.unit, assignment, stores)
            if assignment is not None and assignment.needs_review
            else []
        )
        if assignment is not None:
            into_stores(stores, step, assignment)
        made.append(
            Job(
                id=f"unit:{position}",
                stage="LOOP",
                label=block.split("\n", 2)[-1][:40],
                column=columns[index],
                after=[f"unit:{position - 1}"] if position else [],
                route=route,
                text=block,
                log=[] if assignment is None else _log_lines(log, assignment),
                record={
                    "assignment": _dump(assignment),
                    "compressions": [
                        c.model_dump(mode="json") for c in compressions if c.through == index
                    ],
                },
            )
        )
    return made


def _route(unit: Unit, record: Assignment, stores: Stores) -> list[str]:
    """The call letters on `unit`'s path, given the graph before it ran.

    Mirrors the branches of the extract loop. Without a topic the graph that decides
    calls B and L is unknown, so they end in ?; so do B and C after call A, whose answer
    decides whether the unit has intents at all.
    """
    store = stores.stores.get(record.topic) if record.topic is not None else None
    if isinstance(unit, Observation):
        return ["D"] if unit.tool_use_id is not None else ["T", "D"]
    if isinstance(unit, ToolCall):
        may_match = record.topic is None or (store is not None and store.meta.tool_intents)
        return ["T", "G", *(["L?"] if may_match else []), "C"]
    if isinstance(unit, SpeechSegment) and not unit.segment.intents:
        return ["T"]
    actor = unit.speech.speaker if isinstance(unit, SpeechSegment) else "user"
    known = record.topic is not None
    has_candidates = store is not None and any(
        isinstance(node, IntentNode) and node.actor == actor
        for node in store.meta.graph.nodes.values()
    )
    if isinstance(unit, SpeechSegment):
        registry = ["B"] if has_candidates else [] if known else ["B?"]
        return ["T", *registry, "C"]
    registry = ["B?"] if has_candidates or not known else []
    return ["T", "A", *registry, "C?"]


def _log_lines(log: Sequence[str], assignment: Assignment) -> list[str]:
    """run.log names a unit by its tool_use_id, or its uuid (and chars for a segment)."""
    ref = assignment.unit
    marks = [ref.tool_use_id] if ref.tool_use_id else [ref.uuid]
    if ref.chars is not None:
        marks.append(f"chars [{ref.chars[0]}, {ref.chars[1]}]")
    return [line for line in log if all(mark in line for mark in marks)]


def _dump(model: BaseModel | None) -> Any:
    return None if model is None else model.model_dump(mode="json")


class GraphChange(BaseModel):
    """What one extract unit changed in its topic's graph when its record was folded in.

    `nodes` are the nodes it added or changed, `edges` the edges it added; `removed`
    and `removed_edges` what it took away. A unit without a topic changes nothing.
    """

    model_config = ConfigDict(frozen=True)

    topic: str | None
    nodes: dict[str, Any]
    edges: list[Any]
    removed: list[str]
    removed_edges: list[Any]


def graph_changes(run: Run, conversation: Conversation) -> list[GraphChange]:
    """Fold assignments.jsonl into the stores as extract does, one unit at a time."""
    stores = Stores(merged_file=str(run.merged), items=conversation.items)
    before: dict[str, tuple[dict[str, Any], list[Any]]] = {}
    changes: list[GraphChange] = []
    steps = walk_so_far(conversation.items, conversation.segments, conversation.digests)
    for step, record in zip(steps, conversation.assignments, strict=False):
        into_stores(stores, step, record)
        topic = record.topic
        if topic is None:
            changes.append(
                GraphChange(topic=None, nodes={}, edges=[], removed=[], removed_edges=[])
            )
            continue
        graph = stores.stores[topic].meta.graph.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        nodes, edges = graph["nodes"], graph["edges"]
        old_nodes, old_edges = before.get(topic, ({}, []))
        old_keys = {json.dumps(e, sort_keys=True) for e in old_edges}
        keys = {json.dumps(e, sort_keys=True) for e in edges}
        changes.append(
            GraphChange(
                topic=topic,
                nodes={k: v for k, v in nodes.items() if old_nodes.get(k) != v},
                edges=[e for e in edges if json.dumps(e, sort_keys=True) not in old_keys],
                removed=[k for k in old_nodes if k not in nodes],
                removed_edges=[e for e in old_edges if json.dumps(e, sort_keys=True) not in keys],
            )
        )
        before[topic] = (nodes, edges)
    return changes


class Page(BaseModel):
    """What the page's script reads; systems, schemas and rules are stored once per ask.

    `since` is the start of the time span transcripts were matched in, None for a
    logged run. `graphs` has one change per extract unit with a record; `files` names
    the file each stage writes its records to.
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
    rules: dict[str, list[Rule]]
    window: list[str]
    window_rounds: int
    calls: list[Attempt]
    jobs: list[Job]
    graphs: list[GraphChange]
    topics: dict[str, str]
    files: dict[str, str]
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
    asks = {c.ask for c in calls}
    return Page(
        directory=str(run.directory),
        merged=str(run.merged),
        backend=run.backend,
        model=run.model,
        logged=logged,
        since=since,
        systems={c.ask: c.exchange.system for c in calls},
        schemas={c.ask: c.exchange.schema for c in calls},
        rules={ask: ask_rules(ask) for ask in asks},
        window=window_rules(run.window_rounds),
        window_rounds=conversation.window_rounds,
        calls=judge(calls, conversation),
        jobs=run_jobs(
            conversation,
            run.extract,
            Stores(merged_file=str(run.merged), items=conversation.items),
        ),
        graphs=graph_changes(run, conversation),
        topics={a.topic: a.new_title for a in conversation.assignments if a.topic and a.new_title},
        files={
            "L1": str(run.digests),
            "SEG": str(run.segments),
            "LOOP": str(run.extract / "assignments.jsonl"),
            "inline": str(run.extract / "tool_compressions.jsonl"),
        },
        unplaced=[str(p) for p in unplaced],
        attempts={
            "L1": compress.MAX_ATTEMPTS,
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


def render(page: Page, refresh_seconds: int | None) -> str:
    """The template with the page data inlined; a page with `refresh_seconds` reloads itself."""
    data = page.model_dump_json().replace("</", "<\\/")
    refresh = "null" if refresh_seconds is None else str(refresh_seconds)
    libraries = "\n".join(path.read_text(encoding="utf-8") for path in GRAPH_LIBRARIES)
    template = TEMPLATE.read_text(encoding="utf-8").replace("/*GRAPH_LIBRARIES*/", libraries, 1)
    return template.replace("/*PAGE_DATA*/null", data, 1).replace(
        "/*REFRESH_SECONDS*/null", refresh, 1
    )


def _publish(args: argparse.Namespace) -> None:
    """Replace the page in one step, so a reload never reads a half-written file."""
    page = build_page(parse_run(args.run.resolve()), args.since, args.until)
    partial = args.out.with_name(f"{args.out.name}.partial")
    partial.write_text(render(page, args.watch), encoding="utf-8")
    partial.replace(args.out)
    stages: dict[str, int] = defaultdict(int)
    for call in page.calls:
        stages[call.stage] += 1
    print(
        f"{len(page.calls)} calls {dict(stages)}, {len(page.unplaced)} unplaced -> {args.out}",
        flush=True,
    )


def _watch(args: argparse.Namespace) -> None:
    """Rebuild forever; a rebuild that fails leaves the last good page for the next try.

    The run is still appending to its files, so the last line of one can be half written.
    """
    while True:
        try:
            _publish(args)
        except (OSError, ValueError) as error:
            print(f"Rebuild failed, keeping the last page: {error}", flush=True)
        time.sleep(args.watch)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Replay a pipeline run as a local HTML page.")
    parser.add_argument("run", type=Path, help="run directory with its run.sh")
    parser.add_argument("--out", type=Path, help="HTML page to write; default RUN/replay.html")
    parser.add_argument("--since", type=_time, help="ignore transcripts before this time")
    parser.add_argument("--until", type=_time, help="ignore transcripts after this time")
    parser.add_argument(
        "--watch", type=int, metavar="SECONDS", help="rebuild and reload the page every SECONDS"
    )
    args = parser.parse_args()
    args.out = args.out or args.run / "replay.html"
    if args.watch is None:
        _publish(args)
    else:
        _watch(args)


def _time(text: str) -> datetime:
    """ISO time; a naive one is local time."""
    return datetime.fromisoformat(text).astimezone(UTC)


if __name__ == "__main__":
    main()
