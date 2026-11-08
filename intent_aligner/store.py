"""Topic store files and the builder that fills them as the pointer walks.

The YAML shape is the one docs/trajectory-extractor.md defines; the models below are
that shape, and the hand-made playground/topics/*.yaml load through them. The builder
is a pure fold over the units and their records, so a resumed run rebuilds the same
stores by replaying assignments.jsonl.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import attrs
import yaml
from pydantic import BaseModel, ConfigDict, Field

from intent_aligner.units import AIText, Item, ToolCall, UserMessage, UserSignal

STORE_HEADER = "# 结构和填写规则见 docs/trajectory-extractor.md（路径相对于工具根目录）\n"
UNASSIGNED_FILE = "unassigned.yaml"

Relation = Literal[
    "prerequisite", "serves", "refines", "supersedes", "produces", "satisfies", "blocks", "triggers"
]


class IntentNode(BaseModel):
    """A user intent in a topic's registry; `label` is its one standard text."""

    model_config = ConfigDict(frozen=True)

    type: Literal["intent"] = "intent"
    actor: Literal["user"] = "user"
    label: str = Field(min_length=1)


class AggregateNode(BaseModel):
    """What a run of tool calls is doing, at the level of what the user says.

    Tool-call intents are too low-level for the graph; `Meta.tool_intents` lists them
    under their aggregate.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["aggregate"] = "aggregate"
    label: str = Field(min_length=1)


class ObservationSource(BaseModel):
    """The record an observation came from; `tool_use_id` only for tool results."""

    model_config = ConfigDict(frozen=True)

    uuid: str
    tool_use_id: str | None = None


class ObservationNode(BaseModel):
    """An observation that moved the trajectory, or one whose extraction failed."""

    model_config = ConfigDict(frozen=True)

    type: Literal["observation"] = "observation"
    label: str = Field(min_length=1)
    source: ObservationSource
    needs_review: Literal[True] | None = None


Node = Annotated[IntentNode | AggregateNode | ObservationNode, Field(discriminator="type")]


class Edge(BaseModel):
    """A relation between two nodes of one topic, set up in span `span`."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    relation: Relation
    span: str
    evidence: str = Field(min_length=1)


class Graph(BaseModel):
    """A topic's intent graph; its intent nodes are also the topic's intent registry."""

    model_config = ConfigDict(frozen=True)

    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)


class Meta(BaseModel):
    """`topic` is None for the unassigned file.

    `tool_intents` maps aggregate id -> tool-call intent id -> its text. It sits outside
    `graph` because the graph goes into every call's header and these do not.
    """

    model_config = ConfigDict(frozen=True)

    topic: str | None
    graph: Graph = Field(default_factory=Graph)
    tool_intents: dict[str, dict[str, str]] = Field(default_factory=dict)


class SpanSource(BaseModel):
    """Where a span's content is in the merged conversation.

    A user text or user signal span names its record by `uuid`; a tool call span names
    its calls by `tool_use_ids`; a rejection also names the rejected call.
    """

    model_config = ConfigDict(frozen=True)

    merged_file: str
    session: str
    uuid: str | None = None
    tool_use_ids: list[str] | None = None
    tool_use_id: str | None = None
    timestamp: str


class SpanIntent(BaseModel):
    """How intent `id` shows up in one span; `note` gives the grounds of an inferred one."""

    model_config = ConfigDict(frozen=True)

    id: str
    evidence: str = Field(min_length=1)
    explicit: bool
    note: str | None = None


class Span(BaseModel):
    """A stretch of the timeline over which the intent set does not change."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["user_text", "tool_calls", "user_signal"]
    source: SpanSource
    chars: tuple[int, int] | None = None
    text: str
    intents: list[SpanIntent]
    note: str | None = None
    needs_review: Literal[True] | None = None


class TopicStore(BaseModel):
    """One topic file, or the unassigned file."""

    model_config = ConfigDict(frozen=True)

    meta: Meta
    spans: list[Span] = Field(default_factory=list)


def load_store(path: Path) -> TopicStore:
    """Read and validate one store file."""
    return TopicStore.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def dump_store(store: TopicStore) -> str:
    """The store as YAML in the field order of the format, leaving out unset fields."""
    data = store.model_dump(by_alias=True, exclude_none=True)
    # The unassigned file says `topic: null` rather than leaving the key out.
    data["meta"] = {"topic": store.meta.topic, **data["meta"]}
    if not store.meta.tool_intents:
        del data["meta"]["tool_intents"]
    return STORE_HEADER + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class _OpenToolSpan:
    """The last tool call span, which the next call may still join."""

    topic: str | None
    position: int
    aggregates: frozenset[str]
    item_index: int


@attrs.define(slots=True, kw_only=True)
class Stores:
    """Every store of one run, filled unit by unit; ids are global across topics.

    `None` keys the unassigned file. Merging a tool call into the span before it
    needs the timeline, to see that nothing the user or the AI said came between.
    """

    merged_file: str
    items: Sequence[Item]
    stores: dict[str | None, TopicStore] = attrs.field(factory=dict)
    intents: int = 0
    aggregates: int = 0
    tool_intents: int = 0
    spans: int = 0
    _open_tool_span: _OpenToolSpan | None = None

    def next_intent_id(self, offset: int = 0) -> str:
        """The id the intent registered `offset` places after the next one will take."""
        return f"I{self.intents + offset + 1:03d}"

    def topic_nodes(self, topic: str) -> dict[str, Node]:
        """The nodes of `topic`'s graph so far."""
        return self.stores[topic].meta.graph.nodes

    def open_topic(self, topic: str) -> None:
        """Start the empty store of a newly opened topic."""
        self.stores[topic] = TopicStore(meta=Meta(topic=topic))

    def next_aggregate_id(self, offset: int = 0) -> str:
        """The id the aggregate registered `offset` places after the next one will take."""
        return f"G{self.aggregates + offset + 1:03d}"

    def next_tool_intent_id(self, offset: int = 0) -> str:
        """The id the tool-call intent registered `offset` places after the next will take."""
        return f"L{self.tool_intents + offset + 1:03d}"

    def register_intent(self, topic: str, node_id: str, label: str) -> None:
        """Add user intent `node_id` to `topic`'s registry; it must be the next global id."""
        if node_id != self.next_intent_id():
            raise AssertionError(
                f"Registering {node_id} when the next id is {self.next_intent_id()}."
            )
        self._add_node(topic, node_id, IntentNode(label=label))
        self.intents += 1

    def register_aggregate(self, topic: str, node_id: str, label: str) -> None:
        """Add aggregate `node_id` to `topic`'s graph; it must be the next global id."""
        if node_id != self.next_aggregate_id():
            raise AssertionError(
                f"Registering {node_id} when the next id is {self.next_aggregate_id()}."
            )
        self._add_node(topic, node_id, AggregateNode(label=label))
        self.aggregates += 1

    def register_tool_intent(self, topic: str, aggregate: str, intent_id: str, text: str) -> None:
        """List tool-call intent `intent_id` under `aggregate`; it must be the next global id."""
        if intent_id != self.next_tool_intent_id():
            raise AssertionError(
                f"Registering {intent_id} when the next id is {self.next_tool_intent_id()}."
            )
        store = self.stores[topic]
        members = store.meta.tool_intents.get(aggregate, {})
        mapping = {**store.meta.tool_intents, aggregate: {**members, intent_id: text}}
        self.stores[topic] = store.model_copy(
            update={"meta": store.meta.model_copy(update={"tool_intents": mapping})}
        )
        self.tool_intents += 1

    def _add_node(self, topic: str, node_id: str, node: IntentNode | AggregateNode) -> None:
        store = self.stores[topic]
        nodes = {**store.meta.graph.nodes, node_id: node}
        graph = store.meta.graph.model_copy(update={"nodes": nodes})
        self.stores[topic] = store.model_copy(
            update={"meta": store.meta.model_copy(update={"graph": graph})}
        )

    def add_user_text(
        self,
        topic: str | None,
        message: UserMessage,
        chars: tuple[int, int],
        intents: list[SpanIntent],
        needs_review: bool,
    ) -> None:
        """A span for one user segment; the last segment of a message triggers its coverage check."""
        source = self._source(message.source.session, message.source.timestamp.isoformat())
        span = self._span(
            kind="user_text",
            source=source.model_copy(update={"uuid": message.source.uuid}),
            chars=chars,
            text=message.body[chars[0] : chars[1]],
            intents=intents,
            needs_review=needs_review,
        )
        self._append(topic, span)
        self._open_tool_span = None
        if chars[1] == len(message.body):
            self.check_coverage(message)

    def add_user_signal(
        self, topic: str | None, signal: UserSignal, intents: list[SpanIntent], needs_review: bool
    ) -> None:
        """A user signal always stands alone."""
        source = self._source(signal.source.session, signal.source.timestamp.isoformat())
        update = {"uuid": signal.source.uuid, "tool_use_id": signal.tool_use_id}
        span = self._span(
            kind="user_signal",
            source=source.model_copy(update=update),
            text=signal.text,
            intents=intents,
            needs_review=needs_review,
        )
        self._append(topic, span)
        self._open_tool_span = None

    def add_tool_call(
        self,
        topic: str | None,
        call: ToolCall,
        item_index: int,
        intents: list[SpanIntent],
        aggregates: frozenset[str],
        needs_review: bool,
    ) -> None:
        """Join the span before when docs/workflow.md's four merge conditions hold, else open one.

        The merge compares `aggregates`, the level of what the user says; a joined call
        adds its id, its text and every one of its tool-call intents.
        """
        last = self._open_tool_span
        if (
            last is not None
            and not needs_review
            and last.topic == topic
            and last.aggregates == aggregates
            and self._adjacent(last.item_index, item_index)
        ):
            spans = list(self.stores[topic].spans)
            joined = spans[last.position]
            assert joined.source.tool_use_ids is not None
            source = joined.source.model_copy(
                update={"tool_use_ids": [*joined.source.tool_use_ids, call.tool_use_id]}
            )
            spans[last.position] = joined.model_copy(
                update={
                    "source": source,
                    "text": f"{joined.text}\n\n{call.text}",
                    "intents": [*joined.intents, *intents],
                }
            )
            self.stores[topic] = self.stores[topic].model_copy(update={"spans": spans})
            self._open_tool_span = attrs.evolve(last, item_index=item_index)
            return
        source = self._source(call.source.session, call.source.timestamp.isoformat())
        span = self._span(
            kind="tool_calls",
            source=source.model_copy(update={"tool_use_ids": [call.tool_use_id]}),
            text=call.text,
            intents=intents,
            needs_review=needs_review,
        )
        position = self._append(topic, span)
        self._open_tool_span = (
            None
            if needs_review
            else _OpenToolSpan(
                topic=topic, position=position, aggregates=aggregates, item_index=item_index
            )
        )

    def check_coverage(self, message: UserMessage) -> None:
        """Raise when the message's spans across all stores do not tile its body exactly."""
        chars = sorted(
            span.chars
            for store in self.stores.values()
            for span in store.spans
            if span.kind == "user_text"
            and span.source.uuid == message.source.uuid
            and span.chars is not None
        )
        ends = [0, *(end for _, end in chars)]
        starts = [start for start, _ in chars]
        if starts != ends[:-1] or ends[-1] != len(message.body):
            raise AssertionError(
                f"The spans of message {message.source.uuid} cover {chars}, not [0, "
                f"{len(message.body)}) end to end; this is a bug in the span step."
            )

    def _adjacent(self, before: int, after: int) -> bool:
        """No user text, user signal or AI text lies between the two timeline items."""
        said = (UserMessage, UserSignal, AIText)
        return not any(isinstance(item, said) for item in self.items[before + 1 : after])

    def _source(self, session: str, timestamp: str) -> SpanSource:
        return SpanSource(merged_file=self.merged_file, session=session, timestamp=timestamp)

    def _span(self, *, needs_review: bool, **fields: object) -> Span:
        self.spans += 1
        return Span.model_validate(
            {"id": f"S{self.spans:03d}", "needs_review": True if needs_review else None, **fields}
        )

    def _append(self, topic: str | None, span: Span) -> int:
        store = self.stores.get(topic) or TopicStore(meta=Meta(topic=topic))
        self.stores[topic] = store.model_copy(update={"spans": [*store.spans, span]})
        return len(store.spans)


def write_stores(topics_dir: Path, stores: Stores, files: dict[str, Path]) -> None:
    """Write every store: a topic's to its file, the unassigned one to unassigned.yaml."""
    topics_dir.mkdir(parents=True, exist_ok=True)
    for topic, store in stores.stores.items():
        path = files[topic] if topic is not None else topics_dir / UNASSIGNED_FILE
        path.write_text(dump_store(store), encoding="utf-8")
