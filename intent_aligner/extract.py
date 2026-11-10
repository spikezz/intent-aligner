"""The main loop of trajectory extraction: the pointer, call T, the intents and the stores.

The pointer walks the units of a conversation in timeline order, each speech
(a user message or an AI text) replaced by its segments (intent_aligner.segment).
A tool result takes the topic of the call that produced it; every
other unit goes to call T, which picks a registered topic or opens a new one.

A unit on the intent path then gets its intents: a speech segment's from segmentation,
a tool call's from level-1 compression, a user signal's from call A. Call B matches
intents against the registry entries of the same actor; call G files each tool-call intent under
an aggregate, the only form tool calls take in the graph, and call L matches it
against that aggregate's tool-call intents, so a repeated one keeps its id. The code gives new nodes
global ids, and the unit's span goes into the topic's store (intent_aligner.store).
Call C then links the unit's intents to the topic's nodes. An observation goes to call
D. A tool result is paired with its call outside the graph and updates the observation
node of the call's aggregate, one node per aggregate. Any other observation that neither
satisfies nor blocks an intent stays pending, a candidate for a later triggers edge, and
gets a node only if call C picks it.

Tool work enters the window as level-1 digests, one per tool call and result. When a
prompt would not fit, the oldest tool records of the window become one passage, batch by
batch until it fits, so the newer records stay one by one. Call G alone also sees what
follows its unit (window.render_following), because an aggregate names what a run of
operations adds up to.

A run writes these files under --out:

    assignments.jsonl      one line per unit in pointer order: its topic and its intents
                           with their ids; a rerun resumes after the last line
    tool_compressions.jsonl  one line per passage that replaced tool records of the window
    topics.yaml            the topic list (format in docs/trajectory-extractor.md)
    topics/<id>.yaml       one store per topic; topics/unassigned.yaml for units without one
    run.log                new topics, intentless speeches and needs_review units

topics.yaml and the stores are rebuilt from assignments.jsonl, the only record of
what the models decided, so they always agree with it.

Usage: python -m intent_aligner.extract MERGED.jsonl --digests DIGESTS.jsonl
           --segments SEGMENTS.jsonl --backend codex --model gpt-6.1-sol
           [--window-rounds 2] --out DIR
The digests (intent_aligner.compress) must cover the whole conversation. The main loop
cuts each speech when it reaches it, with the graphs built so far, and appends to --segments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import attrs
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intent_aligner.compress import (
    ROUND_SYSTEM_PROMPT,
    SUMMARY_SCHEMA,
    summarize_trajectory,
)
from intent_aligner.intent import INTENT_SCHEMA, Intent, is_excerpt, stray_evidence
from intent_aligner.llm import (
    CALL_SUBJECT,
    CALLS_FILE,
    TRUNCATED_FEEDBACK,
    Ask,
    CallLog,
    TruncatedReply,
    asker,
)
from intent_aligner.segment import (
    SCHEMA as SEGMENT_SCHEMA,
)
from intent_aligner.segment import (
    SYSTEM_PROMPT as SEGMENT_SYSTEM_PROMPT,
)
from intent_aligner.segment import (
    MessageSegments,
    Segment,
    message_window,
    next_segment,
    review_segment,
)
from intent_aligner.store import (
    STORE_HEADER,
    AggregateNode,
    AggregateObservationNode,
    Edge,
    IntentNode,
    ObservationNode,
    ObservationSource,
    PendingObservation,
    SpanIntent,
    Stores,
    graph_with_status,
    write_stores,
)
from intent_aligner.units import (
    SPEAKER_NAMES,
    Item,
    Observation,
    Speech,
    ToolCall,
    UserSignal,
    conversation_units,
)
from intent_aligner.window import (
    COMPRESSION_BATCH,
    DEFAULT_WINDOW_ROUNDS,
    PROMPT_BUDGET,
    Count,
    Digests,
    ToolCompression,
    WindowError,
    covering,
    is_tool_record,
    load_digests,
    qwen_tokens,
    render_following,
    round_entry,
    window_rounds,
    window_start,
    window_text,
)

T_ATTEMPTS = 2
CALL_ATTEMPTS = 3
# Room kept free for retry feedback, which quotes a pydantic error at most.
FEEDBACK_ROOM = 500
NEW = "NEW"

T_SYSTEM_PROMPT = """\
你在判断对话里的一个单元属于哪个 topic。topic 是对话里一件独立的事情：一个 session 里可以穿插几件互不相干的事，一件事也可以延续好几个 session。

<header> 是结构化的信息：topics 是已经登记的 topic 列表（编号和标题）；graphs 是每个 topic 的意图图谱，键是 topic 编号，nodes 是已经登记的意图、聚合和观测，edges 是它们之间的依赖；unit 是当前单元的类型、发出者和时间，发言段落还有它在发言正文里的字符区间 chars。

<此前的对话> 是当前单元之前的对话。更早的轮次里，AI 的工具操作压成了每轮一段概括（【AI 的工具操作】）；当前这一轮已经发生的工具调用和工具结果逐条列出，每条是一句概括（【AI 的工具调用】【工具结果】）。

<当前单元> 是要判断的单元。发言段落给出整次发言，用 <当前段落> 和 </当前段落> 标出要判断的那一段；工具调用给出它的意图和操作对象的概括。

回答两项：
- topic：当前单元在继续、执行或支撑某个 topic 图谱里的意图时，回答这个 topic 的编号；和已有 topic 的图谱都接不上时，回答 NEW。
- title：topic 为 NEW 时，写新 topic 的标题，一句完整的话，写清楚这件事是什么；topic 不是 NEW 时为空字符串。
"""

T_SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}, "title": {"type": "string"}},
    "required": ["topic", "title"],
    "additionalProperties": False,
}

A_SYSTEM_PROMPT = """\
你在判断用户的一个信号表达了什么意图。用户信号是用户不通过打字表达的表态：用户按 ESC 中断 AI，记录里留下 [Request interrupted by user]；或者用户拒绝 AI 的一个工具调用，记录里留下以 The user doesn't want to 开头的信息，用户写了话时，话附在 the user said: 后面。

<header> 是结构化的信息：topics 是已经登记的 topic 列表；current_topic 是这个信号所属的 topic；unit 是这个信号的类型和时间；graph 是这个 topic 的意图图谱，nodes 是已经登记的意图（编号、发出者、标准文本），edges 是它们之间的关系。

<此前的对话> 是这个信号之前的对话。更早的轮次里，AI 的工具操作压成了每轮一段概括；当前这一轮已经发生的工具调用和工具结果逐条列出，每条是一句概括。

<当前单元> 是这个信号的原文。

回答 intents：用户通过这个信号表达的意图，每个意图一项，没有意图时是空列表。信号本身通常只说明用户要 AI 停下或不要这样做，停下来针对的是什么、为什么，要靠此前的对话推断。用户在信号之后接着发的消息另外处理，这里只写这个信号本身表达的意图。每项回答四项：
- text：意图，一句完整的话，用此前的对话把“这个”“那里”换成具体的事，脱离对话也能看懂。
- evidence：这个意图在 <当前单元> 原文里的依据，原样摘录。可以用 … 省略中间的部分，但被 … 分开的每个片段都必须原样出现在原文里，并且按原文的先后顺序排列。
- explicit：用户写的话里说出了这个意图就是 true；意图要靠此前的对话或常识推断时是 false。
- note：explicit 为 false 时写推断出了什么、依据的是哪件事；explicit 为 true 时写空字符串。
"""

A_SCHEMA = {
    "type": "object",
    "properties": {"intents": {"type": "array", "items": INTENT_SCHEMA}},
    "required": ["intents"],
    "additionalProperties": False,
}

B_SYSTEM_PROMPT = """\
你在把对话里一个单元的意图对照意图注册表：判断每个意图是这个 topic 已经登记的哪一个意图，还是新意图。

<header> 是结构化的信息：topics 是已经登记的 topic 列表；current_topic 是这个单元所属的 topic；unit 是这个单元的类型、发出者和时间；graph 是这个 topic 的意图图谱，nodes 是已经登记的意图（编号、发出者、标准文本），也就是注册表，edges 是它们之间的关系；candidates 是可以回答的已有编号，只包含和这个单元同一发出者的意图：用户的意图只和用户的意图比较，AI 的意图只和 AI 的意图比较。

<此前的对话> 是这个单元之前的对话。<当前单元> 是这个单元：发言段落给出整次发言，用 <当前段落> 和 </当前段落> 标出这一段；工具调用给出它的意图和操作对象的概括。<本单元的意图> 按顺序列出这个单元的每个意图：编号、文本和原文依据。

按 <本单元的意图> 的顺序，每个意图回答一项：
- match：这个意图和注册表里的某个意图是同一个意图时，回答那个意图的编号；和注册表里的意图都不同时，回答 NEW。判断看意图本身是否相同，不看措辞是否相似：同一个意图换一种说法再出现，仍然回答原来的编号；措辞相近但要做的事不同，就是不同的意图。
- complete：match 是已有编号时回答，已有意图的文本是否完整包含了这个单元关于这个意图的信息。这个单元给已有意图补充了新的信息，比如指定了具体用哪个文件、加了限定条件，就回答 false。match 是 NEW 时回答 true。
"""

G_SYSTEM_PROMPT = """\
你在把 AI 一个工具调用的底层意图归入意图聚合。底层意图是具体操作，比如读某个文件的某几行；意图聚合说明一串操作合起来在做什么，它和发言的意图在同一个层级，比如“调查 somnia-farm 的文档和目录结构”“搭建音频分析环境”。

<header> 是结构化的信息：topics 是已经登记的 topic 列表；current_topic 是这个工具调用所属的 topic；unit 是这个单元的类型和时间；graph 是这个 topic 的意图图谱，type 为 intent 的节点是用户或 AI 发言的意图（actor 说明是谁），type 为 aggregate 的节点是已有的意图聚合，label 是它们的标准文本；candidates 是可以回答的已有聚合编号，可以是空列表。

<此前的对话> 是这个工具调用之前的对话。<当前单元> 是这个工具调用的意图和操作对象的概括。<本单元的意图> 按顺序列出这个调用的每个底层意图：编号、文本和原文依据。<此后的对话> 是这个工具调用之后接着发生的事，用来看清这个操作最终是为了做成什么；要回答的只有 <本单元的意图> 里的意图。

按 <本单元的意图> 的顺序，每个底层意图回答一项：
- aggregate：这个操作是在继续某个已有聚合做的事时，回答那个聚合的编号；它在做一件已有聚合都没有覆盖的事时，回答 NEW。一个聚合是 current_topic 里的一段工具操作，范围比 topic 小。比如写任务分发脚本、提交小规模测试、查看测试输出，合起来是“跑通关键帧生成流程”这一个聚合。
- title：aggregate 是 NEW 时，写这个聚合的标准文本：一句完整的话，说明这一串操作最终要做成什么，要涵盖 <此后的对话> 里为做成同一样东西继续进行的操作，比如先检查依赖、再安装、再运行分析，标题写“完成歌曲结构分析”；<此后的对话> 里开始做成另一样东西的操作不算在内。这个聚合不能涵盖 current_topic 的全部操作，能涵盖就说明粒度太粗。标题要脱离对话也能看懂，只写要做成的目标，类名、参数名、文件名、行号和命令都留在底层意图里。aggregate 是已有编号时写空字符串。同一个调用里多个底层意图做的是同一件新事时，title 写成完全相同的文本，它们会归入同一个新聚合。
"""

L_SYSTEM_PROMPT = """\
你在给 AI 一个工具调用的底层意图去重：判断每个底层意图和同一个意图聚合里已经登记的底层意图是不是同一个。

<header> 是结构化的信息：topics 是已经登记的 topic 列表；current_topic 是这个工具调用所属的 topic；unit 是这个单元的类型和时间；graph 是这个 topic 的意图图谱；candidates 是可以回答的已有底层意图编号。

<此前的对话> 是这个工具调用之前的对话。<当前单元> 是这个工具调用的意图和操作对象的概括。<本单元的意图> 按顺序列出这个调用归入这个聚合的底层意图：编号、文本和原文依据。<已有的底层意图> 列出这个聚合已经登记的底层意图：编号和文本。

按 <本单元的意图> 的顺序，每个意图回答一项：
- match：这个底层意图和某个已有底层意图是同一个，并且已有的文本完整包含了这个意图的信息时，回答那个编号；否则回答 NEW。判断看要做的事和操作对象是否相同，不看措辞是否相似：同一个操作换一种说法仍然回答原来的编号；操作对象不同，比如读的是另一个文件，就是 NEW。
"""

L_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"match": {"type": "string"}},
                "required": ["match"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

G_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"aggregate": {"type": "string"}, "title": {"type": "string"}},
                "required": ["aggregate", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

B_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"match": {"type": "string"}, "complete": {"type": "boolean"}},
                "required": ["match", "complete"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

_C_TASK = """\
你在判断对话里一个单元的意图和这个 topic 的其他节点之间有什么关系。

<header> 是结构化的信息：topics 是已经登记的 topic 列表；current_topic 是这个单元所属的 topic；unit 是这个单元的类型、发出者和时间；graph 是这个 topic 的意图图谱，nodes 是已有节点（intent 是用户或 AI 的意图，actor 说明是谁，aggregate 是工具操作的意图聚合，observation 是环境推送的观测，aggregate_observation 是一个聚合到目前为止的观测），edges 是已有的边；unit_intents 是这个单元的意图或聚合，编号和文本；candidates 是可以连接的节点：字符串是图谱里的编号，带 id 和 text 的对象是还没有建节点的观测；relations 是可以使用的关系，from 和 to 说明两端的节点种类，intent 既指用户或 AI 的意图，也指聚合。

<此前的对话> 是这个单元之前的对话。<当前单元> 是这个单元：发言段落给出整次发言，用 <当前段落> 和 </当前段落> 标出这一段；工具调用给出它的意图和操作对象的概括。

回答 edges：每条关系一项，没有关系时是空列表。每条边的两端必须有一端在 unit_intents 里，另一端在 candidates 里。每项回答四项：
- from、to：起点和终点的编号。
- relation：关系类型，含义见 relations。
- evidence：这条关系的依据，从 <当前单元> 或 <此前的对话> 原样摘录。可以用 … 省略中间的部分，但被 … 分开的每个片段都必须原样出现，并且按原文的先后顺序排列。
"""

C_SYSTEM_PROMPT = (
    _C_TASK
    + """
某个观测引出这个单元的意图时，记这个观测到这个意图的 triggers 边。triggers 的 to 是产生这个观测的聚合之外的意图或聚合；同一个聚合内部，结果引出下一步操作属于这个聚合本身的推进。记下这个单元实际表现出来的全部关系。
"""
)

# A tool call filed under an aggregate that already has a node: the aggregate's edges
# were recorded when it was set up, and asked for "all relations of this unit" the model
# kept weighing whether to copy them (unit 30 of claude-87eeb4bd-qwen thought to the cap).
C_STEP_SYSTEM_PROMPT = (
    _C_TASK
    + """
这个单元是一个已有聚合里的又一步工具调用：unit_intents 就是这个聚合，图谱里有这个聚合已有的边。记下这一步新表现出来的关系：产生于这个聚合之外的观测引出了这一步时，记这个观测到这个聚合的 triggers 边；这一步表明这个聚合同时在为另一个意图服务时，记这个聚合到那个意图的 serves 边。
"""
)

_C_RELATIONS: dict[str, tuple[str, str, str]] = {
    "serves": ("intent", "intent", "from 直接服务于 to"),
    "triggers": ("observation", "intent", "from 这个观测引出了 to 这个意图"),
    "responds": ("intent", "intent", "from 回应对方说的话 to：一端是用户，另一端是 AI"),
}

C_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "relation": {"type": "string", "enum": list(_C_RELATIONS)},
                    "evidence": {"type": "string", "minLength": 1},
                },
                "required": ["from", "to", "relation", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edges"],
    "additionalProperties": False,
}

_D_COMMON = """\
你在判断环境返回的一个观测对已有的意图有什么影响。

<header> 是结构化的信息：topics 是已经登记的 topic 列表；current_topic 是这个观测所属的 topic；unit 是这个观测的类型和时间；graph 是这个 topic 的意图图谱；candidates 是可以回答的意图或聚合编号；relations 是可以使用的关系。

<此前的对话> 是这个观测之前的对话。<当前单元> 是这个观测。

回答 outcomes：这个观测达成或阻碍的每个已有意图各一项，都没有时是空列表。每项回答两项：
- relation：satisfies 表示这个观测表明这件事已经做成；blocks 表示这个观测表明这个意图没有达成，或者遇到了障碍。
- target：这个意图的编号，只能是 candidates 里的编号。

意图允许一直不被完成，对话结束时仍然“待完成”是正常的结果。

观测表明这件事已经做成时，才回答 satisfies。观测只说明事情推进了一步，或者看不出是否做成时，outcomes 里不写这个意图。
"""

D_SYSTEM_PROMPT = _D_COMMON

D_UPDATE_SYSTEM_PROMPT = (
    _D_COMMON
    + """
这个工具结果属于聚合 aggregate，aggregate_observation 是这个聚合到上一个结果为止的观测。

另外回答 summary：根据这个结果，在 aggregate_observation 的基础上重写成一句完整的话，说明这个聚合现在做到了哪一步，和聚合的标题在同一层，脱离对话也能看懂。还没有结果的就写还没有结果。

outcomes 回答合并后的观测现在表明的全部结果：之前成立并且仍然成立的要继续写，已经不成立的不写。
"""
)

D_SUMMARY_SYSTEM_PROMPT = (
    _D_COMMON
    + """
另外回答 summary：用一句完整的话概括这个观测报告了什么，脱离对话也能看懂。
"""
)

_D_OUTCOMES = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "relation": {"type": "string", "enum": ["satisfies", "blocks"]},
            "target": {"type": "string"},
        },
        "required": ["relation", "target"],
        "additionalProperties": False,
    },
}

D_SCHEMA = {
    "type": "object",
    "properties": {"outcomes": _D_OUTCOMES},
    "required": ["outcomes"],
    "additionalProperties": False,
}

D_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"outcomes": _D_OUTCOMES, "summary": {"type": "string"}},
    "required": ["outcomes", "summary"],
    "additionalProperties": False,
}

UnitKind = Literal["user_segment", "ai_segment", "user_signal", "tool_call", "observation"]


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class SpeechSegment:
    """One segment of a speech; the pointer visits it in place of the whole speech."""

    speech: Speech
    segment: Segment


Unit = SpeechSegment | UserSignal | ToolCall | Observation


class UnitRef(BaseModel):
    """Where a unit sits in the merged conversation; a resumed run checks it still matches."""

    model_config = ConfigDict(frozen=True)

    kind: UnitKind
    uuid: str
    tool_use_id: str | None = None
    chars: tuple[int, int] | None = None


class UnitIntent(BaseModel):
    """One intent of a unit under its registry id, with how it shows up in the unit.

    `label` is set when this unit registered the node. `details_of` is the existing
    intent call B found incomplete; this new one serves it. A tool
    call's intent has its own `id` and its `aggregate`; `text` is set when this unit
    registered it, and `label` when it opened that aggregate.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    aggregate: str | None = None
    text: str | None = None
    evidence: str
    explicit: bool
    note: str
    label: str | None = None
    details_of: str | None = None


class NewObservation(BaseModel):
    """An observation node this unit built; `item` is the observation's timeline index."""

    model_config = ConfigDict(frozen=True)

    id: str
    item: int
    label: str
    source: ObservationSource
    needs_review: bool = False


class NewAggregateObservation(BaseModel):
    """The latest text of `aggregate`'s observation node, new or updated by this unit."""

    model_config = ConfigDict(frozen=True)

    id: str
    aggregate: str
    label: str


class Assignment(BaseModel):
    """One line of assignments.jsonl; `topic` is None when the unit is unassigned.

    `new_title` is set on the unit that opened its topic. `review_reason` says why a
    unit needs review; `attempts` counts call T's calls and is 0 when none was made.
    `edges` are the edges this unit set up, their spans filled in when the stores are
    folded. `observations` are the nodes it built: its own, or those its triggers edges
    pointed at. `aggregate_observations` update the nodes of a tool result's aggregates,
    and `tool_result` is the text paired with its call. `pending_text` is the text of an
    observation left without a node, and `placeholder` marks a whole speech without intents.
    """

    model_config = ConfigDict(frozen=True)

    unit: UnitRef
    topic: str | None
    new_title: str | None = None
    intents: list[UnitIntent] = Field(default_factory=list)
    needs_review: bool = False
    review_reason: str | None = None
    attempts: int
    edges: list[Edge] = Field(default_factory=list)
    observations: list[NewObservation] = Field(default_factory=list)
    aggregate_observations: list[NewAggregateObservation] = Field(default_factory=list)
    tool_result: str | None = None
    pending_text: str | None = None
    placeholder: bool = False


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


class _AAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    intents: list[Intent]


class _Match(BaseModel):
    model_config = ConfigDict(frozen=True)

    match: str
    complete: bool


class _Filed(BaseModel):
    model_config = ConfigDict(frozen=True)

    aggregate: str
    title: str


class _GAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answers: list[_Filed]


class _ToolMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    match: str


class _LAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answers: list[_ToolMatch]


class _BAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answers: list[_Match]


class _CEdge(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    relation: Literal["serves", "triggers", "responds"]
    evidence: str = Field(min_length=1)


class _CAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    edges: list[_CEdge]


class _Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation: Literal["satisfies", "blocks"]
    target: str


_OBSERVATION_NODES = (AggregateObservationNode, ObservationNode)


class _DAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcomes: list[_Outcome]
    summary: str = ""


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Asks:
    """The model behind each call; `round` must carry compress.ROUND_SYSTEM_PROMPT."""

    topic: Ask
    signal: Ask
    registry: Ask
    aggregate: Ask
    tool_intent: Ask
    relation: Ask
    step_relation: Ask
    outcome: Ask
    outcome_summary: Ask
    outcome_update: Ask
    round: Ask
    segment: Ask


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class _Step:
    """A unit, the timeline index of its item, and where that item's round opens.

    `round_start` is None for tool work before the first speech.
    """

    unit: Unit
    index: int
    round_start: int | None


def walk(items: Sequence[Item], segments: dict[str, list[Segment]]) -> Iterator[_Step]:
    """The units in timeline order."""
    round_start: int | None = None
    for index, item in enumerate(items):
        if isinstance(item, Speech):
            round_start = index
            if item.source.uuid not in segments:
                raise ValueError(
                    f"Speech {item.source.uuid} has no segments; run "
                    "intent_aligner.segment on the whole conversation first."
                )
            for segment in segments[item.source.uuid]:
                unit = SpeechSegment(speech=item, segment=segment)
                yield _Step(unit=unit, index=index, round_start=round_start)
        else:
            yield _Step(unit=item, index=index, round_start=round_start)


def walk_so_far(
    items: Sequence[Item], segments: dict[str, list[Segment]], digests: Digests
) -> list[_Step]:
    """The units a run has reached: those before the first one still lacking its segments or digest."""
    reached: list[Item] = []
    for item in items:
        if isinstance(item, Speech) and item.source.uuid not in segments:
            break
        if isinstance(item, ToolCall) and item.tool_use_id not in digests.calls:
            break
        if (
            isinstance(item, Observation)
            and item.tool_use_id is not None
            and item.tool_use_id not in digests.results
        ):
            break
        reached.append(item)
    return list(walk(reached, segments))


def unit_ref(unit: Unit) -> UnitRef:
    """The UnitRef of `unit`."""
    if isinstance(unit, SpeechSegment):
        chars = (unit.segment.start, unit.segment.end)
        speech_kind: UnitKind = "user_segment" if unit.speech.speaker == "user" else "ai_segment"
        return UnitRef(kind=speech_kind, uuid=unit.speech.source.uuid, chars=chars)
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


def extract_units(
    items: Sequence[Item],
    segments: dict[str, list[Segment]],
    digests: Digests,
    done: Sequence[Assignment],
    compressed: Sequence[ToolCompression],
    asks: Asks,
    count: Count,
    stores: Stores,
    window_rounds: int,
) -> Iterator[Assignment | ToolCompression | MessageSegments]:
    """Yield a record for every unit after the `done` ones, and what was made on the way.

    The units run one at a time because each call sees the graphs built before it, so
    a speech is cut right before its segments run: every segment missing from
    `segments` is yielded when made, and added to it. `stores` takes every record, the
    `done` ones replayed first, so it holds the stores as of the last record yielded. A
    compression is yielded before the record whose prompt needed it.
    Raises ValueError when `done` no longer lines up with the conversation's units.
    """
    compressions = list(compressed)
    topics = [(a.topic, a.new_title) for a in done if a.topic and a.new_title]
    context = _Context(
        items=items,
        digests=digests,
        topics=topics,
        count=count,
        window_rounds=window_rounds,
        stores=stores,
    )
    call_topics: dict[str, str | None] = {}
    position = 0

    def process(step: _Step) -> Iterator[Assignment | ToolCompression]:
        nonlocal position
        ref = unit_ref(step.unit)
        if position < len(done):
            record = done[position]
            if record.unit != ref:
                raise ValueError(
                    f"Unit {position} is {ref} but assignments.jsonl has {record.unit}; "
                    "the conversation or its segments changed, so start a new --out."
                )
            into_stores(stores, step, record)
        else:
            window = _Round(in_force=compressions, ask=asks.round)
            CALL_SUBJECT.set(f"unit:{position}")
            record = _assign(step, ref, context, window, call_topics, asks.topic)
            if record.new_title is not None:
                assert record.topic is not None
                topics.append((record.topic, record.new_title))
            if isinstance(step.unit, Observation):
                record = _observe(step, step.unit, record, context, window, asks, stores)
            else:
                record = _with_intents(step, record, context, window, asks, stores)
                if record.intents and not record.needs_review:
                    record = _relate(step, record, context, window, asks, stores)
            into_stores(stores, step, record)
            yield from window.fresh
            yield record
        if isinstance(step.unit, ToolCall):
            call_topics[step.unit.tool_use_id] = record.topic
        position += 1

    round_start: int | None = None
    for index, item in enumerate(items):
        if isinstance(item, Speech):
            round_start = index
            for made in _speech_segments(index, item, segments, compressions, context, asks):
                if isinstance(made, Segment):
                    unit = SpeechSegment(speech=item, segment=made)
                    yield from process(_Step(unit=unit, index=index, round_start=round_start))
                else:
                    yield made
        else:
            yield from process(_Step(unit=item, index=index, round_start=round_start))


def _graphs(stores: Stores) -> dict[str, object]:
    return {
        topic: graph_with_status(store.meta.graph)
        for topic, store in stores.stores.items()
        if topic is not None
    }


def _graphs_block(stores: Stores) -> str:
    """The header of segmentation, which runs before a topic is known."""
    return header_block({"graphs": _graphs(stores)})


def header_block(header: dict[str, object]) -> str:
    """The <header> block of a prompt, one topic, graph node or edge per line (see workflow.md)."""
    members = [f"{_compact(key)}:{_spread(value, key)}" for key, value in header.items()]
    return "<header>\n{\n" + ",\n".join(members) + "\n}\n</header>"


# The header fields whose members each take a line; under "graphs" each member is a graph.
_SPREAD_KEYS = frozenset({"topics", "graphs", "graph", "nodes", "edges"})


def _spread(value: object, key: str) -> str:
    if key not in _SPREAD_KEYS or not value:
        return _compact(value)
    if isinstance(value, dict):
        inner = ((k, "graph" if key == "graphs" else k) for k in value)
        members = [f"{_compact(k)}:{_spread(value[k], kind)}" for k, kind in inner]
        return "{\n" + ",\n".join(members) + "\n}"
    if isinstance(value, list):
        return "[\n" + ",\n".join(_compact(v) for v in value) + "\n]"
    return _compact(value)


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _speech_segments(
    index: int,
    speech: Speech,
    segments: dict[str, list[Segment]],
    compressions: list[ToolCompression],
    context: _Context,
    asks: Asks,
) -> Iterator[Segment | MessageSegments | ToolCompression]:
    """The segments of `speech` in order, each made only when the one before has been processed."""
    uuid, body = speech.source.uuid, speech.body
    if uuid not in segments and not body:
        segments[uuid] = []
        yield MessageSegments(uuid=uuid, segments=[])
        return
    known = segments.get(uuid)
    yielded = 0
    while True:
        if known is not None and yielded < len(known):
            segment = known[yielded]
        elif known is not None and (not known or known[-1].end >= len(body)):
            return
        else:
            window = _Round(in_force=compressions, ask=asks.round)
            start = 0 if known is None else known[-1].end
            segment = _make_segment(index, speech, start, context, window, asks.segment)
            yield from window.fresh
            known = segments.setdefault(uuid, [])
            known.append(segment)
            yield MessageSegments(uuid=uuid, segments=[segment])
        yield segment
        yielded += 1


def _make_segment(
    index: int, speech: Speech, start: int, context: _Context, window: _Round, ask: Ask
) -> Segment:
    CALL_SUBJECT.set(f"message:{speech.source.uuid}")

    header = _graphs_block(context.stores)

    def fitted(rounds: int) -> str:
        return message_window(
            header, context.items, index, context.digests, window.in_force, context.count, rounds
        )

    try:
        text = _within_budget(index, index, context, window, fitted)
    except WindowError as error:
        print(f"needs_review: {error}", file=sys.stderr, flush=True)
        return review_segment(speech, start, 0)
    return next_segment(header, text, speech, start, ask)


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class _Context:
    """What every prompt of one run is built from; `topics` and `stores` grow as units run."""

    items: Sequence[Item]
    digests: Digests
    topics: Sequence[tuple[str, str]]
    count: Count
    window_rounds: int
    stores: Stores


def _round_low(step: _Step) -> int:
    """Where the round shown before `step` starts; a speech segment opens its round, so none is."""
    if isinstance(step.unit, SpeechSegment):
        return step.index
    return step.round_start or 0


@attrs.define(slots=True, kw_only=True)
class _Round:
    """The passages in force around one unit; `fresh` are the ones made for it.

    `in_force` is the run's own list, so a passage made for one call of a unit is in
    force for its later calls and for the units after it.
    """

    in_force: list[ToolCompression]
    ask: Ask
    fresh: list[ToolCompression] = attrs.field(factory=list)


def _within_budget(
    low: int, index: int, context: _Context, window: _Round, attempt: Callable[[int], str]
) -> str:
    """`attempt(rounds)`, compressing the oldest tool records until it fits.

    A round is given up only when no tool record is left to compress, because dropping
    it loses what was said as well. Raises WindowError when one round does not fit.
    """
    rounds = context.window_rounds
    while True:
        try:
            return attempt(rounds)
        except WindowError:
            if _compress_oldest(low, index, rounds, context, window):
                continue
            if rounds == 1:
                raise
            rounds -= 1


def _compress_oldest(low: int, index: int, rounds: int, context: _Context, window: _Round) -> bool:
    """Turn the oldest tool records of the window into one passage; False when none are left.

    A batch stays inside one round, because its passage is written against that round's
    speech.
    """
    items, digests, count = context.items, context.digests, context.count
    batch: list[int] = []
    used = 0
    for position in range(window_start(items, low, rounds), index):
        item = items[position]
        if isinstance(item, Speech) and batch:
            break
        if is_tool_record(item) and covering(position, window.in_force) is None:
            batch.append(position)
            used += count(round_entry(item, digests))
            if used >= COMPRESSION_BATCH:
                break
    if not batch:
        return False
    lines = [" ".join(round_entry(items[position], digests).split()) for position in batch]
    speeches = [item for item in items[: batch[0]] if isinstance(item, Speech)]
    summary = summarize_trajectory(
        speeches[-1] if speeches else None,
        lines,
        window.ask,
        count,
        f"items {batch[0]} to {batch[-1]}",
    )
    compression = ToolCompression(begin=batch[0], through=batch[-1] + 1, summary=summary)
    window.in_force.append(compression)
    window.fresh.append(compression)
    return True


def _fitted_prompt(
    step: _Step,
    context: _Context,
    system: str,
    head: str,
    window: _Round,
    tail: str = "",
) -> str:
    """The prompt of one call, compressing tool records first when it would not fit otherwise.

    A tool result calls nothing, so the records it adds are checked at the next unit
    that does. Raises WindowError when the prompt does not fit even after compressing.
    """

    def fitted(rounds: int) -> str:
        return _prompt(step, context, system, head, window.in_force, tail, rounds)

    return _within_budget(_round_low(step), step.index, context, window, fitted)


def _assign(
    step: _Step,
    ref: UnitRef,
    context: _Context,
    window: _Round,
    call_topics: dict[str, str | None],
    ask: Ask,
) -> Assignment:
    unit = step.unit
    if isinstance(unit, Observation) and unit.tool_use_id is not None:
        return Assignment(unit=ref, topic=call_topics[unit.tool_use_id], attempts=0)
    head = _header(unit, context.topics, {"graphs": _graphs(context.stores)})
    try:
        prompt = _fitted_prompt(step, context, T_SYSTEM_PROMPT, head, window)
    except WindowError as error:
        return Assignment(
            unit=ref, topic=None, needs_review=True, review_reason=str(error), attempts=0
        )
    topics = context.topics
    ids = {topic_id for topic_id, _ in topics}

    def parse(reply: str) -> _TAnswer:
        answer = _TAnswer.model_validate_json(reply)
        if answer.topic == NEW and not answer.title.strip():
            raise ValueError("topic 为 NEW 时 title 不能为空。")
        if answer.topic != NEW and answer.topic not in ids:
            raise ValueError(f"{answer.topic!r} 不在 topics 列表里，只能回答列表里的编号或 NEW。")
        return answer

    answer, attempts, feedback = _ask_until_valid(ask, prompt, parse, T_ATTEMPTS)
    if answer is None:
        return Assignment(
            unit=ref,
            topic=None,
            needs_review=True,
            review_reason=f"call T gave no valid answer in {T_ATTEMPTS} attempts: {feedback}",
            attempts=attempts,
        )
    if answer.topic == NEW:
        topic_id = f"T{len(topics) + 1:03d}"
        return Assignment(
            unit=ref, topic=topic_id, new_title=answer.title.strip(), attempts=attempts
        )
    return Assignment(unit=ref, topic=answer.topic, attempts=attempts)


def _ask_until_valid[Answer](
    ask: Ask, prompt: str, parse: Callable[[str], Answer], attempts: int
) -> tuple[Answer | None, int, str]:
    """The first reply `parse` accepts, the calls made, and the last feedback.

    `parse` raises ValueError with what is wrong (a pydantic ValidationError when the
    reply is not the schema); that message goes back to the model with the prompt. A truncated reply counts
    as a failed call too.
    """
    feedback = ""
    for attempt in range(1, attempts + 1):
        try:
            return parse(ask(prompt + feedback)), attempt, ""
        except TruncatedReply:
            feedback = f"\n\n{TRUNCATED_FEEDBACK}"
        except ValidationError as error:
            feedback = f"\n\n你上一次的回答不是要求的 JSON，请重新回答：{error}"
        except ValueError as error:
            feedback = f"\n\n你上一次的回答有问题，请重新回答：{error}"
    return None, attempts, feedback.strip()


def _with_intents(
    step: _Step,
    record: Assignment,
    context: _Context,
    window: _Round,
    asks: Asks,
    stores: Stores,
) -> Assignment:
    """`record` with the unit's intents under their registry ids, or marked needs_review.

    A unit without a topic has no registry to match against, so it passes through unchanged.
    """
    unit, topic = step.unit, record.topic
    assert not isinstance(unit, Observation)
    if topic is None:
        return record
    if record.new_title is not None:
        stores.open_topic(topic)
    if isinstance(unit, SpeechSegment):
        intents = unit.segment.intents
    elif isinstance(unit, ToolCall):
        filed = context.digests.calls[unit.tool_use_id].intents
        return _file_under_aggregates(
            step, record, topic, filed, context, window, asks.aggregate, asks.tool_intent, stores
        )
    else:
        try:
            head = _graph_header(step, context, topic, stores)
            prompt = _fitted_prompt(step, context, A_SYSTEM_PROMPT, head, window)
        except WindowError as error:
            return _review(record, str(error))
        signal_text = unit.text

        def parse_a(reply: str) -> list[Intent]:
            found = _AAnswer.model_validate_json(reply).intents
            strays = stray_evidence(found, signal_text)
            if strays:
                raise ValueError(
                    f"这些 evidence 不是 <当前单元> 原文的摘录：{strays!r}。evidence 被 … 分开的"
                    "每个片段都必须原样出现在原文里，并且按原文的先后顺序排列。"
                )
            return found

        answered, _, feedback = _ask_until_valid(asks.signal, prompt, parse_a, CALL_ATTEMPTS)
        if answered is None:
            return _review(
                record, f"call A gave no valid answer in {CALL_ATTEMPTS} attempts: {feedback}"
            )
        intents = answered
    if not intents:
        whole = (
            isinstance(unit, SpeechSegment)
            and (unit.segment.start, unit.segment.end) == (0, len(unit.speech.body))
            and not unit.segment.needs_review
        )
        return record.model_copy(update={"placeholder": whole})
    return _match_intents(step, record, topic, intents, context, window, asks.registry, stores)


def _listed(intents: list[Intent]) -> str:
    lines = "\n".join(
        f"{n}. {i.text}（依据：{i.evidence}）" for n, i in enumerate(intents, start=1)
    )
    return f"<本单元的意图>\n{lines}\n</本单元的意图>"


def _file_under_aggregates(
    step: _Step,
    record: Assignment,
    topic: str,
    intents: list[Intent],
    context: _Context,
    window: _Round,
    ask: Ask,
    ask_match: Ask,
    stores: Stores,
) -> Assignment:
    """Call G, then call L for each aggregate that has members, then the ids.

    NEW answers with the same title share one new aggregate. A new aggregate has no
    members to match, so it skips call L.
    """
    candidates = [i for i, n in stores.topic_nodes(topic).items() if n.type == "aggregate"]
    head = _graph_header(step, context, topic, stores, candidates)
    try:
        following = render_following(context.items, step.index, context.digests, context.count)
        tail = f"{_listed(intents)}\n\n<此后的对话>\n{following}\n</此后的对话>"
        prompt = _fitted_prompt(step, context, G_SYSTEM_PROMPT, head, window, tail)
    except WindowError as error:
        return _review(record, str(error))
    allowed = {*candidates, NEW}

    def parse(reply: str) -> list[_Filed]:
        got = _GAnswer.model_validate_json(reply).answers
        if len(got) != len(intents):
            raise ValueError(
                f"answers 有 {len(got)} 项，但 <本单元的意图> 有 {len(intents)} 个，"
                "要按顺序给每个意图回答一项。"
            )
        wrong = [f.aggregate for f in got if f.aggregate not in allowed]
        if wrong:
            raise ValueError(
                f"{wrong!r} 不在 candidates 里，aggregate 只能是 candidates 里的编号或 NEW。"
            )
        if any(f.aggregate == NEW and not f.title.strip() for f in got):
            raise ValueError("aggregate 为 NEW 时 title 不能为空。")
        return got

    answered, _, feedback = _ask_until_valid(ask, prompt, parse, CALL_ATTEMPTS)
    if answered is None:
        return _review(
            record, f"call G gave no valid answer in {CALL_ATTEMPTS} attempts: {feedback}"
        )
    opened: dict[str, str] = {}
    filed: list[tuple[Intent, str, str | None]] = []
    for intent, answer in zip(intents, answered, strict=True):
        title = answer.title.strip()
        if answer.aggregate != NEW:
            filed.append((intent, answer.aggregate, None))
        elif title in opened:
            filed.append((intent, opened[title], None))
        else:
            opened[title] = stores.next_aggregate_id(len(opened))
            filed.append((intent, opened[title], title))

    matches: dict[int, str] = {}
    registered = stores.stores[topic].meta.tool_intents
    for aggregate in dict.fromkeys(f[1] for f in filed):
        if aggregate not in registered:
            continue
        places = [n for n, f in enumerate(filed) if f[1] == aggregate]
        members = registered[aggregate]
        got = _match_tool_intents(
            step, topic, [intents[n] for n in places], members, context, window, ask_match, stores
        )
        if isinstance(got, str):
            return _review(record, got)
        matches.update(zip(places, got, strict=True))

    resolved: list[UnitIntent] = []
    new = 0
    for n, (intent, aggregate, label) in enumerate(filed):
        found = intent.model_dump(exclude={"text"})
        matched = matches.get(n, NEW)
        if matched != NEW:
            if all(r.id != matched for r in resolved):
                resolved.append(UnitIntent(id=matched, aggregate=aggregate, **found))
            continue
        node_id = stores.next_tool_intent_id(new)
        new += 1
        resolved.append(
            UnitIntent(id=node_id, aggregate=aggregate, text=intent.text, label=label, **found)
        )
    return record.model_copy(update={"intents": resolved})


def _match_tool_intents(
    step: _Step,
    topic: str,
    intents: list[Intent],
    members: dict[str, str],
    context: _Context,
    window: _Round,
    ask: Ask,
    stores: Stores,
) -> list[str] | str:
    """Call L against one aggregate's members: an id or NEW per intent, or why it failed."""
    head = _graph_header(step, context, topic, stores, list(members))
    registered = "\n".join(f"{i}：{text}" for i, text in members.items())
    tail = f"{_listed(intents)}\n\n<已有的底层意图>\n{registered}\n</已有的底层意图>"
    try:
        prompt = _fitted_prompt(step, context, L_SYSTEM_PROMPT, head, window, tail)
    except WindowError as error:
        return str(error)
    allowed = {*members, NEW}

    def parse(reply: str) -> list[str]:
        got = [m.match for m in _LAnswer.model_validate_json(reply).answers]
        if len(got) != len(intents):
            raise ValueError(
                f"answers 有 {len(got)} 项，但 <本单元的意图> 有 {len(intents)} 个，"
                "要按顺序给每个意图回答一项。"
            )
        wrong = [m for m in got if m not in allowed]
        if wrong:
            raise ValueError(
                f"{wrong!r} 不在 candidates 里，match 只能是 candidates 里的编号或 NEW。"
            )
        return got

    answered, _, feedback = _ask_until_valid(ask, prompt, parse, CALL_ATTEMPTS)
    if answered is None:
        return f"call L gave no valid answer in {CALL_ATTEMPTS} attempts: {feedback}"
    return answered


def _match_intents(
    step: _Step,
    record: Assignment,
    topic: str,
    intents: list[Intent],
    context: _Context,
    window: _Round,
    ask: Ask,
    stores: Stores,
) -> Assignment:
    """Call B, then the ids: a registered intent keeps its id once per unit, others get new ones.

    Only what someone stated comes here; a tool call's intents go to call G. The
    candidates are the same actor's, because an AI statement matched to a user intent
    would leave no trace that the AI said it. With no candidates call B is skipped.
    """
    actor = _actor(step.unit)
    candidates = [
        node_id
        for node_id, node in stores.topic_nodes(topic).items()
        if isinstance(node, IntentNode) and node.actor == actor
    ]
    if not candidates:
        matches = [_Match(match=NEW, complete=True) for _ in intents]
    else:
        head = _graph_header(step, context, topic, stores, candidates)
        try:
            prompt = _fitted_prompt(step, context, B_SYSTEM_PROMPT, head, window, _listed(intents))
        except WindowError as error:
            return _review(record, str(error))
        allowed = {*candidates, NEW}

        def parse_b(reply: str) -> list[_Match]:
            got = _BAnswer.model_validate_json(reply).answers
            if len(got) != len(intents):
                raise ValueError(
                    f"answers 有 {len(got)} 项，但 <本单元的意图> 有 {len(intents)} 个，"
                    "要按顺序给每个意图回答一项。"
                )
            wrong = [m.match for m in got if m.match not in allowed]
            if wrong:
                raise ValueError(
                    f"{wrong!r} 不在 candidates 里，match 只能是 candidates 里的编号或 NEW。"
                )
            return got

        answered, _, feedback = _ask_until_valid(ask, prompt, parse_b, CALL_ATTEMPTS)
        if answered is None:
            return _review(
                record, f"call B gave no valid answer in {CALL_ATTEMPTS} attempts: {feedback}"
            )
        matches = answered
    resolved: list[UnitIntent] = []
    new = 0
    for intent, matched in zip(intents, matches, strict=True):
        found = intent.model_dump(exclude={"text"})
        if matched.match != NEW and matched.complete:
            if all(r.id != matched.match for r in resolved):
                resolved.append(UnitIntent(id=matched.match, **found))
            continue
        details_of = None if matched.match == NEW else matched.match
        node_id = stores.next_intent_id(new)
        new += 1
        resolved.append(UnitIntent(id=node_id, label=intent.text, details_of=details_of, **found))
    return record.model_copy(update={"intents": resolved})


def _actor(unit: Unit) -> Literal["user", "ai"]:
    """Who stated the intents of a unit on the intent path outside call G."""
    return unit.speech.speaker if isinstance(unit, SpeechSegment) else "user"


def _visible_pending(
    step: _Step, topic: str, context: _Context, stores: Stores
) -> dict[str, PendingObservation]:
    """The pending observations of `topic` the window shows, keyed by call C's handle.

    Only pushed observations wait here, because a tool result lives on its aggregate's node.
    One leaves when it falls outside the window's rounds.
    """
    earliest = window_start(context.items, _round_low(step), context.window_rounds)
    return {
        f"P{index}": pending
        for index, pending in stores.pending.items()
        if pending.topic == topic and earliest <= index < step.index
    }


def _unit_nodes(record: Assignment, topic: str, stores: Stores) -> dict[str, dict[str, str]]:
    """What call C connects for this unit: each intent or aggregate, once, with its text."""
    labels = {node_id: node.label for node_id, node in stores.topic_nodes(topic).items()}
    labels |= {i.aggregate or i.id: i.label for i in record.intents if i.label is not None}
    nodes: dict[str, dict[str, str]] = {}
    for intent in record.intents:
        node_id = intent.aggregate or intent.id
        nodes.setdefault(node_id, {"id": node_id, "text": labels[node_id]})
    return nodes


def _unit_source(unit: Unit) -> str:
    if isinstance(unit, SpeechSegment):
        return unit.speech.body[unit.segment.start : unit.segment.end]
    return unit.text


def _relate(
    step: _Step,
    record: Assignment,
    context: _Context,
    window: _Round,
    asks: Asks,
    stores: Stores,
) -> Assignment:
    """Call C: the edges of this unit's intents, and a node for each observation they name.

    A unit's intents may relate to each other: "A before B" often opens both at once. A
    tool call filed under an aggregate that already has a node is asked only for what
    this step adds (C_STEP_SYSTEM_PROMPT).
    """
    topic = record.topic
    assert topic is not None
    unit_nodes = _unit_nodes(record, topic, stores)
    known = stores.topic_nodes(topic)
    if isinstance(step.unit, ToolCall) and set(unit_nodes) <= set(known):
        system, ask = C_STEP_SYSTEM_PROMPT, asks.step_relation
    else:
        system, ask = C_SYSTEM_PROMPT, asks.relation
    relations = [
        {"relation": name, "from": start, "to": end, "meaning": meaning}
        for name, (start, end, meaning) in _C_RELATIONS.items()
    ]

    same_speech = set(unit_nodes)
    if isinstance(step.unit, SpeechSegment):
        uuid = step.unit.speech.source.uuid
        same_speech |= {
            i.id
            for span in stores.stores[topic].spans
            if span.source.uuid == uuid
            for i in span.intents
        }
    linkable = [i for i in known if i not in same_speech]
    pending = _visible_pending(step, topic, context, stores)
    offered = [{"id": handle, "text": p.text} for handle, p in pending.items()]
    more: dict[str, object] = {
        "relations": relations,
        "unit_intents": list(unit_nodes.values()),
    }
    head = _graph_header(step, context, topic, stores, [*linkable, *offered], more)
    try:
        prompt = _fitted_prompt(step, context, system, head, window)
    except WindowError as error:
        return _review(record, str(error))
    own_aggregate = {
        i: node.aggregate for i, node in known.items() if isinstance(node, AggregateObservationNode)
    }
    observations = {i for i, n in known.items() if isinstance(n, _OBSERVATION_NODES)} | set(pending)
    allowed = {*linkable, *pending}
    speakers = {i: n.actor for i, n in known.items() if isinstance(n, IntentNode)}
    if isinstance(step.unit, SpeechSegment):
        speakers |= {i: step.unit.speech.speaker for i in unit_nodes if i not in speakers}
    source = _unit_source(step.unit)
    shown_input = prompt[prompt.index("<此前的对话>") :]

    def parse(reply: str) -> list[_CEdge]:
        got = _CAnswer.model_validate_json(reply).edges
        for edge in got:
            ends = (edge.from_, edge.to)
            if len(set(ends) & same_speech) == 2:
                raise ValueError(
                    f"{ends!r} 出自同一次发言。同一次发言里的几个意图之间没有关系，去掉这条边。"
                )
            if not set(ends) & set(unit_nodes) or not set(ends) & allowed:
                raise ValueError(f"{ends!r} 必须一端在 unit_intents 里，另一端在 candidates 里。")
            start, end, _ = _C_RELATIONS[edge.relation]
            kinds = tuple("observation" if e in observations else "intent" for e in ends)
            if kinds != (start, end):
                raise ValueError(f"{edge.relation} 的方向必须是 {start} -> {end}，{ends!r} 不是。")
            if edge.relation == "triggers" and edge.to == own_aggregate.get(edge.from_):
                raise ValueError(
                    f"{edge.from_} 是 {edge.to} 自己的工具调用产生的结果。同一个聚合内部，结果引出下一步操作属于这个聚合本身的推进，去掉这条 triggers 边。"
                )
            if edge.relation == "responds" and {speakers.get(e) for e in ends} != {"user", "ai"}:
                raise ValueError(
                    f"responds 只连用户和 AI 之间相互回应的话：{ends!r} 必须一端是用户的意图，另一端是 AI 的意图。"
                )
            if not is_excerpt(edge.evidence, source) and not is_excerpt(edge.evidence, shown_input):
                raise ValueError(f"evidence {edge.evidence!r} 不是当前单元或此前对话的原文摘录。")
        return got

    answered, _, feedback = _ask_until_valid(ask, prompt, parse, CALL_ATTEMPTS)
    if answered is None:
        return _review(
            record, f"call C gave no valid answer in {CALL_ATTEMPTS} attempts: {feedback}"
        )
    named: dict[str, str] = {}
    built: list[NewObservation] = []
    for handle in dict.fromkeys(e.from_ for e in answered if e.from_ in pending):
        named[handle] = stores.next_observation_id(len(built))
        built.append(
            NewObservation(
                id=named[handle],
                item=int(handle[1:]),
                label=pending[handle].text,
                source=pending[handle].source,
            )
        )
    edges = [
        Edge.model_validate(
            {
                "from": named.get(e.from_, e.from_),
                "to": e.to,
                "relation": e.relation,
                "evidence": e.evidence,
            }
        )
        for e in answered
    ]
    return record.model_copy(update={"edges": edges, "observations": built})


def _parse_outcomes(reply: str, candidates: Sequence[str], needs_summary: bool) -> _DAnswer:
    answer = _DAnswer.model_validate_json(reply)
    wrong = [o.target for o in answer.outcomes if o.target not in candidates]
    if wrong:
        raise ValueError(f"{wrong!r} 不在 candidates 里。")
    if needs_summary and not answer.summary.strip():
        raise ValueError("summary 不能为空。")
    return answer


def _outcome_edges(
    node_id: str, answer: _DAnswer, evidence: str, earlier: dict[tuple[str, str], str]
) -> list[Edge]:
    """Edges of the answer; one `earlier` already holds keeps the evidence it was first given."""
    return [
        Edge.model_validate(
            {
                "from": node_id,
                "to": o.target,
                "relation": o.relation,
                "evidence": earlier.get((o.target, o.relation), evidence),
            }
        )
        for o in answer.outcomes
    ]


def _observe(
    step: _Step,
    unit: Observation,
    record: Assignment,
    context: _Context,
    window: _Round,
    asks: Asks,
    stores: Stores,
) -> Assignment:
    """Call D: which intents the observation shows done or blocked, and where its text goes.

    A tool result updates its aggregates' observation nodes, whatever the outcomes. A
    pushed observation gets a node only with outcomes, else it stays pending. One that
    cannot be judged still gets a needs_review node, so it is not silently lost.
    """
    source = ObservationSource(uuid=unit.source.uuid, tool_use_id=unit.tool_use_id)
    known_text = (
        " ".join(context.digests.results[unit.tool_use_id].summaries)
        if unit.tool_use_id is not None
        else unit.text
    )
    aggregates = stores.aggregates_of(unit.tool_use_id) if unit.tool_use_id is not None else []
    if record.topic is not None and aggregates:
        record = record.model_copy(update={"tool_result": known_text})

    def failed(reason: str) -> Assignment:
        built = NewObservation(
            id=stores.next_observation_id(),
            item=step.index,
            label=known_text,
            source=source,
            needs_review=True,
        )
        return _review(record, reason).model_copy(update={"observations": [built]})

    topic = record.topic
    if topic is None:
        return failed(record.review_reason or "The tool call that produced this has no topic.")
    if record.new_title is not None:
        stores.open_topic(topic)
    candidates = [
        i for i, n in stores.topic_nodes(topic).items() if isinstance(n, IntentNode | AggregateNode)
    ]

    def judge(
        system: str, ask: Ask, needs_summary: bool, more: dict[str, object]
    ) -> _DAnswer | str:
        """The answer of call D, or why there is none."""
        try:
            head = _graph_header(
                step,
                context,
                topic,
                stores,
                candidates,
                {"relations": ["satisfies", "blocks"], **more},
            )
            prompt = _fitted_prompt(step, context, system, head, window)
        except WindowError as error:
            return str(error)
        answered, _, feedback = _ask_until_valid(
            ask, prompt, lambda r: _parse_outcomes(r, candidates, needs_summary), CALL_ATTEMPTS
        )
        if answered is None:
            return f"call D gave no valid answer in {CALL_ATTEMPTS} attempts: {feedback}"
        return answered

    if unit.tool_use_id is None:
        judged = judge(D_SUMMARY_SYSTEM_PROMPT, asks.outcome_summary, True, {})
        if isinstance(judged, str):
            return failed(judged)
        text = judged.summary.strip()
        if not judged.outcomes:
            return record.model_copy(update={"pending_text": text})
        node_id = stores.next_observation_id()
        built = NewObservation(id=node_id, item=step.index, label=text, source=source)
        edges = _outcome_edges(node_id, judged, unit.source.uuid, {})
        return record.model_copy(update={"edges": edges, "observations": [built]})

    if not aggregates:
        return failed("The tool call that produced this has no aggregate to file it under.")
    updated: list[NewAggregateObservation] = []
    all_edges: list[Edge] = []
    opened = 0
    for aggregate in aggregates:
        existing = stores.aggregate_observation(topic, aggregate)
        if existing is None:
            judged = judge(D_SYSTEM_PROMPT, asks.outcome, False, {"aggregate": aggregate})
        else:
            more: dict[str, object] = {
                "aggregate": aggregate,
                "aggregate_observation": stores.topic_nodes(topic)[existing].label,
            }
            judged = judge(D_UPDATE_SYSTEM_PROMPT, asks.outcome_update, True, more)
        if isinstance(judged, str):
            return failed(judged)
        label = known_text if existing is None else judged.summary.strip()
        node_id = existing or stores.next_observation_id(opened)
        opened += existing is None
        updated.append(NewAggregateObservation(id=node_id, aggregate=aggregate, label=label))
        earlier: dict[tuple[str, str], str] = {
            (e.to, e.relation): e.evidence
            for e in stores.stores[topic].meta.graph.edges
            if e.from_ == node_id
        }
        all_edges.extend(_outcome_edges(node_id, judged, unit.tool_use_id, earlier))
    return record.model_copy(update={"edges": all_edges, "aggregate_observations": updated})


def _review(record: Assignment, reason: str) -> Assignment:
    return record.model_copy(update={"needs_review": True, "review_reason": reason})


def into_stores(stores: Stores, step: _Step, record: Assignment) -> None:
    """Register the record's new intents and put the unit's span in its store.

    An observation has no span: it goes on a node, or waits as a triggers candidate.
    """
    unit, topic = step.unit, record.topic
    if topic is not None and topic not in stores.stores:
        stores.open_topic(topic)
    if isinstance(unit, Observation):
        _add_observations(stores, topic, record)
        if record.tool_result is not None:
            assert topic is not None and unit.tool_use_id is not None
            stores.add_tool_pair(topic, unit.tool_use_id, record.tool_result)
        for built in record.aggregate_observations:
            assert topic is not None
            stores.set_aggregate_observation(topic, built.id, built.aggregate, built.label)
        stores.add_edges(topic, record.edges)
        if record.pending_text is not None:
            assert topic is not None
            source = ObservationSource(uuid=unit.source.uuid, tool_use_id=unit.tool_use_id)
            stores.pending[step.index] = PendingObservation(
                topic=topic, text=record.pending_text, source=source
            )
        return
    aggregates: set[str] = set()
    for intent in record.intents:
        assert topic is not None
        if intent.aggregate is None:
            if intent.label is not None:
                stores.register_intent(topic, intent.id, intent.label, _actor(unit))
            continue
        aggregates.add(intent.aggregate)
        if intent.label is not None:
            stores.register_aggregate(topic, intent.aggregate, intent.label)
        if intent.text is not None:
            stores.register_tool_intent(topic, intent.aggregate, intent.id, intent.text)
    spans = [
        SpanIntent(id=i.id, evidence=i.evidence, explicit=i.explicit, note=i.note or None)
        for i in record.intents
    ]
    if isinstance(unit, SpeechSegment):
        segment = unit.segment
        review = record.needs_review or segment.needs_review
        span = stores.add_speech_text(
            topic, unit.speech, (segment.start, segment.end), spans, review
        )
    elif isinstance(unit, UserSignal):
        span = stores.add_user_signal(topic, unit, spans, record.needs_review)
    else:
        span = stores.add_tool_call(
            topic, unit, step.index, spans, frozenset(aggregates), record.needs_review
        )
    _add_observations(stores, topic, record)
    details = [
        Edge.model_validate(
            {"from": i.id, "to": i.details_of, "relation": "serves", "evidence": i.evidence}
        )
        for i in record.intents
        if i.details_of is not None
    ]
    stores.add_edges(
        topic, [e.model_copy(update={"span": span}) for e in [*details, *record.edges]]
    )


def _add_observations(stores: Stores, topic: str | None, record: Assignment) -> None:
    for built in record.observations:
        stores.add_observation(
            topic, built.id, built.label, built.source, review=built.needs_review
        )
        stores.pending.pop(built.item, None)


def _graph_header(
    step: _Step,
    context: _Context,
    topic: str,
    stores: Stores,
    candidates: Sequence[object] | None = None,
    more: dict[str, object] | None = None,
) -> str:
    """The header of calls after T: the current topic and its whole graph."""
    graph = graph_with_status(stores.stores[topic].meta.graph)
    extra: dict[str, object] = {"current_topic": topic, "graph": graph}
    if candidates is not None:
        extra["candidates"] = candidates
    extra.update(more or {})
    return _header(step.unit, context.topics, extra)


def _prompt(
    step: _Step,
    context: _Context,
    system: str,
    head: str,
    in_force: Sequence[ToolCompression],
    tail: str,
    rounds: int,
) -> str:
    """The prompt of one call for `step`; raises WindowError when the window does not fit.

    The window ends right before this unit, so the unit sees what led to it but not how
    its round ended. `tail` follows the current unit; `system` only counts against the
    budget.
    """
    items, digests, count = context.items, context.digests, context.count
    current = f"<当前单元>\n{render_unit(step.unit, digests)}\n</当前单元>"
    if tail:
        current = f"{current}\n\n{tail}"
    room = PROMPT_BUDGET - count(system) - count(head) - count(current) - FEEDBACK_ROOM
    window = window_text(
        items, _round_low(step), step.index, digests, in_force, room, count, rounds
    )
    return f"{head}\n\n<此前的对话>\n{window}\n</此前的对话>\n\n{current}"


def _header(unit: Unit, topics: Sequence[tuple[str, str]], extra: dict[str, object]) -> str:
    ref = unit_ref(unit)
    source = unit.speech.source if isinstance(unit, SpeechSegment) else unit.source
    actor = {
        "user_segment": "user",
        "ai_segment": "ai",
        "user_signal": "user",
        "tool_call": "ai",
    }.get(ref.kind, "environment")
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
        **{key: value for key, value in extra.items() if value is not None},
    }
    return header_block(header)


def render_unit(unit: Unit, digests: Digests) -> str:
    """The unit as the <当前单元> block shows it; a speech segment is marked inside its speech."""
    if not isinstance(unit, SpeechSegment):
        return round_entry(unit, digests)
    body, segment = unit.speech.body, unit.segment
    marked = (
        f"{body[: segment.start]}<当前段落>{body[segment.start : segment.end]}"
        f"</当前段落>{body[segment.end :]}"
    )
    images = "".join(f"\n[图片] {path}" for path in unit.speech.images)
    return f"【{SPEAKER_NAMES[unit.speech.speaker]}】\n{marked}{images}"


def write_topic_files(out: Path, topics: Sequence[TopicEntry], stores: Stores) -> None:
    """Rewrite topics.yaml and every store file, the unassigned one included."""
    listing = {"topics": [topic.model_dump() for topic in topics]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "topics.yaml").write_text(
        STORE_HEADER + yaml.safe_dump(listing, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_stores(out / "topics", stores, {topic.id: Path(topic.file) for topic in topics})


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
    """The segments file, by speech uuid; a speech can span lines, or have none yet."""
    segments: dict[str, list[Segment]] = {}
    if not path.exists():
        return segments
    for line in path.read_text(encoding="utf-8").splitlines():
        message = MessageSegments.model_validate_json(line)
        segments.setdefault(message.uuid, []).extend(message.segments)
    return segments


def log_lines(assignment: Assignment) -> list[str]:
    """The run.log lines for `assignment`; empty when it is not worth a human's look."""
    unit = assignment.unit
    where = f"{unit.kind} {unit.uuid}" + (f" chars {list(unit.chars)}" if unit.chars else "")
    if unit.tool_use_id:
        where += f" tool_use_id {unit.tool_use_id}"
    lines = []
    if assignment.new_title is not None:
        lines.append(f"new topic {assignment.topic}: {assignment.new_title} (opened by {where})")
    if assignment.placeholder:
        lines.append(f"whole speech has no intent, placeholder span: {where}")
    if assignment.review_reason is not None:
        place = assignment.topic or "unassigned"
        lines.append(f"needs_review in {place}: {where}: {assignment.review_reason}")
    return lines


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
    parser.add_argument("--window-rounds", type=window_rounds, default=DEFAULT_WINDOW_ROUNDS)
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
    compressions_path = args.out / "tool_compressions.jsonl"
    calls = CallLog(path=args.out / CALLS_FILE)

    def ask(name: str, system: str, schema: dict[str, Any]) -> Ask:
        return asker(
            args.backend, args.model, args.effort, system, schema, args.stream, calls, name
        )

    asks = Asks(
        segment=ask("segment", SEGMENT_SYSTEM_PROMPT, SEGMENT_SCHEMA),
        topic=ask("topic", T_SYSTEM_PROMPT, T_SCHEMA),
        signal=ask("signal", A_SYSTEM_PROMPT, A_SCHEMA),
        registry=ask("registry", B_SYSTEM_PROMPT, B_SCHEMA),
        aggregate=ask("aggregate", G_SYSTEM_PROMPT, G_SCHEMA),
        tool_intent=ask("tool_intent", L_SYSTEM_PROMPT, L_SCHEMA),
        relation=ask("relation", C_SYSTEM_PROMPT, C_SCHEMA),
        step_relation=ask("step_relation", C_STEP_SYSTEM_PROMPT, C_SCHEMA),
        outcome=ask("outcome", D_SYSTEM_PROMPT, D_SCHEMA),
        outcome_summary=ask("outcome_summary", D_SUMMARY_SYSTEM_PROMPT, D_SUMMARY_SCHEMA),
        outcome_update=ask("outcome_update", D_UPDATE_SYSTEM_PROMPT, D_SUMMARY_SCHEMA),
        round=ask("round", ROUND_SYSTEM_PROMPT, SUMMARY_SCHEMA),
    )
    stores = Stores(merged_file=str(args.merged), items=items)
    results = extract_units(
        items,
        load_segments(args.segments),
        load_digests(args.digests),
        done,
        load_compressions(compressions_path),
        asks,
        qwen_tokens,
        stores,
        args.window_rounds,
    )
    position = len(done)
    with (
        assignments_path.open("a", encoding="utf-8") as out,
        compressions_path.open("a", encoding="utf-8") as compressions,
        (args.out / "run.log").open("a", encoding="utf-8") as log,
        args.segments.open("a", encoding="utf-8") as segment_file,
    ):
        for assignment in results:
            if isinstance(assignment, MessageSegments):
                segment_file.write(assignment.model_dump_json() + "\n")
                segment_file.flush()
                continue
            if isinstance(assignment, ToolCompression):
                compressions.write(assignment.model_dump_json() + "\n")
                compressions.flush()
                print(
                    f"compressed the tool records of items {assignment.begin}-"
                    f"{assignment.through - 1}",
                    flush=True,
                )
                continue
            out.write(assignment.model_dump_json() + "\n")
            out.flush()
            if assignment.new_title is not None:
                topics = [*topics, *topics_from([assignment], args.out)]
            write_topic_files(args.out, topics, stores)
            for line in log_lines(assignment):
                log.write(line + "\n")
                log.flush()
                print(line, file=sys.stderr, flush=True)
            print(f"{position}: {assignment.unit.kind} -> {assignment.topic}", flush=True)
            position += 1


if __name__ == "__main__":
    main()
