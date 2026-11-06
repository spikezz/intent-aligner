"""Merge the files of one conversation into a single time-ordered JSONL."""

from __future__ import annotations

import heapq
import json
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import attrs

from intent_aligner.jsonl import DamagedLine, Record, read_jsonl

CopyKey = Callable[[dict[str, Any]], str | None]


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Source:
    """One input file; `label` is the provenance path written into merged records."""

    session_id: str
    path: Path
    label: str


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class SourceDamage:
    """A damaged line, tagged with the provenance label of its file."""

    label: str
    damaged: DamagedLine


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class MergeResult:
    """`records` counts written lines; fork copies and unrecoverable damage are not in it."""

    records: int
    copies_dropped: int
    salvaged: int
    damage: tuple[SourceDamage, ...]


def merge(sources: Sequence[Source], out: Path, copy_key: CopyKey | None) -> MergeResult:
    """Write all records of `sources` to `out`, ordered by timestamp.

    Each file keeps its internal order, and on equal timestamps earlier sources win,
    so `sources` must list parents before children. A record whose copy_key occurs
    in an *earlier* source is a fork copy and is dropped; repeats inside one file are
    genuine and kept. Every record gains a `_merge` provenance field:
    {session, file, line, salvaged}.
    """
    # Decided from a pre-pass, not from emission order: a parent's original can sort
    # after the child's copy when the parent's timestamps step back.
    keys_by_source = [_copy_keys(s, copy_key) for s in sources] if copy_key else []
    damage: list[SourceDamage] = []
    records = copies_dropped = salvaged = 0

    streams = [_keyed(i, s, damage) for i, s in enumerate(sources)]
    with out.open("w", encoding="utf-8") as f:
        for _, index, record in heapq.merge(*streams, key=lambda item: (item[0], item[1])):
            key = copy_key(record.data) if copy_key else None
            if key is not None and any(key in keys for keys in keys_by_source[:index]):
                copies_dropped += 1
                continue
            source = sources[index]
            provenance = {
                "session": source.session_id,
                "file": source.label,
                "line": record.line,
                "salvaged": record.salvaged,
            }
            f.write(json.dumps({**record.data, "_merge": provenance}, ensure_ascii=False))
            f.write("\n")
            records += 1
            salvaged += record.salvaged

    return MergeResult(
        records=records, copies_dropped=copies_dropped, salvaged=salvaged, damage=tuple(damage)
    )


def _copy_keys(source: Source, copy_key: CopyKey) -> set[str]:
    keys = (copy_key(item.data) for item in read_jsonl(source.path) if isinstance(item, Record))
    return {k for k in keys if k is not None}


def _keyed(
    index: int, source: Source, damage: list[SourceDamage]
) -> Iterator[tuple[datetime, int, Record]]:
    """Attach a monotonic sort key to each record of one file.

    The key is the running maximum timestamp, because heapq.merge needs sorted
    inputs and transcripts occasionally step back in time. Untimestamped rows
    (session-state lines) take the key of the next timestamped record so that a
    child's header rows do not jump ahead of its parent's history.
    """
    current: datetime | None = None
    waiting: list[Record] = []
    for item in read_jsonl(source.path):
        if isinstance(item, DamagedLine):
            damage.append(SourceDamage(label=source.label, damaged=item))
            continue
        stamp = item.data.get("timestamp")
        if not isinstance(stamp, str):
            if current is None:
                waiting.append(item)
            else:
                yield current, index, item
            continue
        parsed = datetime.fromisoformat(stamp)
        current = parsed if current is None else max(current, parsed)
        for held in waiting:
            yield current, index, held
        waiting.clear()
        yield current, index, item
    if waiting:
        raise ValueError(f"{source.path} has no timestamped record; cannot place it in time.")
