"""Tolerant JSONL reading for agent transcripts that contain torn writes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import attrs

_HEAD_CHARS = 300


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class Record:
    """A parsed line; `salvaged` marks a record recovered from behind a torn write."""

    line: int
    data: dict[str, Any]
    salvaged: bool


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class DamagedLine:
    """Leading characters of a line from which no record could be recovered."""

    line: int
    chars: int
    head: str


def read_jsonl(path: Path) -> Iterator[Record | DamagedLine]:
    """Yield one item per non-empty line (0-based line numbers).

    When the disk fills up, both CLIs leave a truncated record and append the next
    record to the same line (incident 2026-10-21). The trailing complete record is
    salvaged; the truncated prefix is reported as a DamagedLine before it.
    """
    with path.open("rb") as f:
        for line_no, raw in enumerate(f):
            raw = raw.rstrip(b"\r\n")
            if not raw:
                continue
            try:
                yield Record(line=line_no, data=_loads(raw.decode("utf-8")), salvaged=False)
            except ValueError:  # covers UnicodeDecodeError and JSONDecodeError
                yield from _salvage(line_no, raw.decode("utf-8", errors="replace"))


def _loads(text: str) -> dict[str, Any]:
    # strict=False: Codex writes raw control characters inside strings.
    obj = json.loads(text, strict=False)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    # JSON object keys are always strings.
    return cast(dict[str, Any], obj)


def _salvage(line_no: int, text: str) -> Iterator[Record | DamagedLine]:
    start = text.find('{"', 1)
    while start != -1:
        try:
            data = _loads(text[start:])
        except ValueError:
            start = text.find('{"', start + 1)
            continue
        yield DamagedLine(line=line_no, chars=start, head=text[:_HEAD_CHARS])
        yield Record(line=line_no, data=data, salvaged=True)
        return
    yield DamagedLine(line=line_no, chars=len(text), head=text[:_HEAD_CHARS])
