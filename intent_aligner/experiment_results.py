"""Durable experiment replies, with optional usage enrichment and offline recovery.

Receipts are written before transcript parsing. A crashed collector can rebuild its
result file without calling the model again. Final rows take precedence over receipts.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intent_aligner.replay import TokenUse, _tokens, read_exchange


class ExperimentResult(BaseModel):
    """The experiment identifies a repeat before making its paid endpoint call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: str
    run: int
    model: str
    system: str
    transcript: str | None = None
    answer: str | None = None
    error: str | None = None
    tokens: TokenUse | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata_error: str | None = None


def _enrich(row: ExperimentResult) -> ExperimentResult:
    if row.transcript is None or row.answer is None:
        return row
    try:
        exchange = read_exchange(Path(row.transcript))
        if exchange is None:
            raise ValueError(f"Unsupported transcript: {row.transcript}")
        return row.model_copy(
            update={"tokens": _tokens(exchange), "usage": exchange.usage, "metadata_error": None}
        )
    except (
        Exception
    ) as error:  # Metadata must never discard a paid reply already saved in a receipt.
        return row.model_copy(update={"metadata_error": f"{type(error).__name__}: {error}"})


def record_reply(results: Path, row: ExperimentResult) -> ExperimentResult:
    """Persist the raw answer before optional statistics; safe for concurrent workers."""
    receipts = results.with_suffix(".receipts.jsonl")
    with receipts.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(row.model_dump_json(exclude_none=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _enrich(row)


def refresh_results(results: Path) -> int:
    """Refresh usage and restore receipts while the experiment's collector is stopped.

    No endpoint calls occur here. Original answers and the order of final rows are
    preserved; an existing successful repeat is not replaced by another receipt.
    """
    rows = (
        [ExperimentResult.model_validate_json(line) for line in results.read_text().splitlines()]
        if results.exists()
        else []
    )
    successful = {(r.unit, r.run) for r in rows if r.answer is not None and r.error is None}
    receipts = results.with_suffix(".receipts.jsonl")
    if receipts.exists():
        for line in receipts.read_text().splitlines():
            row = ExperimentResult.model_validate_json(line)
            key = (row.unit, row.run)
            if key not in successful:
                rows.append(row)
                if row.answer is not None and row.error is None:
                    successful.add(key)
    if not rows:
        return 0
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=results.parent, prefix=f".{results.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            if results.exists():
                temporary.chmod(results.stat().st_mode & 0o777)
            for row in rows:
                handle.write(_enrich(row).model_dump_json(exclude_none=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(results)
        finally:
            temporary.unlink(missing_ok=True)
    return len(rows)
