"""Invariants of the built corpus; run `python -m intent_aligner.corpus` first."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from intent_aligner.corpus import TOOL_ROOT, CorpusIndex, MergedConversation
from intent_aligner.jsonl import Record, read_jsonl

CORPUS = TOOL_ROOT / "corpus"
INDEX = CORPUS / "index.json"

pytestmark = pytest.mark.skipif(not INDEX.exists(), reason="corpus not built")


def _conversations() -> list[MergedConversation]:
    return CorpusIndex.model_validate_json(INDEX.read_text()).conversations


def _provenance(conv: MergedConversation) -> list[dict[str, object]]:
    rows = []
    with (TOOL_ROOT / conv.merged_file).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line, strict=False)
            rows.append({"uuid": record.get("uuid"), **record["_merge"]})
    return rows


@pytest.fixture(
    scope="module",
    params=_conversations() if INDEX.exists() else [],
    ids=lambda conv: conv.merged_file,
)
def merged(request: pytest.FixtureRequest) -> tuple[MergedConversation, list[dict[str, object]]]:
    conv: MergedConversation = request.param
    return conv, _provenance(conv)


def test_each_source_keeps_its_line_order(
    merged: tuple[MergedConversation, list[dict[str, object]]],
) -> None:
    conv, rows = merged
    last: dict[object, int] = {}
    for row in rows:
        line = int(str(row["line"]))
        assert line > last.get(row["file"], -1), f"{conv.merged_file}: {row} out of order"
        last[row["file"]] = line


def test_records_are_accounted_for(
    merged: tuple[MergedConversation, list[dict[str, object]]],
) -> None:
    conv, rows = merged
    files = sorted({str(r["file"]) for r in rows})
    parsed = sum(isinstance(item, Record) for f in files for item in read_jsonl(TOOL_ROOT / f))
    got = (len(rows), conv.records, parsed - conv.copies_dropped)
    assert got[0] == got[1] == got[2], f"{conv.merged_file}: written/indexed/expected {got}"


def test_claude_uuids_are_unique(
    merged: tuple[MergedConversation, list[dict[str, object]]],
) -> None:
    conv, rows = merged
    if conv.platform != "claude":
        pytest.skip("Codex rollouts carry no uuid")
    dupes = [u for u, n in Counter(r["uuid"] for r in rows if r["uuid"]).items() if n > 1]
    assert not dupes, f"{conv.merged_file}: {len(dupes)} duplicated uuids, e.g. {dupes[:3]}"


def test_fork_copies_resolve_to_the_parent(
    merged: tuple[MergedConversation, list[dict[str, object]]],
) -> None:
    conv, rows = merged
    if conv.platform != "claude" or len(conv.members) == 1:
        pytest.skip("no fork in this conversation")
    rank = {sid: i for i, sid in enumerate(conv.members)}
    owner: dict[object, int] = {}
    for sid in conv.members:
        for item in read_jsonl(CORPUS / "raw" / "claude" / f"{sid}.jsonl"):
            if isinstance(item, Record) and "uuid" in item.data:
                owner.setdefault(item.data["uuid"], rank[sid])
    stolen = [r for r in rows if r["uuid"] and owner[r["uuid"]] < rank[str(r["session"])]]
    assert not stolen, (
        f"{conv.merged_file}: {len(stolen)} copies kept instead of originals, e.g. {stolen[:2]}"
    )
