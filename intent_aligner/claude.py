"""Claude Code session discovery and fork grouping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import attrs

from intent_aligner.jsonl import Record, read_jsonl

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class ClaudeSession:
    """A transcript plus its optional sidecar directory (subagents/, tool-results/)."""

    session_id: str
    transcript: Path
    sidecar_dir: Path | None


def session_at(transcript: Path) -> ClaudeSession:
    """Describe the session stored in `transcript`."""
    sidecar = transcript.with_suffix("")
    return ClaudeSession(
        session_id=transcript.stem,
        transcript=transcript,
        sidecar_dir=sidecar if sidecar.is_dir() else None,
    )


def find_session(prefix: str, projects: Path) -> ClaudeSession:
    """Resolve a session id prefix across all project directories."""
    matches = sorted(projects.glob(f"*/{prefix}*.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"Claude session prefix {prefix!r} matched {len(matches)} transcripts under "
            f"{projects}: {[str(m) for m in matches]}. Use a longer prefix in sessions.yaml."
        )
    return session_at(matches[0])


def group_forks(seeds: Iterable[ClaudeSession], projects: Path) -> list[list[ClaudeSession]]:
    """Group seeds with every transcript they share forked messages with.

    A fork copies the parent's recent messages under their original uuids, so the
    child's first message uuid also appears in the parent transcript. Relatives
    outside the seed list are pulled in until the closure is complete. Each group is
    ordered by first timestamp, which puts parents before children.
    """
    transcripts = sorted(projects.glob("*/*.jsonl"))
    known = {s.session_id: s for s in seeds}
    parent_of: dict[str, str] = {sid: sid for sid in known}
    pending = list(known.values())

    while pending:
        needles = {
            f'"uuid":"{_first_message_uuid(s.transcript)}"'.encode(): s.session_id for s in pending
        }
        pending = []
        for path in transcripts:
            data = path.read_bytes()
            for needle, child_id in needles.items():
                if path.stem == child_id or needle not in data:
                    continue
                if path.stem not in known:
                    known[path.stem] = session_at(path)
                    parent_of[path.stem] = path.stem
                    pending.append(known[path.stem])
                _union(parent_of, path.stem, child_id)

    groups: dict[str, list[ClaudeSession]] = {}
    for sid, session in known.items():
        groups.setdefault(_root(parent_of, sid), []).append(session)
    return [sorted(g, key=lambda s: _first_timestamp(s.transcript)) for g in groups.values()]


def copy_key(record: dict[str, Any]) -> str | None:
    """Identity of a record that a fork may have copied from its parent.

    Copies keep the uuid but differ in bookkeeping (sessionKind, zeroed usage), so
    uuid is the identity. Uuid-less session-state rows (ai-title, mode, ...) recur
    legitimately in every session and are never treated as copies; only verbatim
    file-history snapshots are.
    """
    if "uuid" in record:
        return f"uuid:{record['uuid']}"
    if record.get("type") == "file-history-snapshot":
        rest = {k: v for k, v in record.items() if k != "sessionId"}
        digest = hashlib.sha256(json.dumps(rest, sort_keys=True).encode()).hexdigest()
        return f"snapshot:{digest}"
    return None


def _first_message_uuid(transcript: Path) -> str:
    for item in read_jsonl(transcript):
        if isinstance(item, Record) and item.data.get("type") in ("user", "assistant"):
            return str(item.data["uuid"])
    raise ValueError(f"No user/assistant message in {transcript}; cannot detect forks of it.")


def _first_timestamp(transcript: Path) -> str:
    for item in read_jsonl(transcript):
        if isinstance(item, Record) and "timestamp" in item.data:
            return str(item.data["timestamp"])
    raise ValueError(f"No timestamped record in {transcript}.")


def _root(parent_of: dict[str, str], sid: str) -> str:
    while parent_of[sid] != sid:
        sid = parent_of[sid]
    return sid


def _union(parent_of: dict[str, str], a: str, b: str) -> None:
    parent_of[_root(parent_of, a)] = _root(parent_of, b)
