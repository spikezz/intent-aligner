"""Codex rollout discovery: continuation segments, fork chains and subagent logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import attrs

from intent_aligner.jsonl import Record, read_jsonl

CODEX_SESSIONS = Path.home() / ".codex" / "sessions"

# rollout-<local start>-<thread id>[_<segment id>].jsonl; a thread is split into a
# new segment file every time it is resumed.
_ROLLOUT = re.compile(
    r"rollout-\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d-(?P<thread>[0-9a-f-]{36})(?:_[0-9a-f-]{36})?\.jsonl"
)


@attrs.frozen(slots=True, auto_attribs=True, kw_only=True)
class CodexThread:
    """A thread's segments in chronological order, plus its subagent rollouts."""

    thread_id: str
    segments: tuple[Path, ...]
    forked_from: str | None
    subagents: tuple[Path, ...]


def index_threads(sessions_root: Path) -> dict[str, CodexThread]:
    """Index every rollout under `sessions_root` by thread id."""
    segments: dict[str, list[Path]] = {}
    subagents: dict[str, list[Path]] = {}
    for path in sessions_root.rglob("rollout-*.jsonl"):
        match = _ROLLOUT.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized rollout file name {path}; update _ROLLOUT.")
        parent = _session_meta(path).get("parent_thread_id")
        if parent:
            subagents.setdefault(str(parent), []).append(path)
        else:
            segments.setdefault(match["thread"], []).append(path)

    threads = {}
    for thread_id, paths in segments.items():
        # File names start with the segment's start time, so name order is time order.
        ordered = tuple(sorted(paths, key=lambda p: p.name))
        forked_from = _session_meta(ordered[0]).get("forked_from_id")
        threads[thread_id] = CodexThread(
            thread_id=thread_id,
            segments=ordered,
            forked_from=str(forked_from) if forked_from else None,
            subagents=tuple(sorted(subagents.get(thread_id, []), key=lambda p: p.name)),
        )
    return threads


def resolve_prefix(prefix: str, threads: dict[str, CodexThread]) -> CodexThread:
    """Resolve a thread id prefix; subagent rollouts are not candidates."""
    matches = [t for tid, t in threads.items() if tid.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"Codex thread prefix {prefix!r} matched {[t.thread_id for t in matches]}. "
            "Use a longer prefix in sessions.yaml."
        )
    return matches[0]


def fork_chain(thread: CodexThread, threads: dict[str, CodexThread]) -> list[CodexThread]:
    """Ancestors first, ending with `thread`.

    A fork child does not copy its parent's history; it only references it via
    forked_from_id, so the ancestors are required to see the full conversation.
    """
    chain = [thread]
    while chain[0].forked_from:
        parent_id = chain[0].forked_from
        if parent_id not in threads:
            raise ValueError(
                f"Thread {chain[0].thread_id} forks from {parent_id}, which has no rollout "
                f"under {CODEX_SESSIONS}. Was it archived or deleted?"
            )
        chain.insert(0, threads[parent_id])
    return chain


def _session_meta(path: Path) -> dict[str, Any]:
    first = next(read_jsonl(path), None)
    if not isinstance(first, Record) or first.data.get("type") != "session_meta":
        raise ValueError(f"{path} does not start with a session_meta record.")
    payload: dict[str, Any] = first.data["payload"]
    return payload
