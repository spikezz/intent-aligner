"""Build the corpus: mirror raw transcripts and merge each conversation into one file.

Every path written into index.json and into merged records is relative to TOOL_ROOT.

Usage: python -m intent_aligner.corpus [--seeds sessions.yaml] [--corpus corpus/]

Layout written under the corpus directory, which must lie inside TOOL_ROOT:
    raw/claude/<session>.jsonl, raw/claude/<session>/     verbatim copies
    raw/codex/<thread>/rollout-*.jsonl, raw/codex/<thread>/subagents/   verbatim copies
    merged/<platform>-<seed>.jsonl                         one file per conversation
    index.json                                             CorpusIndex
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from intent_aligner import claude, codex
from intent_aligner.merge import CopyKey, MergeResult, Source, merge
from intent_aligner.seeds import Platform, SessionSeed, load_seeds

TOOL_ROOT = Path(__file__).resolve().parent.parent


class DamagedLineReport(BaseModel):
    """A line whose leading `chars` characters were lost to a torn write."""

    model_config = ConfigDict(frozen=True)

    file: str
    line: int
    chars: int
    head: str


class MergedConversation(BaseModel):
    """One merged file; `members` lists the merged session/thread ids, parents first."""

    model_config = ConfigDict(frozen=True)

    platform: Platform
    merged_file: str
    seeds: list[str]
    members: list[str]
    records: int
    copies_dropped: int
    salvaged: int
    damaged: list[DamagedLineReport]


class CorpusIndex(BaseModel):
    """Contents of corpus/index.json; conversations follow sessions.yaml order.

    `merged_file` and `DamagedLineReport.file` are relative to TOOL_ROOT.
    """

    model_config = ConfigDict(frozen=True)

    conversations: list[MergedConversation]


def build(seeds: Sequence[SessionSeed], corpus: Path) -> CorpusIndex:
    """Mirror and merge every conversation reachable from `seeds`."""
    if not corpus.resolve().is_relative_to(TOOL_ROOT):
        raise ValueError(
            f"Corpus directory {corpus} is outside {TOOL_ROOT}; recorded paths are relative "
            "to the tool root, so the corpus must live inside it."
        )
    (corpus / "merged").mkdir(parents=True, exist_ok=True)
    conversations = [
        *_build_claude([s for s in seeds if s.platform == "claude"], corpus),
        *_build_codex([s for s in seeds if s.platform == "codex"], corpus),
    ]
    order = {s.session: i for i, s in enumerate(seeds)}
    conversations.sort(key=lambda c: order[c.seeds[0]])
    index = CorpusIndex(conversations=conversations)
    (corpus / "index.json").write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return index


def _build_claude(seeds: list[SessionSeed], corpus: Path) -> list[MergedConversation]:
    found = [claude.find_session(s.session, claude.CLAUDE_PROJECTS) for s in seeds]
    by_id = {session.session_id: seed for session, seed in zip(found, seeds, strict=True)}
    raw = corpus / "raw" / "claude"
    results = []
    for group in claude.group_forks(found, claude.CLAUDE_PROJECTS):
        sources = []
        for session in group:
            target = raw / session.transcript.name
            _mirror(session.transcript, target)
            if session.sidecar_dir:
                _mirror_tree(session.sidecar_dir, raw / session.sidecar_dir.name)
            sources.append(Source(session_id=session.session_id, path=target, label=_rel(target)))
        group_seeds = [by_id[s.session_id] for s in group if s.session_id in by_id]
        results.append(_merge_group("claude", group_seeds, sources, corpus, claude.copy_key))
    return results


def _build_codex(seeds: list[SessionSeed], corpus: Path) -> list[MergedConversation]:
    threads = codex.index_threads(codex.CODEX_SESSIONS)
    # Seeds whose fork chains share a root are one conversation (possibly branched).
    members: dict[str, dict[str, codex.CodexThread]] = {}
    group_seeds: dict[str, list[SessionSeed]] = {}
    for seed in seeds:
        chain = codex.fork_chain(codex.resolve_prefix(seed.session, threads), threads)
        root = chain[0].thread_id
        # dict keeps first insertion, so ancestors stay ahead of descendants.
        members.setdefault(root, {}).update({t.thread_id: t for t in chain})
        group_seeds.setdefault(root, []).append(seed)

    results = []
    for root, group in members.items():
        sources = []
        for thread in group.values():
            folder = corpus / "raw" / "codex" / thread.thread_id
            for segment in thread.segments:
                target = folder / segment.name
                _mirror(segment, target)
                sources.append(Source(session_id=thread.thread_id, path=target, label=_rel(target)))
            for sub in thread.subagents:
                _mirror(sub, folder / "subagents" / sub.name)
        results.append(_merge_group("codex", group_seeds[root], sources, corpus, None))
    return results


def _merge_group(
    platform: Platform,
    seeds: list[SessionSeed],
    sources: list[Source],
    corpus: Path,
    copy_key: CopyKey | None,
) -> MergedConversation:
    out = corpus / "merged" / f"{platform}-{seeds[0].session}.jsonl"
    result: MergeResult = merge(sources, out, copy_key)
    members = list(dict.fromkeys(s.session_id for s in sources))
    print(
        f"{out.name}: {len(members)} member(s), {result.records} records, "
        f"{result.copies_dropped} fork copies dropped, {result.salvaged} salvaged, "
        f"{len(result.damage)} damaged line(s)"
    )
    return MergedConversation(
        platform=platform,
        merged_file=_rel(out),
        seeds=[s.session for s in seeds],
        members=members,
        records=result.records,
        copies_dropped=result.copies_dropped,
        salvaged=result.salvaged,
        damaged=[
            DamagedLineReport(
                file=d.label, line=d.damaged.line, chars=d.damaged.chars, head=d.damaged.head
            )
            for d in result.damage
        ],
    )


def _mirror(src: Path, dst: Path) -> None:
    """Copy unless dst already matches; copy2 preserves mtime, so size+mtime is the check."""
    if dst.exists():
        a, b = src.stat(), dst.stat()
        if a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime):
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _mirror_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        if path.is_file():
            _mirror(path, dst / path.relative_to(src))


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(TOOL_ROOT))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Mirror and merge the intent corpus.")
    parser.add_argument("--seeds", type=Path, default=TOOL_ROOT / "sessions.yaml")
    parser.add_argument("--corpus", type=Path, default=TOOL_ROOT / "corpus")
    args = parser.parse_args()
    index = build(load_seeds(args.seeds), args.corpus)
    print(f"{len(index.conversations)} conversations -> {args.corpus / 'index.json'}")


if __name__ == "__main__":
    main()
