"""Human prompts of a merged Claude conversation: the turn boundaries of a trajectory.

Usage: python -m intent_aligner.turns MERGED.jsonl [--index N] --out PATH
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def human_prompts(merged: Path) -> Iterator[dict[str, Any]]:
    """Yield the records the user typed or queued, in conversation order.

    Claude Code tags them origin.kind == "human"; tool results, task notifications
    and injected meta messages are also type "user" but carry another origin or none.
    """
    with merged.open(encoding="utf-8") as f:
        for line in f:
            record: dict[str, Any] = json.loads(line)
            if (record.get("origin") or {}).get("kind") == "human":
                yield record


def main() -> None:
    """CLI entry point: copy the Nth human prompt record to --out as indented JSON."""
    parser = argparse.ArgumentParser(description="Copy one human prompt record.")
    parser.add_argument("merged", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    count = 0
    for record in human_prompts(args.merged):
        if count == args.index:
            args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            return
        count += 1
    raise SystemExit(
        f"{args.merged} has {count} human prompts; --index {args.index} is out of range."
    )


if __name__ == "__main__":
    main()
