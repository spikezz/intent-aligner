"""Replays, under the current prompts, the units of the problem table whose fix is only partly verified
(2026-11-10): units 30 and 64 (items 1, 4, 6: tool calls joining an existing aggregate, step prompt) and
units 58 and 59 (item 3: the <当前单元> sentence; 59 drops I010, said earlier in its own speech).
Relation list and schema are the current ones. 3 runs each; meant for an override endpoint such as
claude claude-opus-5-5."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from intent_aligner import extract
from intent_aligner.experiment_results import ExperimentResult, record_reply, refresh_results
from intent_aligner.llm import CALL_SUBJECT, LLM_HOME, LLMCall, LLMError, complete

OUT = Path(sys.argv[1])
# Optional `backend model`: sends every unit to that endpoint (e.g. claude claude-opus-5-5, which answers
# faster than the local Qwen) and keeps its rows in their own file.
OVERRIDE = {"backend": sys.argv[2], "model": sys.argv[3]} if len(sys.argv) > 3 else None
RESULTS = OUT / (
    f"results-unverified-{OVERRIDE['model']}.jsonl" if OVERRIDE else "results-unverified.jsonl"
)
RELATIONS = json.dumps(
    [
        {"relation": n, "from": a, "to": b, "meaning": m}
        for n, (a, b, m) in extract._C_RELATIONS.items()
    ],
    ensure_ascii=False,
    separators=(",", ":"),
)
assert '"from 直接服务于 to"' in RELATIONS


def _request(tid: str) -> dict[str, Any]:
    path = LLM_HOME / "hive" / f"{tid}.jsonl"
    if not path.exists():
        path = next((LLM_HOME / "codex").glob(f"{tid}*.jsonl"))
    rq: dict[str, Any] = json.loads(path.read_text().splitlines()[0])
    if OVERRIDE:
        return rq | OVERRIDE
    return rq | {"model": "deployment/pc-qwen38-np6"} if rq["backend"] == "hive" else rq


STEP = {"unit:30", "unit:64"}
UNITS: list[tuple[str, str, set[str]]] = [
    ("unit:30", "e5b32d03e42e494d8975f2ccf616e92b", set()),
    ("unit:64", "b0d1ec75f8e543daa9ab6f743314f02c", set()),
    ("unit:58", "d3c71b68eb6946d493e00d2e96c544df", set()),
    ("unit:59", "975d4078347d4737a684383b7fc8c9c9", {"I010"}),
]
jobs = []
for unit, tid, same_speech in UNITS:
    rq = _request(tid)
    system = extract.C_STEP_SYSTEM_PROMPT if unit in STEP else extract.C_SYSTEM_PROMPT
    prompt, count = re.subn(
        r'"relations":\[.*?\](?=,\s*"unit_intents")',
        lambda _: f'"relations":{RELATIONS}',
        rq["prompt"],
    )
    assert count == 1, unit
    candidates = re.search(r'"candidates":(\[.*?\])(?=,\s*")', prompt)
    assert candidates is not None, unit
    listed = json.loads(candidates.group(1))
    assert same_speech <= set(listed), (unit, listed)
    kept = json.dumps(
        [c for c in listed if c not in same_speech], ensure_ascii=False, separators=(",", ":")
    )
    prompt = re.sub(
        r'"candidates":\[.*?\](?=,\s*")',
        lambda _: f'"candidates":{kept}',  # noqa: B023 - re.sub calls this before the loop advances.
        prompt,
        count=1,
    )
    jobs += [(unit, n, rq, system, prompt) for n in range(3)]

# A resumed run skips the (unit, run) pairs already answered, so an interrupted batch costs only what is left.
results = RESULTS
refresh_results(results)
done = (
    {
        (r["unit"], r["run"])
        for r in map(json.loads, results.read_text().splitlines())
        if "error" not in r
    }
    if results.exists()
    else set()
)
jobs = [j for j in jobs if (j[0], j[1]) not in done]


def _run(job: tuple[str, int, dict[str, Any], str, str]) -> ExperimentResult:
    unit, n, rq, system, prompt = job
    CALL_SUBJECT.set(f"exp unverified {unit} {n}")
    row = ExperimentResult(
        unit=unit,
        run=n,
        model=rq["model"],
        system="step" if system is extract.C_STEP_SYSTEM_PROMPT else "c",
    )
    try:
        reply = complete(
            LLMCall(
                backend=rq["backend"],
                model=rq["model"],
                effort=rq["effort"],
                system_prompt=system,
                prompt=prompt,
                json_schema=extract.C_SCHEMA,
                stream=True,
            )
        )
    except LLMError as error:
        return row.model_copy(update={"error": str(error)[:500]})
    return record_reply(
        RESULTS,
        row.model_copy(
            update={
                "transcript": str(reply.transcript),
                "answer": reply.text,
            }
        ),
    )


with ThreadPoolExecutor(5) as pool, RESULTS.open("a") as f:
    for row in pool.map(_run, jobs):
        f.write(row.model_dump_json(exclude_none=True) + "\n")
        f.flush()
        print(row.unit, row.run, row.tokens or row.error, flush=True)
