"""Replays with serves meaning "from 直接服务于 to" (2026-11-10), replacing "from 是 to 的手段：不先做
from 就做不了 to" and the earlier "from 是实现 to 的手段".

Qwen units 27 and 53 (serves linked to ancestor intents) and 7-10 (responds; 8 and 9 drop the earlier
intents of their own speech, as in exp_responds.py). Sol units of the last full rerun that wrote the
serves cycles G003<->I011 and G004<->I018/I019: the speech segments that announced I011, I018, I019
(their own speech's earlier intents dropped) and the tool calls that then joined G003 and G004 (step
prompt). The recorded relation list and schema are replaced by the current ones. 3 runs each.
"""

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
    f"results-direct-{OVERRIDE['model']}.jsonl" if OVERRIDE else "results-direct.jsonl"
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


UNITS: list[tuple[str, str, set[str]]] = [
    ("unit:7", "bcea59372a9341069015465eb82ca902", set()),
    ("unit:8", "588a70afe6b14d48861791aec73c9e46", {"I002"}),
    ("unit:9", "3d2c0e5849c74701a59a9fdcedb23cbf", {"I002", "I003"}),
    ("unit:10", "c69455523d1942b798e1d8fb347e43bd", set()),
    ("unit:27", "552aeda5ce54493686afb522ef28485c", set()),
    ("unit:53", "95607a3a4172494dab96ff0339873fe4", set()),
    ("sol I011", "01a1c570-f417", {"I010"}),
    ("sol G003 step", "01a1c572-e5ab", set()),
    ("sol I018", "01a1c591-6f0e", {"I017"}),
    ("sol I019", "01a1c592-afa9", {"I017", "I018"}),
    ("sol G004 step", "01a1c593-a480", set()),
]
jobs = []
for unit, tid, same_speech in UNITS:
    rq = _request(tid)
    system = (
        extract.C_STEP_SYSTEM_PROMPT
        if "又一步工具调用" in rq["system_prompt"]
        else extract.C_SYSTEM_PROMPT
    )
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
    CALL_SUBJECT.set(f"exp direct {unit} {n}")
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
