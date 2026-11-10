from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from intent_aligner import experiment_results as results
from intent_aligner.experiment_results import ExperimentResult, record_reply, refresh_results


def _reply(n: int = 0) -> ExperimentResult:
    return ExperimentResult(
        unit="unit:7",
        run=n,
        model="claude-opus-5-5",
        system="c",
        transcript="/missing/transcript.jsonl",
        answer='{"edges": []}',
    )


def test_answer_is_saved_before_optional_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.jsonl"
    receipts = path.with_suffix(".receipts.jsonl")

    def fail(_: Path) -> None:
        saved = ExperimentResult.model_validate_json(receipts.read_text())
        assert saved.answer == _reply().answer
        raise RuntimeError("statistics broke")

    monkeypatch.setattr(results, "read_exchange", fail)
    got = record_reply(path, _reply())
    assert got.answer == _reply().answer
    assert got.metadata_error == "RuntimeError: statistics broke"
    assert got.error is None
    assert not path.exists(), "simulate collector exiting before writing a row"
    assert refresh_results(path) == 1
    restored = ExperimentResult.model_validate_json(path.read_text())
    assert restored.answer == _reply().answer
    assert restored.transcript == _reply().transcript
    assert refresh_results(path) == 1, "recovery must not duplicate the repeat"


def test_unknown_statistics_does_not_hide_answer(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps({"type": "request", "model": "m", "system_prompt": "s", "prompt": "p"}) + "\n"
    )
    got = record_reply(
        tmp_path / "results.jsonl", _reply().model_copy(update={"transcript": str(transcript)})
    )
    assert got.answer == _reply().answer and got.tokens is None and got.metadata_error is None


def test_concurrent_receipts_recover_all_completed_calls(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    with ThreadPoolExecutor(5) as pool:
        saved = list(pool.map(lambda n: record_reply(path, _reply(n)), range(20)))
    assert len(saved) == 20
    assert refresh_results(path) == 20
    rows = [ExperimentResult.model_validate_json(line) for line in path.read_text().splitlines()]
    assert {r.run for r in rows} == set(range(20))
    assert all(r.answer == _reply().answer for r in rows)
    assert refresh_results(path) == 20


def test_final_answer_is_preserved_when_receipt_also_exists(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    row = _reply().model_copy(update={"answer": "final"})
    path.write_text(row.model_dump_json(exclude_none=True) + "\n")
    record_reply(path, _reply())
    assert refresh_results(path) == 1
    assert ExperimentResult.model_validate_json(path.read_text()).answer == "final"


def test_receipt_write_failure_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "absent" / "results.jsonl"
    with pytest.raises(FileNotFoundError):
        record_reply(path, _reply())
