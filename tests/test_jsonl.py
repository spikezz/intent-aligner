from __future__ import annotations

from pathlib import Path

from intent_aligner.jsonl import DamagedLine, Record, read_jsonl


def _write(tmp_path: Path, *lines: bytes) -> Path:
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"\n".join(lines) + b"\n")
    return path


def test_torn_write_salvages_trailing_record(tmp_path: Path) -> None:
    torn = b'{"uuid":"a","message":{"usage":{"cache' + b'{"uuid":"b","n":1}'
    items = list(read_jsonl(_write(tmp_path, b'{"uuid":"0"}', torn)))

    assert items[0] == Record(line=0, data={"uuid": "0"}, salvaged=False), items
    assert isinstance(items[1], DamagedLine) and items[1].chars == 38, items[1]
    assert items[2] == Record(line=1, data={"uuid": "b", "n": 1}, salvaged=True), items[2]


def test_torn_multibyte_prefix_still_salvages(tmp_path: Path) -> None:
    torn = '{"text":"磁盘满'.encode()[:-1] + b'{"uuid":"c"}'
    items = list(read_jsonl(_write(tmp_path, torn)))

    assert [type(i).__name__ for i in items] == ["DamagedLine", "Record"], items
    assert isinstance(items[1], Record) and items[1].data == {"uuid": "c"}, items[1]


def test_truncated_line_without_successor_is_damage_only(tmp_path: Path) -> None:
    items = list(read_jsonl(_write(tmp_path, b'{"ordinal":7,"payload":{"x":"abc')))

    assert len(items) == 1 and isinstance(items[0], DamagedLine), items


def test_raw_control_characters_are_accepted(tmp_path: Path) -> None:
    items = list(read_jsonl(_write(tmp_path, b'{"cmd":"a\tb"}')))

    assert items == [Record(line=0, data={"cmd": "a\tb"}, salvaged=False)], items
