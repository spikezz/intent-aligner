"""The llm-chat.sh REPL, driven through stdin with a fake Python that records each call."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from intent_aligner.corpus import TOOL_ROOT

SCRIPT = TOOL_ROOT / "scripts" / "llm-chat.sh"

# Stands in for `python -m intent_aligner.llm`: logs argv, stdin and the system file.
FAKE_PY = """#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
system = open(args[args.index("--system") + 1]).read()
with open(sys.argv[0] + ".log", "a") as f:
    f.write(json.dumps({"argv": args, "stdin": sys.stdin.read(), "system": system}) + "\\n")
print("fake reply")
"""


def _run(tmp_path: Path, stdin: str, *args: str) -> tuple[str, list[dict[str, object]]]:
    fake = tmp_path / "fake-python"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env={"PY": str(fake), "PATH": "/usr/bin:/bin"},
        check=True,
    )
    log = tmp_path / "fake-python.log"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc.stdout, calls


def _flag(call: dict[str, object], name: str) -> str:
    argv = call["argv"]
    assert isinstance(argv, list), argv
    return str(argv[argv.index(name) + 1])


def test_each_message_is_one_call_with_current_settings(tmp_path: Path) -> None:
    stdin = "hello\n/backend codex\n/effort low\n/system Be terse.\nsecond\n/quit\n"
    out, calls = _run(tmp_path, stdin)

    assert len(calls) == 2, calls
    first, second = calls
    assert (_flag(first, "--backend"), _flag(first, "--model")) == ("claude", "sonnet"), first
    assert _flag(first, "--effort") == "xhigh", first
    assert first["stdin"] == "hello", first
    assert (_flag(second, "--backend"), _flag(second, "--model")) == ("codex", "gpt-6.1-sol")
    assert _flag(second, "--effort") == "low", second
    assert second["system"] == "Be terse.", second
    assert second["stdin"] == "second", "no history may be carried into the next call"
    assert out.count("fake reply") == 2, out


def test_start_backend_and_model_override(tmp_path: Path) -> None:
    _, calls = _run(tmp_path, "/model my-model\nhi\n", "pi")

    assert (_flag(calls[0], "--backend"), _flag(calls[0], "--model")) == ("pi", "my-model")


def test_invalid_commands_send_nothing(tmp_path: Path) -> None:
    out, calls = _run(tmp_path, "/backend gemini\n/effort max\n/nope\n")

    assert calls == [], calls
    assert "Unknown backend 'gemini'" in out and "Unknown effort 'max'" in out, out
    assert "Unknown command '/nope'" in out, out


def test_home_is_passed_on_only_when_given(tmp_path: Path) -> None:
    _, given = _run(tmp_path, "hi\n", "--home", "/x/home", "codex")
    (tmp_path / "fake-python.log").unlink()
    _, default = _run(tmp_path, "hi\n")

    assert _flag(given[0], "--home") == "/x/home", given
    assert _flag(given[0], "--backend") == "codex", given
    assert "--home" not in default[0]["argv"], "the default home belongs to the wrapper"  # type: ignore[operator]


def test_unknown_start_backend_exits(tmp_path: Path) -> None:
    proc = subprocess.run(["bash", str(SCRIPT), "gemini"], capture_output=True, text=True)

    assert proc.returncode == 2 and "Unknown backend" in proc.stderr, proc
