"""The compressor with scripted models, on the timeline from test_window."""

from __future__ import annotations

import doctest
import json
from collections.abc import Sequence

import pytest

from intent_aligner import compress
from intent_aligner.compress import compress_conversation
from intent_aligner.llm import TRUNCATED_FEEDBACK, Ask, LLMError, TruncatedReply
from intent_aligner.units import Item, UserSignal
from intent_aligner.window import CallDigest, Digest, Digests, ResultDigest, RoundDigest
from tests.test_segment import _Script
from tests.test_window import TIMELINE, _call, _call_digest, _result, _user


def test_docstring_examples_hold() -> None:
    result = doctest.testmod(compress)

    assert result.failed == 0 and result.attempted > 0, result


def _intents_answer(prompt: str, text: str, targets: str) -> str:
    """One inferred intent quoting the first line of the call, which is always in it."""
    first_line = prompt.split("\n")[1]
    intent = {"text": text, "evidence": first_line, "explicit": False, "note": "从命令推断"}
    return json.dumps({"intents": [intent], "targets": targets}, ensure_ascii=False)


def _call_answer(prompt: str) -> str:
    return _intents_answer(prompt, "看目录", "ls")


def _result_answer(prompt: str) -> str:
    return json.dumps({"summary": "列出了文件"}, ensure_ascii=False)


def _round_answer(prompt: str) -> str:
    return json.dumps({"summary": "看了目录"}, ensure_ascii=False)


def _run(
    items: Sequence[Item],
    done: Digests,
    ask_call: Ask,
    ask_result: Ask,
    ask_round: Ask = _round_answer,
) -> list[Digest]:
    return list(compress_conversation(items, done, ask_call, ask_result, ask_round, len, 1))


def test_level_one_then_one_digest_per_round() -> None:
    made = list(
        compress_conversation(
            TIMELINE, Digests(), _call_answer, _result_answer, _round_answer, len, 4
        )
    )

    assert sorted(
        (d.kind, d.round_uuid if isinstance(d, RoundDigest) else d.tool_use_id) for d in made
    ) == [
        ("call", "c1"),
        ("call", "c2"),
        ("result", "c1"),
        ("result", "c2"),
        ("round", "u1"),
        ("round", "u2"),
    ], made
    assert made[-1].kind == "round", made


def test_evidence_outside_the_call_is_retried() -> None:
    stray = {"text": "看目录", "evidence": "rm -rf", "explicit": True, "note": ""}
    calls = _Script(
        json.dumps({"intents": [stray], "targets": "ls"}, ensure_ascii=False),
        _call_answer("<调用>\nBash:"),
    )
    made = _run([_call("c1")], Digests(), calls, _result_answer)

    assert "不是 <调用> 原文的摘录" in calls.prompts[1], calls.prompts[1]
    assert [d.kind for d in made] == ["call", "round"], made


def test_a_truncated_reply_is_retried() -> None:
    calls = _Script(TruncatedReply("length"), _call_answer("<调用>\nBash:"))
    made = _run([_call("c1")], Digests(), calls, _result_answer)

    assert TRUNCATED_FEEDBACK in calls.prompts[1], calls.prompts[1]
    assert [d.kind for d in made] == ["call", "round"], made


def test_a_call_with_two_purposes_keeps_both() -> None:
    both = [
        {"text": "建分析目录", "evidence": "Bash", "explicit": False, "note": "mkdir"},
        {"text": "读文档", "evidence": "ls", "explicit": False, "note": "cat"},
    ]
    answer = json.dumps({"intents": both, "targets": "ls"}, ensure_ascii=False)
    made = _run([_call("c1")], Digests(), lambda _: answer, _result_answer)

    call = next(d for d in made if isinstance(d, CallDigest))
    assert call.purpose == "建分析目录；读文档", call


def test_a_call_is_read_alone() -> None:
    calls = _Script(*[_call_answer("<调用>\nBash:")] * 2)
    _run(TIMELINE, Digests(), calls, _result_answer)

    assert calls.prompts[0] == "<调用>\nBash:\ncommand: ls\n</调用>", calls.prompts[0]


def test_a_result_is_read_against_its_calls_intent_and_full_text() -> None:
    call = _call("c1").model_copy(update={"text": "Bash:\ncommand: " + "z" * 2_000})
    results = _Script(_result_answer(""))
    _run([call, _result("c1")], Digests(), _call_answer, results)

    expected = f"<调用>\n意图：看目录\n原文：\n{call.text}\n</调用>"
    assert expected in results.prompts[0], results.prompts[0][:200]


def test_a_call_that_leaves_no_room_for_its_result_fails_with_its_id() -> None:
    call = _call("c1").model_copy(update={"text": "z" * 7_000})

    with pytest.raises(LLMError, match="1 of 1 jobs failed"):
        _run([call, _result("c1")], Digests(), _call_answer, _result_answer)


def test_the_round_reads_intents_targets_and_results_not_raw_text() -> None:
    rounds = _Script(*[_round_answer("")] * 2)
    _run(TIMELINE, Digests(), _call_answer, _result_answer, rounds)

    expected = "<用户消息>\n第一条\n</用户消息>\n<轨迹>\nBash 意图：看目录 对象：ls 结果：列出了文件\n</轨迹>"
    assert rounds.prompts[0] == expected, rounds.prompts[0]


def test_a_rejected_call_enters_the_trajectory_as_rejected() -> None:
    signal = UserSignal(
        source=TIMELINE[0].source, signal="rejection", text="不要", tool_use_id="c1"
    )
    rounds = _Script(_round_answer(""))
    _run([TIMELINE[0], _call("c1"), signal], Digests(), _call_answer, _result_answer, rounds)

    assert "结果：用户拒绝了这次调用：不要" in rounds.prompts[0], rounds.prompts[0]


def test_a_long_trajectory_is_summarized_again() -> None:
    line = "意" * 5_000
    many: list[Item] = [_user("u1", "做"), _call("c1"), _call("c2")]
    rounds = _Script(_round_answer(""), _round_answer(""), _round_answer(""))
    made = _run(many, Digests(), lambda p: _intents_answer(p, line, "t"), _result_answer, rounds)

    assert sum(d.kind == "round" for d in made) == 1, made
    assert "（第 2/2 段）" in rounds.prompts[1], rounds.prompts[1][:60]
    assert (
        rounds.prompts[2] == "<用户消息>\n做\n</用户消息>\n<轨迹>\n看了目录\n看了目录\n</轨迹>"
    ), rounds.prompts[2]


def test_tool_work_before_any_message_has_no_message_block() -> None:
    rounds = _Script(_round_answer(""))
    _run([_call("c1"), _result("c1")], Digests(), _call_answer, _result_answer, rounds)

    assert rounds.prompts[0].startswith("<轨迹>"), rounds.prompts[0]


def test_a_long_queued_message_is_cut_and_marked() -> None:
    queued = _user("u1", "长" * 3_000).model_copy(update={"queued": True})
    rounds = _Script(_round_answer(""))
    _run([queued, _call("c1"), _result("c1")], Digests(), _call_answer, _result_answer, rounds)

    head = rounds.prompts[0].split("</用户消息>")[0]
    assert head.startswith("<用户消息>（AI 工作时发出）（太长，只保留开头）\n"), head[:60]
    body = head.split("\n", 1)[1]
    assert body == "长" * compress.MESSAGE_ROOM + "\n", len(body)


def test_a_long_result_is_summarized_chunk_by_chunk() -> None:
    # Two 5,000-char lines under len() as the count: together they exceed the 9,000 budget.
    long_result = _result("c1").model_copy(update={"text": ("y" * 4_999 + "\n") * 2})
    results = _Script(*[_result_answer("")] * 2)
    made = _run([_call("c1"), long_result], Digests(), _call_answer, results)

    digest = next(d for d in made if isinstance(d, ResultDigest))
    assert len(digest.summaries) == 2, digest
    assert "（第 2/2 段）" in results.prompts[1], results.prompts[1][:200]


def test_existing_digests_are_not_redone() -> None:
    done = Digests(
        calls={"c1": _call_digest("c1", "i")},
        results={"c1": ResultDigest(tool_use_id="c1", summaries=["s"])},
        rounds={"u1": RoundDigest(round_uuid="u1", summary="r")},
    )
    made = _run(TIMELINE, done, _call_answer, _result_answer)

    assert [(d.kind) for d in made] == ["call", "result", "round"], made


def test_a_failed_job_keeps_the_others_and_blocks_level_two() -> None:
    def ask_call(prompt: str) -> str:
        if "command: bad" in prompt:
            raise LLMError("quota")
        return _call_answer(prompt)

    bad = _call("c1").model_copy(update={"text": "Bash:\ncommand: bad"})
    made: list[Digest] = []
    with pytest.raises(LLMError, match="1 of 2 jobs failed"):
        for digest in compress_conversation(
            [bad, _call("c2")], Digests(), ask_call, _result_answer, _round_answer, len, 1
        ):
            made.append(digest)

    assert [d.kind for d in made] == ["call"], made


def test_bad_json_is_retried_then_fails() -> None:
    calls = _Script(*["not json"] * compress.MAX_ATTEMPTS)

    with pytest.raises(LLMError, match="No valid digest for c1"):
        _run(TIMELINE[:3], Digests(), calls, _result_answer)
