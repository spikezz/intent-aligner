"""The compressor with scripted models, on the timeline from test_window."""

from __future__ import annotations

import doctest
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from intent_aligner import compress
from intent_aligner.compress import compress_conversation, summarize_trajectory
from intent_aligner.llm import TRUNCATED_FEEDBACK, Ask, LLMError, ThinkingCapped, TruncatedReply
from intent_aligner.units import Item
from intent_aligner.window import PROMPT_BUDGET, CallDigest, Digest, Digests, ResultDigest
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
) -> list[Digest]:
    return list(compress_conversation(items, done, ask_call, ask_result, len, 1))


def test_each_call_and_result_gets_one_digest() -> None:
    made = list(compress_conversation(TIMELINE, Digests(), _call_answer, _result_answer, len, 4))

    got = sorted((d.kind, d.tool_use_id) for d in made)
    expected = [("call", "c1"), ("call", "c2"), ("result", "c1"), ("result", "c2")]
    assert got == expected, f"{got!r} vs {expected!r}"


def test_evidence_outside_the_call_is_retried() -> None:
    stray = {"text": "看目录", "evidence": "rm -rf", "explicit": True, "note": ""}
    calls = _Script(
        json.dumps({"intents": [stray], "targets": "ls"}, ensure_ascii=False),
        _call_answer("<调用>\nBash:"),
    )
    made = _run([_call("c1")], Digests(), calls, _result_answer)

    assert "不是 <调用> 原文的摘录" in calls.prompts[1], calls.prompts[1]
    assert [d.kind for d in made] == ["call"], made


def test_a_truncated_reply_is_retried() -> None:
    calls = _Script(TruncatedReply("length"), _call_answer("<调用>\nBash:"))
    made = _run([_call("c1")], Digests(), calls, _result_answer)

    assert TRUNCATED_FEEDBACK in calls.prompts[1], calls.prompts[1]
    assert [d.kind for d in made] == ["call"], made


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
    # Fits its own prompt, but quoted in the result prompt it leaves under MIN_RESULT_ROOM.
    call = _call("c1").model_copy(update={"text": "z" * (PROMPT_BUDGET - 2_000)})

    with pytest.raises(LLMError, match="1 of 1 jobs failed"):
        _run([call, _result("c1")], Digests(), _call_answer, _result_answer)


def test_a_long_trajectory_is_summarized_again() -> None:
    rounds = _Script(_round_answer(""), _round_answer(""), _round_answer(""))
    summary, reviews = summarize_trajectory(
        _user("u1", "做"), ["意" * (PROMPT_BUDGET // 2)] * 2, rounds, len, "r"
    )

    assert (summary, reviews) == ("看了目录", []), (summary, reviews)
    assert "（第 2/2 段）" in rounds.prompts[1], rounds.prompts[1][:60]
    assert (
        rounds.prompts[2] == "<发言>（用户）\n做\n</发言>\n<轨迹>\n看了目录\n看了目录\n</轨迹>"
    ), rounds.prompts[2]


def test_tool_work_before_any_message_has_no_message_block() -> None:
    rounds = _Script(_round_answer(""))
    summarize_trajectory(None, ["一行"], rounds, len, "r")

    assert rounds.prompts[0].startswith("<轨迹>"), rounds.prompts[0]


def test_a_long_queued_message_is_cut_and_marked() -> None:
    queued = _user("u1", "长" * 3_000).model_copy(update={"queued": True})
    rounds = _Script(_round_answer(""))
    summarize_trajectory(queued, ["一行"], rounds, len, "r")

    head = rounds.prompts[0].split("</发言>")[0]
    assert head.startswith("<发言>（用户）（AI 工作时发出）（太长，只保留开头）\n"), head[:60]
    body = head.split("\n", 1)[1]
    assert body == "长" * compress.MESSAGE_ROOM + "\n", len(body)


def test_a_long_result_is_summarized_chunk_by_chunk() -> None:
    # Two lines of half the budget under len() as the count: each fits, both together do not.
    line = "y" * (PROMPT_BUDGET // 2 - 1) + "\n"
    long_result = _result("c1").model_copy(update={"text": line * 2})
    results = _Script(*[_result_answer("")] * 2)
    made = _run([_call("c1"), long_result], Digests(), _call_answer, results)

    digest = next(d for d in made if isinstance(d, ResultDigest))
    assert len(digest.summaries) == 2, digest
    assert "（第 2/2 段）" in results.prompts[1], results.prompts[1][:200]


def test_existing_digests_are_not_redone() -> None:
    done = Digests(
        calls={"c1": _call_digest("c1", "i")},
        results={"c1": ResultDigest(tool_use_id="c1", summaries=["s"])},
    )
    made = _run(TIMELINE, done, _call_answer, _result_answer)

    assert [(d.kind) for d in made] == ["call", "result"], made


def test_a_failed_job_keeps_the_others() -> None:
    def ask_call(prompt: str) -> str:
        if "command: bad" in prompt:
            raise LLMError("quota")
        return _call_answer(prompt)

    bad = _call("c1").model_copy(update={"text": "Bash:\ncommand: bad"})
    made: list[Digest] = []
    with pytest.raises(LLMError, match="1 of 2 jobs failed"):
        for digest in compress_conversation(
            [bad, _call("c2")], Digests(), ask_call, _result_answer, len, 1
        ):
            made.append(digest)

    assert [d.kind for d in made] == ["call"], made


def test_bad_json_is_retried_then_fails() -> None:
    calls = _Script(*["not json"] * compress.MAX_ATTEMPTS)

    with pytest.raises(LLMError, match="No valid digest for c1"):
        _run(TIMELINE[:3], Digests(), calls, _result_answer)


def _capped(answer: str) -> ThinkingCapped:
    return ThinkingCapped("capped", answer, Path("capped.jsonl"))


def test_a_capped_call_is_asked_again_with_the_same_prompt() -> None:
    calls = _Script(_capped(_call_answer("<调用>\nBash:")), _call_answer("<调用>\nBash:"))
    made = _run([_call("c1")], Digests(), calls, _result_answer)

    assert calls.prompts[0] == calls.prompts[1], calls.prompts
    assert [(d.kind, d.review_reason) for d in made] == [("call", None)], made


def test_a_call_capped_on_every_attempt_keeps_its_last_answer_for_review() -> None:
    calls = _Script(*[_capped(_call_answer("<调用>\nBash:"))] * compress.MAX_ATTEMPTS)
    made = _run([_call("c1")], Digests(), calls, _result_answer)

    call = next(d for d in made if isinstance(d, CallDigest))
    assert call.purpose == "看目录" and "cap" in (call.review_reason or ""), call


def test_a_result_capped_on_every_attempt_is_marked_for_review() -> None:
    results = _Script(*[_capped(_result_answer(""))] * compress.MAX_ATTEMPTS)
    made = _run([_call("c1"), _result("c1")], Digests(), _call_answer, results)

    result = next(d for d in made if isinstance(d, ResultDigest))
    assert result.summaries == ["列出了文件"], result
    assert "chunk 1/1" in (result.review_reason or ""), result


def test_a_trajectory_capped_on_every_attempt_keeps_its_passage_for_review() -> None:
    rounds = _Script(*[_capped(_round_answer(""))] * compress.MAX_ATTEMPTS)
    summary, reviews = summarize_trajectory(None, ["一行"], rounds, len, "r")

    assert summary == "看了目录" and len(reviews) == 1, (summary, reviews)
    assert "capped.jsonl" in reviews[0], reviews


def test_capped_answers_that_never_validate_fail_the_job() -> None:
    calls = _Script(*[_capped("not json")] * compress.MAX_ATTEMPTS)

    with pytest.raises(LLMError, match="No valid digest for c1"):
        _run([_call("c1")], Digests(), calls, _result_answer)
