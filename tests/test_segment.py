"""The LLM pointer: anchor location offline, the walk with a scripted model."""

from __future__ import annotations

import doctest
import json
import re
from datetime import UTC, datetime

import pytest

from intent_aligner import segment
from intent_aligner.llm import TRUNCATED_FEEDBACK, Ask, LLMError, TruncatedReply
from intent_aligner.segment import (
    AnchorError,
    MessageSegments,
    locate_end,
    segment_conversation,
    segment_message,
)
from intent_aligner.units import Source, UserMessage
from intent_aligner.window import Count
from tests.test_window import DIGESTS, TIMELINE

# The opening of the playground message (c88f10d9) and its hand-cut boundaries 0, 67, 88.
BODY = (
    "调查清楚/home/spikezz/.openclaw/workspace/projects/somnia-farm/项目的使用方法，"
    "我接下来要你用我已经搭好的部分的基础设施，"
    "自己运用工具的所有功能（参数探索）和已经搭建好的wf ,"
    "来生成一段音乐的mv。音乐是这个/home/spikezz/Music/The Wind Remembers.wav\n\n"
)


def test_docstring_examples_hold() -> None:
    result = doctest.testmod(segment)

    assert result.failed == 0 and result.attempted > 0, result


def test_anchor_gives_the_playground_boundaries() -> None:
    assert locate_end(BODY, 0, "项目的使用方法，", from_start=False) == 67
    assert locate_end(BODY, 67, "基础设施，", from_start=False) == 88
    assert locate_end(BODY, 88, "The Wind Remembers.wav", from_start=False) == len(BODY)


def test_earlier_occurrences_do_not_count() -> None:
    assert locate_end("好，好，结束", 2, "好，", from_start=False) == 4


def test_overlapping_occurrences_are_ambiguous() -> None:
    with pytest.raises(AnchorError, match="出现了 2 次") as caught:
        locate_end("哈哈哈", 0, "哈哈", from_start=False)

    assert caught.value.ambiguous


def test_ambiguity_feedback_shows_each_place() -> None:
    with pytest.raises(AnchorError) as caught:
        locate_end("先改A，然后测试，再改B，然后测试，", 0, "然后测试，", from_start=False)

    assert "先改A，【然后测试，】再改B" in str(caught.value), caught.value
    assert "再改B，【然后测试，】" in str(caught.value), caught.value


def test_missing_anchor_is_not_ambiguous() -> None:
    with pytest.raises(AnchorError, match="找不到") as caught:
        locate_end(BODY, 0, "项目的使用方法,", from_start=False)

    assert not caught.value.ambiguous


def test_from_start_must_be_a_prefix() -> None:
    with pytest.raises(AnchorError, match="不是以"):
        locate_end("好，好，结束", 0, "好，结束", from_start=True)


def _message(body: str) -> UserMessage:
    source = Source(session="s", uuid="u", timestamp=datetime(2026, 11, 6, tzinfo=UTC))
    return UserMessage(source=source, body=body, queued=False, images=())


class _Script:
    """A fake model that replays canned answers, raising the ones that are errors."""

    def __init__(self, *answers: str | LLMError) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, LLMError):
            raise answer
        return answer


def _answer(end: str, evidence: str | None = None, *, explicit: bool = True, note: str = "") -> str:
    intent = {"text": "i", "evidence": evidence or end, "explicit": explicit, "note": note}
    return json.dumps({"intents": [intent], "segment_end": end}, ensure_ascii=False)


def test_segments_tile_the_body() -> None:
    script = _Script(_answer("基础设施，"), _answer("The Wind Remembers.wav"))
    segments = segment_message("", _message(BODY), script)

    assert [(s.start, s.end) for s in segments] == [(0, 88), (88, len(BODY))], segments
    assert "<已处理>\n调查清楚" in script.prompts[1], "the second call must see the first segment"


def test_prompt_carries_the_window_first() -> None:
    script = _Script(_answer("好的。"))
    segment_message("【AI】\n要我继续吗？", _message("好的。"), script)

    assert script.prompts[0].startswith("<此前的对话>\n【AI】\n要我继续吗？"), script.prompts[0]


def test_ambiguous_anchor_switches_to_copy_from_start() -> None:
    body = "先改A，然后测试，再改B，然后测试，"
    script = _Script(
        _answer("然后测试，"), _answer("先改A，然后测试，"), _answer("再改B，然后测试，")
    )
    segments = segment_message("", _message(body), script)

    assert [s.text for s in segments] == ["先改A，然后测试，", "再改B，然后测试，"], segments
    assert segments[0].attempts == 2, segments[0]
    assert segment.FROM_START_RULE in script.prompts[1], script.prompts[1]


def test_bad_json_is_retried() -> None:
    segments = segment_message("", _message("好的。"), _Script("not json", _answer("好的。")))

    assert segments[0].attempts == 2, segments


def test_a_truncated_reply_is_retried() -> None:
    script = _Script(TruncatedReply("length"), _answer("好的。"))
    segments = segment_message("", _message("好的。"), script)

    assert segments[0].attempts == 2, segments
    assert TRUNCATED_FEEDBACK in script.prompts[1], script.prompts[1]


def test_evidence_outside_the_segment_is_retried() -> None:
    script = _Script(_answer("好的。", evidence="就这样"), _answer("好的。"), _answer("就这样。"))
    segments = segment_message("", _message("好的。就这样。"), script)

    assert segments[0].attempts == 2, segments[0]
    assert "不是这一段的原文摘录" in script.prompts[1], script.prompts[1]


def test_evidence_may_skip_the_middle_with_an_ellipsis() -> None:
    script = _Script(_answer("就这样。", evidence="好的…就这样"))
    segments = segment_message("", _message("好的。就这样。"), script)

    assert segments[0].attempts == 1, segments[0]


def test_an_inferred_intent_without_grounds_is_retried() -> None:
    script = _Script(
        _answer("继续", explicit=False),
        _answer("继续", explicit=False, note="此前 AI 在排查 blocked 任务"),
    )
    segments = segment_message("", _message("继续"), script)

    assert segments[0].attempts == 2, segments[0]
    assert segments[0].intents[0].note == "此前 AI 在排查 blocked 任务", segments[0]
    assert "note 必须写推断依据" in script.prompts[1], script.prompts[1]


def test_unlocatable_segment_takes_the_rest_for_review() -> None:
    script = _Script(_answer("好的。"), *[_answer("不存在")] * segment.MAX_ATTEMPTS)
    segments = segment_message("", _message("好的。就这样。"), script)

    assert [(s.text, s.needs_review) for s in segments] == [("好的。", False), ("就这样。", True)]
    assert segments[1].intents == [], segments[1]


def _whole_rest(prompt: str) -> str:
    """A model that makes the whole remaining text one segment."""
    found = re.search(r"<剩余>\n(.*)\n</剩余>", prompt, re.DOTALL)
    assert found, prompt
    return _answer(found.group(1))


def _run(ask: Ask, done: set[str], count: Count = len) -> list[MessageSegments]:
    return list(segment_conversation(TIMELINE, DIGESTS, done, ask, count, 2))


def test_every_message_not_done_is_segmented() -> None:
    made = _run(_whole_rest, {"u2"})

    assert sorted(m.uuid for m in made) == ["u1", "u3"], made
    assert all(len(m.segments) == 1 and not m.segments[0].needs_review for m in made), made


def test_a_window_that_does_not_fit_gives_one_review_segment() -> None:
    made = _run(_whole_rest, set(), count=lambda text: 1_000 * len(text))

    assert {(m.uuid, len(m.segments), m.segments[0].needs_review) for m in made} == {
        ("u1", 1, True),
        ("u2", 1, True),
        ("u3", 1, True),
    }, made


def test_a_backend_failure_keeps_the_other_messages() -> None:
    def ask(prompt: str) -> str:
        if "<剩余>\n第二条" in prompt:
            raise LLMError("backend down")
        return _whole_rest(prompt)

    made: list[MessageSegments] = []
    with pytest.raises(LLMError, match="1 of 3 messages failed"):
        made.extend(segment_conversation(TIMELINE, DIGESTS, set(), ask, len, 2))

    assert sorted(m.uuid for m in made) == ["u1", "u3"], made
