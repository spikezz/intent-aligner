"""The LLM pointer: anchor location offline, the walk with a scripted model."""

from __future__ import annotations

import doctest
import json
from datetime import UTC, datetime

import pytest

from intent_aligner import segment
from intent_aligner.llm import LLMError
from intent_aligner.segment import AnchorError, locate_end, segment_message
from intent_aligner.units import Source, UserMessage

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
    """A fake model that replays canned answers and keeps the prompts it was sent."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


def _answer(end: str) -> str:
    return json.dumps({"intents": ["i"], "segment_end": end}, ensure_ascii=False)


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


def test_unlocatable_segment_stops_the_walk() -> None:
    script = _Script(*[_answer("不存在")] * segment.MAX_ATTEMPTS)

    with pytest.raises(LLMError, match="not located"):
        segment_message("", _message("好的。"), script)
