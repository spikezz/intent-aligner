"""An intent as the steps that find one write it: text, evidence, and whether it was said.

Segmentation writes the intents of user segments and level-1 compression those of
tool calls; the main loop takes both as they are, so both share this model, its JSON
schema and the evidence check.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1},
        "evidence": {"type": "string", "minLength": 1},
        "explicit": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["text", "evidence", "explicit", "note"],
    "additionalProperties": False,
}


class Intent(BaseModel):
    """One intent of a unit; `note` gives the grounds whenever `explicit` is false."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    explicit: bool
    note: str

    @model_validator(mode="after")
    def _inferred_has_grounds(self) -> Intent:
        if not self.explicit and not self.note.strip():
            raise ValueError(f"意图 {self.text!r} 的 explicit 是 false，note 必须写推断依据")
        return self


def is_excerpt(evidence: str, text: str) -> bool:
    """Whether every `…`-separated piece of `evidence` occurs in `text`, in order.

    Examples:
        >>> is_excerpt("调查清楚…使用方法", "调查清楚这个项目的使用方法，")
        True
        >>> is_excerpt("使用方法…调查清楚", "调查清楚这个项目的使用方法，")
        False
    """
    position = 0
    for piece in (p.strip() for p in evidence.split("…")):
        found = text.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    return True


def stray_evidence(intents: list[Intent], text: str) -> list[str]:
    """The evidence of `intents` that is not an excerpt of `text`."""
    return [i.evidence for i in intents if not is_excerpt(i.evidence, text)]
