"""Seed sessions: the human-curated entry points of the corpus."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

BERLIN = ZoneInfo("Europe/Berlin")

Platform = Literal["claude", "codex"]


class SessionSeed(BaseModel):
    """One row of sessions.yaml; naive intent_start values are Berlin local time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    platform: Platform
    session: str
    intent_start: datetime

    @field_validator("intent_start")
    @classmethod
    def _assume_berlin(cls, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=BERLIN)


def load_seeds(path: Path) -> list[SessionSeed]:
    """Parse and validate sessions.yaml."""
    return TypeAdapter(list[SessionSeed]).validate_python(yaml.safe_load(path.read_text()))
