"""Public inputs contain user choices, never features, probabilities or clocks."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from match_insight.features.pre import ROLES, PlayerSlot
from match_insight.services.demo import AnalysisInput


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RosterSlot(InputModel):
    role: Literal["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]
    player_id: str = Field(min_length=1, max_length=100)


class ContextInput(InputModel):
    blue_team_id: str = Field(min_length=1, max_length=100)
    red_team_id: str = Field(min_length=1, max_length=100)
    blue_roster: list[RosterSlot] = Field(min_length=5, max_length=5)
    red_roster: list[RosterSlot] = Field(min_length=5, max_length=5)
    patch: str = Field(min_length=1, max_length=30)
    context_label: str | None = Field(default=None, max_length=160)
    user_confirmed: Literal[True]

    @model_validator(mode="after")
    def complete_rosters(self):
        if self.blue_team_id == self.red_team_id:
            raise ValueError("Hai đội phải khác nhau.")
        for roster in (self.blue_roster, self.red_roster):
            if tuple(slot.role for slot in roster) != ROLES:
                raise ValueError("Đội hình phải có đủ năm vị trí theo thứ tự.")
        if len({slot.player_id for slot in self.blue_roster + self.red_roster}) != 10:
            raise ValueError("Cần mười tuyển thủ khác nhau.")
        return self

    def to_selection(self):
        return AnalysisInput(
            self.blue_team_id, self.red_team_id,
            tuple(PlayerSlot(slot.role, slot.player_id) for slot in self.blue_roster),
            tuple(PlayerSlot(slot.role, slot.player_id) for slot in self.red_roster),
            self.patch, True, context_label=self.context_label or None,
        )


class PreInput(InputModel):
    operation_id: UUID
    context: ContextInput


class PostInput(InputModel):
    operation_id: UUID
    pre_evaluation_id: int = Field(gt=0)
    champion_ids: list[str] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def complete_champions(self):
        if any(not value or len(value) > 30 for value in self.champion_ids):
            raise ValueError("Chọn đủ mười tướng hợp lệ.")
        if len(set(self.champion_ids)) != 10:
            raise ValueError("Mười vị trí phải dùng mười tướng khác nhau.")
        return self
