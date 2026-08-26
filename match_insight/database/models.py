"""Các SQLAlchemy ORM model của schema PostgreSQL cốt lõi."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Identity,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from match_insight.database.base import Base


class Tournament(Base):
    """Giải đấu chuyên nghiệp."""

    __tablename__ = "tournament"

    tournament_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    region: Mapped[str] = mapped_column(String(30), nullable=False)
    season: Mapped[int] = mapped_column(SmallInteger, nullable=False)


class TournamentStage(Base):
    """Giai đoạn thuộc một giải đấu."""

    __tablename__ = "tournament_stage"

    stage_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tournament_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tournament.tournament_id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)


class Series(Base):
    """Loạt trận BO1, BO3 hoặc BO5."""

    __tablename__ = "series"
    __table_args__ = (
        CheckConstraint(
            "best_of IN (1, 3, 5)",
            name="ck_series_best_of",
        ),
    )

    series_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    stage_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tournament_stage.stage_id"),
        nullable=False,
    )
    best_of: Mapped[int] = mapped_column(SmallInteger, nullable=False)


class GamePatch(Base):
    """Phiên bản thi đấu của một ván."""

    __tablename__ = "game_patch"

    patch_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    patch_name: Mapped[str] = mapped_column(String(20), nullable=False)


class Team(Base):
    """Đội tuyển chuyên nghiệp."""

    __tablename__ = "team"

    team_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(150), nullable=False)
    display_name: Mapped[str] = mapped_column(String(150), nullable=False)


class Player(Base):
    """Tuyển thủ chuyên nghiệp."""

    __tablename__ = "player"

    player_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(120), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)


class Champion(Base):
    """Tướng Liên Minh Huyền Thoại."""

    __tablename__ = "champion"

    champion_id: Mapped[str] = mapped_column(String(30), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(80), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    image_file: Mapped[str | None] = mapped_column(String(160), nullable=True)


class Game(Base):
    """Một ván đấu, là đơn vị xử lý cốt lõi của hệ thống."""

    __tablename__ = "game"
    __table_args__ = (
        CheckConstraint(
            "winner_team_id IS NULL OR ended_at IS NOT NULL",
            name="ck_game_winner_requires_end",
        ),
        Index("ix_game_scheduled_at", "scheduled_at"),
    )

    game_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    series_id: Mapped[str] = mapped_column(
        String(80),
        ForeignKey("series.series_id"),
        nullable=False,
    )
    patch_id: Mapped[str] = mapped_column(
        String(20),
        ForeignKey("game_patch.patch_id"),
        nullable=False,
    )
    game_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    winner_team_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("team.team_id"),
        nullable=True,
    )


class TeamMembership(Base):
    """Quan hệ đội–tuyển thủ có khoảng thời gian hiệu lực."""

    __tablename__ = "team_membership"
    __table_args__ = (
        CheckConstraint(
            "role IN ('TOP', 'JUNGLE', 'MID', 'BOT', 'SUPPORT')",
            name="ck_team_membership_role",
        ),
        Index(
            "ix_team_membership_team_validity",
            "team_id",
            "valid_from",
            "valid_to",
        ),
        Index(
            "ix_team_membership_player_validity",
            "player_id",
            "valid_from",
            "valid_to",
        ),
    )

    membership_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    team_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("team.team_id"),
        nullable=False,
    )
    player_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("player.player_id"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class GameTeam(Base):
    """Một đội tham gia một ván ở bên BLUE hoặc RED."""

    __tablename__ = "game_team"
    __table_args__ = (
        UniqueConstraint(
            "game_id",
            "side",
            name="uq_game_team_game_side",
        ),
        UniqueConstraint(
            "game_id",
            "team_id",
            name="uq_game_team_game_team",
        ),
        CheckConstraint(
            "side IN ('BLUE', 'RED')",
            name="ck_game_team_side",
        ),
        Index("ix_game_team_team_id", "team_id"),
    )

    game_team_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    game_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("game.game_id"),
        nullable=False,
    )
    team_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("team.team_id"),
        nullable=False,
    )
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    confirmation_status: Mapped[str] = mapped_column(String(20), nullable=False)


class GamePlayer(Base):
    """Một tuyển thủ tham gia ván ở một vị trí xác định."""

    __tablename__ = "game_player"
    __table_args__ = (
        UniqueConstraint(
            "game_team_id",
            "role",
            name="uq_game_player_game_team_role",
        ),
        CheckConstraint(
            "role IN ('TOP', 'JUNGLE', 'MID', 'BOT', 'SUPPORT')",
            name="ck_game_player_role",
        ),
        Index("ix_game_player_player_id", "player_id"),
        Index("ix_game_player_champion_id", "champion_id"),
    )

    game_player_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    game_team_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("game_team.game_team_id"),
        nullable=False,
    )
    player_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("player.player_id"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    champion_id: Mapped[str | None] = mapped_column(
        String(30),
        ForeignKey("champion.champion_id"),
        nullable=True,
    )
    confirmation_status: Mapped[str] = mapped_column(String(20), nullable=False)


class Evaluation(Base):
    """Bản đánh giá PRE hoặc POST bất biến của một ván."""

    __tablename__ = "evaluation"
    __table_args__ = (
        CheckConstraint(
            "evaluation_type IN ('PRE', 'POST')",
            name="ck_evaluation_type",
        ),
        CheckConstraint(
            "blue_win_probability >= 0 AND blue_win_probability <= 1",
            name="ck_evaluation_probability",
        ),
        CheckConstraint(
            "("
            "evaluation_type = 'PRE' AND pre_evaluation_id IS NULL"
            ") OR ("
            "evaluation_type = 'POST' AND pre_evaluation_id IS NOT NULL"
            ")",
            name="ck_evaluation_reference_shape",
        ),
        Index(
            "ix_evaluation_game_type",
            "game_id",
            "evaluation_type",
        ),
        Index(
            "uq_evaluation_one_active_model",
            "game_id",
            "evaluation_type",
            "model_version",
            unique=True,
            postgresql_where=text("is_active IS TRUE"),
        ),
    )

    evaluation_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    game_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("game.game_id"),
        nullable=False,
    )
    evaluation_type: Mapped[str] = mapped_column(String(4), nullable=False)
    blue_win_probability: Mapped[float] = mapped_column(Double, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    history_cutoff_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    data_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    input_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
    )
    pre_evaluation_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("evaluation.evaluation_id"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=true(),
    )

    pre_evaluation: Mapped[Evaluation | None] = relationship(
        remote_side=[evaluation_id],
        foreign_keys=[pre_evaluation_id],
    )


class EvaluationWarning(Base):
    """Cảnh báo gắn với một evaluation đã được tạo."""

    __tablename__ = "evaluation_warning"

    warning_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    evaluation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("evaluation.evaluation_id"),
        nullable=False,
    )
    warning_group: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)