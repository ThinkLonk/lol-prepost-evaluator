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
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
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
    region: Mapped[str | None] = mapped_column(String(30), nullable=True)
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
    __table_args__ = (
        UniqueConstraint(
            "oracle_team_id",
            name="uq_team_oracle_team_id",
        ),
    )

    team_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    oracle_team_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    canonical_name: Mapped[str] = mapped_column(String(150), nullable=False)
    display_name: Mapped[str] = mapped_column(String(150), nullable=False)
    logo_file: Mapped[str | None] = mapped_column(String(160),nullable=True,)


class Player(Base):
    """Tuyển thủ chuyên nghiệp."""

    __tablename__ = "player"
    __table_args__ = (
        UniqueConstraint(
            "oracle_player_id",
            name="uq_player_oracle_player_id",
        ),
    )

    player_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    oracle_player_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    canonical_name: Mapped[str] = mapped_column(String(120), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    photo_file: Mapped[str | None] = mapped_column(String(160),nullable=True,)


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
            "("
            "series_id IS NOT NULL AND stage_id IS NULL"
            ") OR ("
            "series_id IS NULL AND stage_id IS NOT NULL"
            ")",
            name="ck_game_exactly_one_parent",
        ),
        ForeignKeyConstraint(
            ["game_id", "winner_team_id"],
            ["game_team.game_id", "game_team.team_id"],
            name="fk_game_winner_participant",
            deferrable=True,
            initially="DEFERRED",
            use_alter=True,
        ),
        Index("ix_game_scheduled_at", "scheduled_at"),
    )

    game_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    series_id: Mapped[str | None] = mapped_column(
        String(80),
        ForeignKey("series.series_id"),
        nullable=True,
    )
    stage_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey(
            "tournament_stage.stage_id",
            name="fk_game_stage",
        ),
        nullable=True,
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


class AnalysisSession(Base):
    """Independent user analysis; creation records first persistence, not PRE time."""

    __tablename__ = "analysis_session"

    analysis_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


class EvaluationHistory(Base):
    """Immutable, content-addressed history shared by persisted PRE/POST snapshots."""

    __tablename__ = "evaluation_history"
    __table_args__ = (
        CheckConstraint("history_sha256 ~ '^[0-9a-f]{64}$'", name="ck_evaluation_history_hash"),
        CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_evaluation_history_payload"),
    )

    history_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class Evaluation(Base):
    """Bản đánh giá PRE hoặc POST bất biến của một ván."""

    __tablename__ = "evaluation"
    __table_args__ = (
        CheckConstraint(
            "(game_id IS NULL AND analysis_id IS NOT NULL "
            "AND inference_mode = 'INTERACTIVE_ANALYSIS' AND origin = 'INTERACTIVE') OR "
            "(game_id IS NOT NULL AND analysis_id IS NULL "
            "AND inference_mode = 'REAL_RETROSPECTIVE_SIMULATION' "
            "AND origin = 'RETROSPECTIVE_IMPORT')",
            name="ck_evaluation_subject",
        ),
        CheckConstraint("context_key ~ '^[0-9a-f]{64}$'", name="ck_evaluation_context_key"),
        CheckConstraint("idempotency_key ~ '^[0-9a-f]{64}$'", name="ck_evaluation_idempotency_key"),
        CheckConstraint("jsonb_typeof(provenance) = 'object'", name="ck_evaluation_provenance"),
        CheckConstraint("jsonb_typeof(input_snapshot) = 'object'", name="ck_evaluation_snapshot"),
        UniqueConstraint("idempotency_key", name="uq_evaluation_idempotency_key"),
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
        Index("ix_evaluation_analysis_type", "analysis_id", "evaluation_type"),
        Index(
            "uq_evaluation_one_active_analysis_model",
            "analysis_id", "evaluation_type", "model_version",
            unique=True,
            postgresql_where=text("is_active IS TRUE AND analysis_id IS NOT NULL"),
        ),
    )

    evaluation_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    game_id: Mapped[str | None] = mapped_column(
        String(100),
        ForeignKey("game.game_id"),
        nullable=True,
    )
    analysis_id: Mapped[str | None] = mapped_column(
        String(100), ForeignKey("analysis_session.analysis_id"), nullable=True,
    )
    inference_mode: Mapped[str] = mapped_column(String(40), nullable=False)
    origin: Mapped[str] = mapped_column(String(32), nullable=False)
    context_key: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    inferred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provenance: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    history_sha256: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluation_history.history_sha256"), nullable=False,
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
    data_version: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
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
    __table_args__ = (
        UniqueConstraint("evaluation_id", "position", name="uq_evaluation_warning_position"),
        CheckConstraint("position >= 0", name="ck_evaluation_warning_position"),
        CheckConstraint("jsonb_typeof(details) = 'object'", name="ck_evaluation_warning_details"),
    )

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
    warning_code: Mapped[str] = mapped_column(String(50), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    details: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
