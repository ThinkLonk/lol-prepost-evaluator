"""Atomic PostgreSQL evaluation storage; no inference or source-data updates."""

from dataclasses import dataclass
from datetime import UTC

from sqlalchemy import insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from match_insight.database.models import (
    AnalysisSession,
    Champion,
    Evaluation,
    EvaluationHistory,
    EvaluationWarning,
    GamePlayer,
    GameTeam,
    Player,
    Team,
)
from match_insight.features.pre import PreFeatureInputError


class StorageError(PreFeatureInputError):
    """Sanitized storage failure, safe for presentation."""


@dataclass(frozen=True)
class SavedEvaluation:
    evaluation_id: int
    created: bool
    is_active: bool


def _fail(detail):
    raise StorageError("E_STORAGE_CONFLICT", detail)


def _lock(connection, subject, model):
    # All writers and invalidation use the same transaction-scoped subject lock.
    connection.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                       {"key": f"evaluation:{subject}:{model}"})


def _validate_references(connection, packet):
    target = packet["input_snapshot"]["target"]
    teams = {target["blue_team_id"], target["red_team_id"]}
    players = {slot["player_id"] for side in ("blue_roster", "red_roster")
               for slot in target[side]}
    for table, column, wanted in ((Team, Team.team_id, teams),
                                  (Player, Player.player_id, players)):
        actual = set(connection.execute(select(column).where(column.in_(wanted))).scalars())
        if actual != wanted:
            _fail(f"Missing {table.__tablename__} identities")
    lineup = packet["input_snapshot"].get("lineup")
    if lineup is not None:
        wanted = {slot["champion_id"] for slot in lineup["slots"]}
        actual = set(connection.execute(select(Champion.champion_id).where(
            Champion.champion_id.in_(wanted))).scalars())
        if actual != wanted:
            _fail("Missing champion identities")
    if packet["game_id"] is not None:
        rows = connection.execute(select(GameTeam.side, GameTeam.team_id,
                                         GamePlayer.role, GamePlayer.player_id).join(
            GamePlayer, GamePlayer.game_team_id == GameTeam.game_team_id
        ).where(GameTeam.game_id == packet["game_id"])).all()
        expected = {(side, target[f"{side.lower()}_team_id"], slot["role"], slot["player_id"])
                    for side in ("BLUE", "RED") for slot in target[f"{side.lower()}_roster"]}
        if len(rows) != 10 or set(rows) != expected:
            _fail("Historical target context differs from PostgreSQL participation")


def _subject_filter(packet):
    column = Evaluation.analysis_id if packet["analysis_id"] else Evaluation.game_id
    return column == (packet["analysis_id"] or packet["game_id"])


class EvaluationRepository:
    """A connection exists only inside a save/read/invalidation transaction."""

    def __init__(self, engine):
        self.engine = engine
        self._committed_histories = set()

    def _save(self, connection, packet, history, warnings, pre_evaluation_id=None):
        subject = packet["analysis_id"] or packet["game_id"]
        _lock(connection, subject, packet["model_version"])
        values = {**packet, "pre_evaluation_id": pre_evaluation_id}
        existing = connection.execute(select(Evaluation.__table__).where(
            Evaluation.idempotency_key == values["idempotency_key"])).mappings().first()
        if existing is not None:
            # Never reactivate an obsolete result, replace its snapshot or its warnings.
            for key, value in values.items():
                if existing[key] != value:
                    _fail("An existing request has different immutable content")
            stored_warnings = connection.execute(select(
                EvaluationWarning.warning_code, EvaluationWarning.warning_group,
                EvaluationWarning.message, EvaluationWarning.severity, EvaluationWarning.details,
            ).where(EvaluationWarning.evaluation_id == existing["evaluation_id"])
                .order_by(EvaluationWarning.position)).mappings().all()
            if [dict(row) for row in stored_warnings] != warnings:
                _fail("Existing warnings differ from the immutable evaluation")
            return SavedEvaluation(existing["evaluation_id"], False, existing["is_active"])

        _validate_references(connection, values)
        if values["history_sha256"] not in self._committed_histories:
            stored_history = connection.execute(select(EvaluationHistory.payload).where(
                EvaluationHistory.history_sha256 == values["history_sha256"])).scalar_one_or_none()
            if stored_history is None:
                connection.execute(pg_insert(EvaluationHistory).values(
                    history_sha256=values["history_sha256"], payload=history,
                ).on_conflict_do_nothing(index_elements=["history_sha256"]))
                stored_history = connection.execute(select(EvaluationHistory.payload).where(
                    EvaluationHistory.history_sha256 == values["history_sha256"])).scalar_one()
            if stored_history != history:
                _fail("History identity has conflicting immutable content")
        if values["analysis_id"] is not None:
            connection.execute(pg_insert(AnalysisSession).values(
                analysis_id=values["analysis_id"],
            ).on_conflict_do_nothing(index_elements=["analysis_id"]))
        active = (_subject_filter(values), Evaluation.model_version == values["model_version"],
                  Evaluation.is_active.is_(True))
        if values["evaluation_type"] == "PRE":
            # New PRE supersedes both phases in this analysis/model; old rows remain immutable.
            connection.execute(update(Evaluation).where(*active).values(is_active=False))
        else:
            parent = connection.execute(select(Evaluation.__table__).where(
                Evaluation.evaluation_id == pre_evaluation_id).with_for_update()).mappings().first()
            keys = ("game_id", "analysis_id", "inference_mode", "context_key", "history_cutoff_at",
                    "model_version", "data_version", "history_sha256")
            if (parent is None or parent["evaluation_type"] != "PRE" or not parent["is_active"]
                    or any(parent[key] != values[key] for key in keys)):
                _fail("POST requires the active compatible PRE")
            connection.execute(update(Evaluation).where(
                *active, Evaluation.evaluation_type == "POST").values(is_active=False))
        evaluation_id = connection.execute(insert(Evaluation).values(**values).returning(
            Evaluation.evaluation_id)).scalar_one()
        for position, warning in enumerate(warnings):
            connection.execute(insert(EvaluationWarning).values(
                evaluation_id=evaluation_id, position=position, **warning,
            ))
        return SavedEvaluation(evaluation_id, True, True)

    def save(self, packet, history, warnings, *, pre_evaluation_id=None):
        try:
            with self.engine.begin() as connection:
                result = self._save(connection, packet, history, warnings, pre_evaluation_id)
            self._committed_histories.add(packet["history_sha256"])
            return result  # Only expose an ID after successful COMMIT.
        except SQLAlchemyError:
            raise StorageError("E_STORAGE_WRITE", "Không lưu được đánh giá; transaction đã hủy.") \
                from None

    def save_pair(self, pre, post, history, pre_warnings, post_warnings):
        try:
            with self.engine.begin() as connection:
                saved_pre = self._save(connection, pre, history, pre_warnings)
                saved_post = self._save(connection, post, history, post_warnings,
                                        saved_pre.evaluation_id)
            self._committed_histories.add(pre["history_sha256"])
            return saved_pre, saved_post
        except SQLAlchemyError:
            raise StorageError("E_STORAGE_WRITE", "Không lưu được cặp PRE/POST; transaction đã hủy.") \
                from None

    def invalidate(self, analysis_id, model_version, *, post_only=False):
        try:
            with self.engine.begin() as connection:
                _lock(connection, analysis_id, model_version)
                statement = update(Evaluation).where(
                    Evaluation.analysis_id == analysis_id,
                    Evaluation.model_version == model_version,
                    Evaluation.is_active.is_(True),
                )
                if post_only:
                    statement = statement.where(Evaluation.evaluation_type == "POST")
                connection.execute(statement.values(is_active=False))
        except SQLAlchemyError:
            raise StorageError("E_STORAGE_WRITE", "Chưa cập nhật được hiệu lực đánh giá đã lưu.") \
                from None

    def read(self, evaluation_id):
        if type(evaluation_id) is not int or evaluation_id <= 0:
            _fail("Evaluation ID must be a positive integer")
        try:
            with self.engine.connect() as connection, connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                row = connection.execute(select(Evaluation.__table__).where(
                    Evaluation.evaluation_id == evaluation_id)).mappings().first()
                if row is None:
                    raise StorageError("E_STORAGE_READ", "Không tìm thấy đánh giá đã lưu.")
                result = dict(row)
                result["warnings"] = [dict(value) for value in connection.execute(select(
                    EvaluationWarning.warning_code, EvaluationWarning.warning_group,
                    EvaluationWarning.message, EvaluationWarning.severity, EvaluationWarning.details,
                ).where(EvaluationWarning.evaluation_id == evaluation_id)
                    .order_by(EvaluationWarning.position)).mappings()]
                result["paired_post_ids"] = list(connection.execute(select(
                    Evaluation.evaluation_id).where(Evaluation.pre_evaluation_id == evaluation_id)
                    .order_by(Evaluation.evaluation_id)).scalars())
                # The complete history is stored separately, not inferred from its hash.
                result["history_payload"] = connection.execute(select(EvaluationHistory.payload)
                    .where(EvaluationHistory.history_sha256 == row["history_sha256"])).scalar_one()
                for key in ("created_at", "inferred_at", "history_cutoff_at"):
                    result[key] = (None if result[key] is None
                                   else result[key].astimezone(UTC).isoformat())
                return result
        except SQLAlchemyError:
            raise StorageError("E_STORAGE_READ", "Không đọc được đánh giá từ PostgreSQL.") from None
