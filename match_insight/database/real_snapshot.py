"""Read-only PostgreSQL snapshot and pure mapping; no engine import."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime

from sqlalchemy import select

from match_insight.database.models import (
    Champion,
    Game,
    GamePatch,
    GamePlayer,
    GameTeam,
    Player,
    Series,
    Team,
    Tournament,
    TournamentStage,
)
from match_insight.features.post import (
    ChampionSlot,
    FinalLineup,
    HistoricalChampionGame,
    _require_reference,
    _require_slots,
)
from match_insight.features.pre import (
    ROLES,
    HistoricalGame,
    PlayerSlot,
    PreFeatureInputError,
    _as_utc,
    _require_id,
    _require_lineups,
)
from match_insight.ml.models import _digest

# First selected column is the table's primary key.
SELECTS = (
    (Tournament, ("tournament_id",), 64),
    (TournamentStage, ("stage_id", "tournament_id"), 64),
    (Series, ("series_id", "stage_id"), 80),
    (GamePatch, ("patch_id", "patch_name"), 20),
    (Team, ("team_id",), 64),
    (Player, ("player_id",), 64),
    (Champion, ("champion_id",), 30),
    (
        Game,
        (
            "game_id",
            "series_id",
            "stage_id",
            "patch_id",
            "started_at",
            "ended_at",
            "winner_team_id",
        ),
        100,
    ),
    (GameTeam, ("game_team_id", "game_id", "team_id", "side"), None),
    (
        GamePlayer,
        ("game_player_id", "game_team_id", "player_id", "role", "champion_id"),
        None,
    ),
)


class RealDataInputError(PreFeatureInputError):
    """Sanitized integration error with a stable reason code."""


def fail(code, game_id=None):
    raise RealDataInputError(code, "Real-data integration gate rejected input", game_id)


@dataclass(frozen=True, slots=True)
class DatabaseGame:
    """Postgame DB context; this is not an approved PRE target."""

    game: HistoricalGame
    final_lineup: FinalLineup
    started_at: datetime | None


@dataclass(frozen=True, slots=True)
class RealSnapshot:
    source_ref: str
    captured_at: datetime
    sha256: str
    games: tuple
    champion_reference: frozenset
    pre_context_verification: str = "UNVERIFIED"

    @property
    def history(self):
        return tuple(
            HistoricalChampionGame(item.game, item.final_lineup.slots)
            for item in self.games
        )


def pre_context_sha256(item):
    """Identify context only; this function does not verify PRE availability."""
    game = item.game
    return _digest(
        (
            game.game_id,
            item.final_lineup.patch,
            ("BLUE", game.blue_team_id, game.blue_roster),
            ("RED", game.red_team_id, game.red_roster),
        )
    )


def _require_db_id(value, maximum):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        fail("E_DB_ID_INVALID")


def _index(records, columns, max_length):
    result = {}
    key = columns[0]
    for record in records:
        if not isinstance(record, Mapping) or set(record) != set(columns):
            fail("E_DB_ROW_SHAPE")
        value = record[key]
        if max_length is None:
            if type(value) is not int or value <= 0:
                fail("E_DB_ID_INVALID")
        else:
            _require_db_id(value, max_length)
        if value in result:
            fail("E_DB_DUPLICATE_ID")
        result[value] = dict(record)
    return result


def _parent(value, index):
    if type(value) not in (str, int) or value not in index:
        fail("E_DB_PARENT")
    return index[value]


def map_snapshot(rows, captured_at, source_ref="postgresql:official-project"):
    """Validate an already-read snapshot without I/O or implicit PRE cutoffs."""
    if (
        not isinstance(source_ref, str)
        or not source_ref
        or source_ref != source_ref.strip()
        or len(source_ref) > 100
    ):
        fail("E_DB_SOURCE_REF_INVALID")
    captured_at = _as_utc(captured_at, "snapshot_read_at", "snapshot")

    names = {model.__tablename__ for model, _, _ in SELECTS}
    if not isinstance(rows, Mapping) or set(rows) != names:
        fail("E_DB_SNAPSHOT_SHAPE")

    indexes = {
        model.__tablename__: _index(rows[model.__tablename__], columns, length)
        for model, columns, length in SELECTS
    }

    for row in indexes["tournament_stage"].values():
        _parent(row["tournament_id"], indexes["tournament"])
    for row in indexes["series"].values():
        _parent(row["stage_id"], indexes["tournament_stage"])

    teams_by_game = defaultdict(list)
    for row in indexes["game_team"].values():
        _parent(row["game_id"], indexes["game"])
        _parent(row["team_id"], indexes["team"])
        teams_by_game[row["game_id"]].append(row)

    players_by_parent = defaultdict(list)
    for row in indexes["game_player"].values():
        _parent(row["game_team_id"], indexes["game_team"])
        _parent(row["player_id"], indexes["player"])
        _parent(row["champion_id"], indexes["champion"])
        players_by_parent[row["game_team_id"]].append(row)

    reference = frozenset(indexes["champion"])
    if indexes["game"]:
        _require_reference(reference)

    games = []
    for game_id, row in sorted(indexes["game"].items()):
        if (row["series_id"] is None) == (row["stage_id"] is None):
            fail("E_DB_GAME_PARENT", game_id)
        if row["series_id"] is not None:
            _parent(row["series_id"], indexes["series"])
        else:
            _parent(row["stage_id"], indexes["tournament_stage"])

        patch = _parent(row["patch_id"], indexes["game_patch"])["patch_name"]
        if (
            not isinstance(patch, str)
            or not patch
            or patch != patch.strip()
            or len(patch) > 20
        ):
            fail("E_DB_PATCH_INVALID", game_id)

        participants = teams_by_game[game_id]
        if (
            len(participants) != 2
            or {part["side"] for part in participants} != {"BLUE", "RED"}
            or len({part["team_id"] for part in participants}) != 2
        ):
            fail("E_DB_GAME_SIDES", game_id)
        sides = {part["side"]: part for part in participants}

        if row["winner_team_id"] not in {part["team_id"] for part in participants}:
            fail("E_DB_WINNER", game_id)

        start = row["started_at"]
        end = row["ended_at"]
        if start is not None:
            start = _as_utc(start, "started_at", game_id)
        if end is not None:
            end = _as_utc(end, "ended_at", game_id)
        if (start is None) != (end is None):
            fail("E_DB_TIME_PAIR", game_id)
        if start is not None and start >= end:
            fail("E_DB_TIME_ORDER", game_id)

        rosters = {}
        slots = []
        for side in ("BLUE", "RED"):
            participant = sides[side]
            members = players_by_parent[participant["game_team_id"]]
            rosters[side] = tuple(
                PlayerSlot(member["role"], member["player_id"])
                for member in members
            )
            slots.extend(
                ChampionSlot(
                    team_id=participant["team_id"],
                    side=side,
                    role=member["role"],
                    player_id=member["player_id"],
                    champion_id=member["champion_id"],
                )
                for member in members
            )

        game = HistoricalGame(
            game_id=game_id,
            blue_team_id=sides["BLUE"]["team_id"],
            red_team_id=sides["RED"]["team_id"],
            blue_roster=rosters["BLUE"],
            red_roster=rosters["RED"],
            winner_team_id=row["winner_team_id"],
            ended_at=end,
        )
        _require_id(game_id, "game_id", 100)
        _require_lineups(game)
        game = replace(
            game,
            blue_roster=tuple(
                sorted(game.blue_roster, key=lambda slot: ROLES.index(slot.role))
            ),
            red_roster=tuple(
                sorted(game.red_roster, key=lambda slot: ROLES.index(slot.role))
            ),
        )
        final_slots = _require_slots(tuple(slots), game, reference)
        games.append(DatabaseGame(game, FinalLineup(game_id, patch, final_slots), start))

    canonical_rows = {
        name: [index[key] for key in sorted(index)]
        for name, index in sorted(indexes.items())
    }
    return RealSnapshot(
        source_ref=source_ref,
        captured_at=captured_at,
        sha256=_digest((source_ref, canonical_rows)),
        games=tuple(games),
        champion_reference=reference,
    )


def read_snapshot(engine, source_ref="postgresql:official-project"):
    """Read all required tables in one verified read-only transaction."""
    if engine.dialect.name != "postgresql":
        fail("E_DB_DIALECT")

    with engine.connect() as connection:
        connection = connection.execution_options(isolation_level="REPEATABLE READ")
        transaction = connection.begin()
        try:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            readonly = connection.exec_driver_sql(
                "SHOW transaction_read_only"
            ).scalar_one()
            if readonly != "on":
                fail("E_DB_NOT_READ_ONLY")

            captured_at = connection.exec_driver_sql(
                "SELECT transaction_timestamp()"
            ).scalar_one()
            rows = {}
            for model, columns, _ in SELECTS:
                table = model.__table__
                statement = select(*(table.c[name] for name in columns)).order_by(
                    table.c[columns[0]]
                )
                rows[model.__tablename__] = (
                    connection.execute(statement).mappings().all()
                )
            return map_snapshot(rows, captured_at, source_ref)
        finally:
            transaction.rollback()
