"""Read-only display catalog; no model, temporal rules, engine setup or media I/O."""

from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import select

from match_insight.data_processing.reference_common import validate_stable_id
from match_insight.database.models import (
    Champion,
    Game,
    Player,
    Series,
    Team,
    Tournament,
    TournamentStage,
)
from match_insight.features.pre import PreFeatureInputError

CATALOG_SELECTS = (
    (Tournament, ("tournament_id", "name"), 64),
    (TournamentStage, ("stage_id", "tournament_id", "name"), 64),
    (Series, ("series_id", "stage_id"), 80),
    (Game, ("game_id", "series_id", "stage_id", "game_number"), 100),
    (Team, ("team_id", "display_name", "logo_file"), 64),
    (Player, ("player_id", "display_name", "photo_file"), 64),
    (Champion, ("champion_id", "display_name", "image_file"), 30),
)


@dataclass(frozen=True, slots=True)
class CatalogEntity:
    identity: str
    name: str
    media_file: str | None


@dataclass(frozen=True, slots=True)
class GamePresentation:
    game_id: str
    tournament_name: str
    stage_name: str
    game_number: int


@dataclass(frozen=True, slots=True)
class DemoCatalog:
    teams: tuple[CatalogEntity, ...]
    players: tuple[CatalogEntity, ...]
    champions: tuple[CatalogEntity, ...]
    games: tuple[GamePresentation, ...]


def _fail(code, detail):
    raise PreFeatureInputError(code, detail)


def _index(records, columns, maximum):
    try:
        records = tuple(records)
    except TypeError:
        _fail("E_DB_ROW_SHAPE", "Expected display catalog rows")
    indexed = {}
    for row in records:
        if not isinstance(row, Mapping) or set(row) != set(columns):
            _fail("E_DB_ROW_SHAPE", "Display catalog columns do not match the query")
        identity = row[columns[0]]
        if not isinstance(identity, str):
            _fail("E_ID_INVALID", "Catalog identity must be a stable string ID")
        try:
            canonical = validate_stable_id(identity, columns[0], maximum)
        except ValueError:
            _fail("E_ID_INVALID", "Catalog identity is invalid")
        if canonical != identity:
            _fail("E_ID_INVALID", "Catalog identity must already be canonical")
        if identity in indexed:
            _fail("E_SOURCE_CONFLICT", "Duplicate catalog identity; rows are not merged")
        indexed[identity] = dict(row)
    return indexed


def _parent(identity, rows):
    if not isinstance(identity, str) or identity not in rows:
        _fail("E_SOURCE_CONFLICT", "Display context has a missing or conflicting parent")
    return rows[identity]


def _name(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail("E_PRE_INPUT_INCOMPLETE", "Catalog display name must be nonempty")
    return value


def _entities(index, media_field):
    entities = []
    for identity, row in sorted(index.items()):
        media = row[media_field]
        if media is not None and not isinstance(media, str):
            _fail("E_SOURCE_CONFLICT", "Catalog media reference must be text or null")
        # Keep the stored value as presentation data. The UI's local-file boundary
        # checks allowed directories and file existence; this reader opens no media.
        entities.append(CatalogEntity(identity, _name(row["display_name"]), media))
    return tuple(entities)


def map_demo_catalog(rows) -> DemoCatalog:
    """Map already-read display columns without changing inference snapshot identity."""
    expected = {model.__tablename__ for model, _columns, _maximum in CATALOG_SELECTS}
    if not isinstance(rows, Mapping) or set(rows) != expected:
        _fail("E_DB_SNAPSHOT_SHAPE", "Display catalog tables are incomplete")
    indexes = {
        model.__tablename__: _index(rows[model.__tablename__], columns, maximum)
        for model, columns, maximum in CATALOG_SELECTS
    }
    for row in indexes["tournament"].values():
        _name(row["name"])
    for row in indexes["tournament_stage"].values():
        _parent(row["tournament_id"], indexes["tournament"])
        _name(row["name"])
    for row in indexes["series"].values():
        _parent(row["stage_id"], indexes["tournament_stage"])

    games = []
    for game_id, row in sorted(indexes["game"].items()):
        if (row["series_id"] is None) == (row["stage_id"] is None):
            _fail("E_SOURCE_CONFLICT", "A game must have exactly one series or stage parent")
        stage_id = row["stage_id"]
        if row["series_id"] is not None:
            stage_id = _parent(row["series_id"], indexes["series"])["stage_id"]
        stage = _parent(stage_id, indexes["tournament_stage"])
        tournament = _parent(stage["tournament_id"], indexes["tournament"])
        if type(row["game_number"]) is not int or row["game_number"] <= 0:
            _fail("E_SOURCE_CONFLICT", "Game number must be a positive integer")
        games.append(
            GamePresentation(game_id, tournament["name"], stage["name"], row["game_number"])
        )
    return DemoCatalog(
        teams=_entities(indexes["team"], "logo_file"),
        players=_entities(indexes["player"], "photo_file"),
        champions=_entities(indexes["champion"], "image_file"),
        games=tuple(games),
    )


def read_demo_catalog(engine) -> DemoCatalog:
    """Read display data in its own verified read-only REPEATABLE READ transaction."""
    if engine.dialect.name != "postgresql":
        _fail("E_DB_DIALECT", "Display catalog requires PostgreSQL")
    with engine.connect() as connection:
        connection = connection.execution_options(isolation_level="REPEATABLE READ")
        transaction = connection.begin()
        try:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            readonly = connection.exec_driver_sql("SHOW transaction_read_only").scalar_one()
            if readonly != "on":
                _fail("E_DB_NOT_READ_ONLY", "Display catalog transaction must be read-only")
            rows = {}
            for model, columns, _maximum in CATALOG_SELECTS:
                table = model.__table__
                statement = select(*(table.c[column] for column in columns)).order_by(
                    table.c[columns[0]]
                )
                rows[model.__tablename__] = connection.execute(statement).mappings().all()
            return map_demo_catalog(rows)
        finally:
            transaction.rollback()
