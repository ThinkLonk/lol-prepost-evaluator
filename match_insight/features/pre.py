"""PRE features thuần; không I/O, database, clock hoặc model."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from match_insight.data_processing.reference_common import validate_stable_id

ROLES = ("TOP", "JUNGLE", "MID", "BOT", "SUPPORT")
Side = Literal["BLUE", "RED"]


class PreFeatureInputError(ValueError):
    """Lỗi contract; không trả feature một phần khi đầu vào xung đột."""

    def __init__(
        self,
        code: str,
        detail: str,
        game_id: str | None = None,
    ) -> None:
        self.code = code
        self.game_id = game_id
        context = f" [{game_id}]" if game_id is not None else ""
        super().__init__(f"{code}{context}: {detail}")


@dataclass(frozen=True, slots=True)
class PlayerSlot:
    role: str
    player_id: str


@dataclass(frozen=True, slots=True)
class TargetGame:
    """Roster do caller cung cấp trong ngữ cảnh PRE."""

    game_id: str
    blue_team_id: str
    red_team_id: str
    blue_roster: tuple[PlayerSlot, ...]
    red_roster: tuple[PlayerSlot, ...]
    patch: str
    history_cutoff_at: datetime


@dataclass(frozen=True, slots=True)
class HistoricalGame:
    """Một record là một game; không chứa champion hoặc thống kê trong trận."""

    game_id: str
    blue_team_id: str
    red_team_id: str
    blue_roster: tuple[PlayerSlot, ...]
    red_roster: tuple[PlayerSlot, ...]
    winner_team_id: str
    ended_at: datetime | None


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    recent_form_games: int = 10
    side_win_rate_games: int = 20
    head_to_head_games: int = 10


@dataclass(frozen=True, slots=True)
class FeatureStat:
    value: float | None
    sample_count: int
    missing: bool
    game_ids: tuple[str, ...]
    # Chỉ áp dụng cho ba nhóm tỷ lệ thắng.
    win_count: int | None = None


@dataclass(frozen=True, slots=True)
class TeamFeatures:
    team_id: str
    side: Side
    recent_form: FeatureStat
    side_win_rate: FeatureStat
    roster_continuity: FeatureStat


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    game_id: str
    patch: str
    history_cutoff_at: datetime
    config_version: str
    config: FeatureConfig
    cutoff_verification: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"


@dataclass(frozen=True, slots=True)
class HistoryExclusion:
    game_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class HistoryCounts:
    input_games: int
    accepted_games: int
    excluded_games: int
    missing_end_games: int


@dataclass(frozen=True, slots=True)
class HistorySelection:
    """Context và lịch sử bất biến đã qua validator/temporal filter chung."""

    target: TargetGame
    games: tuple[HistoricalGame, ...]
    exclusions: tuple[HistoryExclusion, ...]
    counts: HistoryCounts


@dataclass(frozen=True, slots=True)
class PreFeatureResult:
    metadata: FeatureMetadata
    blue: TeamFeatures
    red: TeamFeatures
    h2h_blue: FeatureStat
    accepted_game_ids: tuple[str, ...]
    exclusions: tuple[HistoryExclusion, ...]
    counts: HistoryCounts
    # Kết quả cũ vẫn đọc được, nhưng POST yêu cầu snapshot được builder tạo.
    history_snapshot: HistorySelection | None = None


def _require_id(
    value: object,
    field: str,
    maximum: int,
    game_id: str | None = None,
) -> None:
    if not isinstance(value, str):
        raise PreFeatureInputError(
            "E_ID_INVALID",
            f"{field} must be a stable string ID",
            game_id,
        )

    try:
        normalized = validate_stable_id(
            value,
            field_name=field,
            maximum=maximum,
        )
    except ValueError:
        raise PreFeatureInputError(
            "E_ID_INVALID",
            f"{field} violates the existing stable ID contract",
            game_id,
        ) from None

    if normalized != value:
        raise PreFeatureInputError(
            "E_ID_INVALID",
            f"{field} must already be canonical; IDs are not rewritten",
            game_id,
        )


def _require_lineups(game: TargetGame | HistoricalGame) -> None:
    _require_id(game.blue_team_id, "blue_team_id", 64, game.game_id)
    _require_id(game.red_team_id, "red_team_id", 64, game.game_id)

    if game.blue_team_id == game.red_team_id:
        raise PreFeatureInputError(
            "E_TEAMS_INVALID",
            "BLUE and RED team IDs must differ",
            game.game_id,
        )

    all_players: list[str] = []

    for side, roster in (
        ("BLUE", game.blue_roster),
        ("RED", game.red_roster),
    ):
        if not isinstance(roster, tuple) or len(roster) != 5:
            raise PreFeatureInputError(
                "E_LINEUP_INVALID",
                f"{side} roster must be a tuple containing five player slots",
                game.game_id,
            )

        roles: list[str] = []
        for slot in roster:
            if not isinstance(slot, PlayerSlot):
                raise PreFeatureInputError(
                    "E_LINEUP_INVALID",
                    f"{side} roster contains a non-PlayerSlot value",
                    game.game_id,
                )

            if not isinstance(slot.role, str) or slot.role not in ROLES:
                raise PreFeatureInputError(
                    "E_LINEUP_INVALID",
                    f"{side} roster contains an unsupported role",
                    game.game_id,
                )

            _require_id(slot.player_id, "player_id", 64, game.game_id)
            roles.append(slot.role)
            all_players.append(slot.player_id)

        if Counter(roles) != Counter(ROLES):
            raise PreFeatureInputError(
                "E_LINEUP_INVALID",
                f"{side} roster must contain each required role exactly once",
                game.game_id,
            )

    if len(set(all_players)) != 10:
        raise PreFeatureInputError(
            "E_LINEUP_INVALID",
            "The game must contain ten distinct player IDs",
            game.game_id,
        )


def _as_utc(
    value: object,
    field: str,
    game_id: str,
) -> datetime:
    if not isinstance(value, datetime):
        raise PreFeatureInputError(
            "E_TIMESTAMP_INVALID",
            f"{field} must be an aware datetime; strings and epochs are not parsed",
            game_id,
        )

    try:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime")
        return value.astimezone(UTC)
    except (OverflowError, TypeError, ValueError):
        raise PreFeatureInputError(
            "E_TIMESTAMP_INVALID",
            f"{field} requires an explicit, valid timezone",
            game_id,
        ) from None


def _validate_config(config: FeatureConfig) -> None:
    if not isinstance(config, FeatureConfig):
        raise PreFeatureInputError(
            "E_FEATURE_CONFIG_INVALID",
            "config must be FeatureConfig",
        )

    for field, value in (
        ("recent_form_games", config.recent_form_games),
        ("side_win_rate_games", config.side_win_rate_games),
        ("head_to_head_games", config.head_to_head_games),
    ):
        if type(value) is not int or value <= 0:
            raise PreFeatureInputError(
                "E_FEATURE_CONFIG_INVALID",
                f"{field} must be a positive Python int, not bool",
            )


def _rate(
    games: tuple[HistoricalGame, ...],
    team_id: str,
) -> FeatureStat:
    count = len(games)
    wins = sum(game.winner_team_id == team_id for game in games)
    return FeatureStat(
        value=wins / count if count else None,
        sample_count=count,
        missing=count == 0,
        game_ids=tuple(game.game_id for game in games),
        win_count=wins,
    )


def _continuity(
    current_roster: tuple[PlayerSlot, ...],
    team_id: str,
    team_games: tuple[HistoricalGame, ...],
) -> FeatureStat:
    """Theo Thiết Kế: chỉ so với ván gần nhất của cùng đội."""
    if not team_games:
        return FeatureStat(
            value=None,
            sample_count=0,
            missing=True,
            game_ids=(),
        )

    previous = team_games[0]
    previous_roster = (
        previous.blue_roster
        if previous.blue_team_id == team_id
        else previous.red_roster
    )
    current_ids = {slot.player_id for slot in current_roster}
    previous_ids = {slot.player_id for slot in previous_roster}

    return FeatureStat(
        value=len(current_ids & previous_ids) / 5,
        sample_count=1,
        missing=False,
        game_ids=(previous.game_id,),
    )


def _team_features(
    team_id: str,
    side: Side,
    current_roster: tuple[PlayerSlot, ...],
    games: tuple[HistoricalGame, ...],
    config: FeatureConfig,
) -> TeamFeatures:
    team_games = tuple(
        game
        for game in games
        if team_id in (game.blue_team_id, game.red_team_id)
    )
    side_games = tuple(
        game
        for game in games
        if (
            game.blue_team_id if side == "BLUE" else game.red_team_id
        ) == team_id
    )

    return TeamFeatures(
        team_id=team_id,
        side=side,
        recent_form=_rate(team_games[:config.recent_form_games], team_id),
        side_win_rate=_rate(side_games[:config.side_win_rate_games], team_id),
        roster_continuity=_continuity(current_roster, team_id, team_games),
    )


def select_pre_history(
    target: TargetGame,
    history: Iterable[HistoricalGame],
) -> HistorySelection:
    """Validator và temporal filter dùng chung cho PRE và POST.

    Không xác minh tính trung thực của nguồn timestamp.
    Duplicate game ID chặn batch, kể cả hai record giống nhau.
    Target bị loại trước khi đọc outcome/lineup/end của record đó.
    """
    if not isinstance(target, TargetGame):
        raise PreFeatureInputError(
            "E_TARGET_INVALID",
            "target must be TargetGame",
        )

    _require_id(target.game_id, "game_id", 100)
    _require_lineups(target)

    if (
        not isinstance(target.patch, str)
        or not target.patch.strip()
        or len(target.patch) > 20
    ):
        raise PreFeatureInputError(
            "E_PATCH_INVALID",
            "patch must be a nonempty string of at most 20 characters",
            target.game_id,
        )

    if target.history_cutoff_at is None:
        raise PreFeatureInputError(
            "E_CUTOFF_MISSING",
            "history_cutoff_at must be supplied explicitly",
            target.game_id,
        )

    cutoff = _as_utc(
        target.history_cutoff_at,
        "history_cutoff_at",
        target.game_id,
    )

    try:
        records = tuple(history)
    except TypeError:
        raise PreFeatureInputError(
            "E_HISTORY_INPUT_INVALID",
            "history must be an iterable of HistoricalGame",
        ) from None

    for record in records:
        if not isinstance(record, HistoricalGame):
            raise PreFeatureInputError(
                "E_HISTORY_INPUT_INVALID",
                "Each history record must be HistoricalGame",
            )
        _require_id(record.game_id, "game_id", 100)

    duplicate_ids = sorted(
        game_id
        for game_id, count in Counter(
            record.game_id for record in records
        ).items()
        if count > 1
    )
    if duplicate_ids:
        raise PreFeatureInputError(
            "E_HISTORY_DUPLICATE_GAME_ID",
            "Duplicate historical IDs: " + ", ".join(duplicate_ids),
        )

    accepted: list[tuple[HistoricalGame, datetime]] = []
    exclusions: list[HistoryExclusion] = []

    for record in sorted(records, key=lambda item: item.game_id):
        if record.game_id == target.game_id:
            exclusions.append(
                HistoryExclusion(record.game_id, "TARGET_GAME")
            )
            continue

        # Non-null invalid timestamps remain errors, including future records.
        end = (
            None
            if record.ended_at is None
            else _as_utc(record.ended_at, "ended_at", record.game_id)
        )

        _require_lineups(record)
        _require_id(
            record.winner_team_id,
            "winner_team_id",
            64,
            record.game_id,
        )
        if record.winner_team_id not in (
            record.blue_team_id,
            record.red_team_id,
        ):
            raise PreFeatureInputError(
                "E_HISTORY_WINNER_INVALID",
                "winner_team_id must identify a participating team",
                record.game_id,
            )

        if end is None:
            exclusions.append(
                HistoryExclusion(record.game_id, "MISSING_ENDED_AT")
            )
        elif end >= cutoff:
            exclusions.append(
                HistoryExclusion(record.game_id, "END_NOT_BEFORE_CUTOFF")
            )
        else:
            accepted.append((record, end))

    # Stable sort retains lexical game_id order for equal UTC instants.
    accepted.sort(key=lambda item: item[1], reverse=True)
    games = tuple(record for record, _end in accepted)

    return HistorySelection(
        target=replace(target, history_cutoff_at=cutoff),
        games=games,
        exclusions=tuple(exclusions),
        counts=HistoryCounts(
            input_games=len(records),
            accepted_games=len(games),
            excluded_games=len(exclusions),
            missing_end_games=sum(
                item.reason == "MISSING_ENDED_AT"
                for item in exclusions
            ),
        ),
    )


def build_pre_features(
    target: TargetGame,
    history: Iterable[HistoricalGame],
    config: FeatureConfig = FeatureConfig(),
) -> PreFeatureResult:
    """Tính bốn nhóm PRE và giữ snapshot cho POST.

    Không chứng nhận nguồn cutoff/end hoặc tạo evaluation chính thức.
    Công thức và cấu hình feature giữ nguyên hành vi của Bước 7G.
    """
    _validate_config(config)
    selection = select_pre_history(target, history)
    target = selection.target
    games = selection.games

    target_teams = {target.blue_team_id, target.red_team_id}
    h2h_games = tuple(
        game
        for game in games
        if {game.blue_team_id, game.red_team_id} == target_teams
    )[:config.head_to_head_games]

    version = (
        f"pre-v1-r{config.recent_form_games}"
        f"-s{config.side_win_rate_games}"
        f"-h{config.head_to_head_games}-roster-latest1"
    )

    return PreFeatureResult(
        metadata=FeatureMetadata(
            game_id=target.game_id,
            patch=target.patch,
            history_cutoff_at=target.history_cutoff_at,
            config_version=version,
            config=config,
        ),
        blue=_team_features(
            target.blue_team_id,
            "BLUE",
            target.blue_roster,
            games,
            config,
        ),
        red=_team_features(
            target.red_team_id,
            "RED",
            target.red_roster,
            games,
            config,
        ),
        h2h_blue=_rate(h2h_games, target.blue_team_id),
        accepted_game_ids=tuple(game.game_id for game in games),
        exclusions=selection.exclusions,
        counts=selection.counts,
        history_snapshot=selection,
    )
