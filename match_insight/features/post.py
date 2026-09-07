"""POST features thuần, giữ nguyên PRE và lịch sử đã chốt.

Không I/O, database, clock, network hoặc model.
Champion IDs là categorical identities, không phải số có thứ tự.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from match_insight.data_processing.reference_common import validate_stable_id
from match_insight.features.pre import (
    ROLES,
    HistoricalGame,
    HistoryCounts,
    HistoryExclusion,
    HistorySelection,
    PreFeatureInputError,
    PreFeatureResult,
    Side,
    TargetGame,
    select_pre_history,
)


class PostFeatureInputError(PreFeatureInputError):
    """Lỗi POST hoặc không tương thích với PRE đã cung cấp."""


@dataclass(frozen=True, slots=True)
class ChampionSlot:
    team_id: str
    side: Side
    role: str
    player_id: str
    champion_id: str


@dataclass(frozen=True, slots=True)
class FinalLineup:
    game_id: str
    patch: str
    slots: tuple[ChampionSlot, ...]


@dataclass(frozen=True, slots=True)
class HistoricalChampionGame:
    """Bổ sung mapping champion cho đúng HistoricalGame của PRE."""

    game: HistoricalGame
    slots: tuple[ChampionSlot, ...]


@dataclass(frozen=True, slots=True)
class PlayerChampionFeature:
    team_id: str
    side: Side
    role: str
    player_id: str
    champion_id: str
    games_count: int
    wins_count: int
    win_rate: float | None
    missing: bool
    game_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PostFeatureMetadata:
    game_id: str
    patch: str
    history_cutoff_at: datetime
    pre_config_version: str
    config_version: str = "post-v1-player-champion-all-pre-history"
    history_scope: Literal["ALL_PRE_ACCEPTED_GAMES"] = "ALL_PRE_ACCEPTED_GAMES"
    cutoff_verification: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"


@dataclass(frozen=True, slots=True)
class PostFeatureResult:
    pre: PreFeatureResult
    metadata: PostFeatureMetadata
    player_champion: tuple[PlayerChampionFeature, ...]
    accepted_game_ids: tuple[str, ...]
    exclusions: tuple[HistoryExclusion, ...]
    counts: HistoryCounts


def _is_stable_id(value: object, field: str, maximum: int) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return validate_stable_id(
            value,
            field_name=field,
            maximum=maximum,
        ) == value
    except ValueError:
        return False


def _require_reference(champion_reference: frozenset[str]) -> None:
    if not isinstance(champion_reference, frozenset) or not champion_reference:
        raise PostFeatureInputError(
            "E_CHAMPION_REFERENCE_INVALID",
            "champion_reference must be a nonempty frozenset of stable string IDs",
        )

    if any(
        not _is_stable_id(champion_id, "champion_id", 30)
        for champion_id in champion_reference
    ):
        raise PostFeatureInputError(
            "E_CHAMPION_REFERENCE_INVALID",
            "Every reference ID must follow the existing champion ID contract",
        )


def _require_pre_context(pre: PreFeatureResult) -> TargetGame:
    if (
        not isinstance(pre, PreFeatureResult)
        or not isinstance(pre.history_snapshot, HistorySelection)
    ):
        raise PostFeatureInputError(
            "E_PRE_SNAPSHOT_REQUIRED",
            "POST requires a PRE result containing its bound history snapshot",
        )

    snapshot = pre.history_snapshot
    context = snapshot.target
    if not isinstance(context, TargetGame):
        raise PostFeatureInputError(
            "E_EVAL_INCOMPATIBLE",
            "PRE snapshot has an invalid target context",
        )

    if (
        pre.metadata.game_id != context.game_id
        or pre.metadata.patch != context.patch
        or pre.metadata.history_cutoff_at != context.history_cutoff_at
        or pre.blue.team_id != context.blue_team_id
        or pre.blue.side != "BLUE"
        or pre.red.team_id != context.red_team_id
        or pre.red.side != "RED"
        or pre.accepted_game_ids != tuple(game.game_id for game in snapshot.games)
        or pre.exclusions != snapshot.exclusions
        or pre.counts != snapshot.counts
    ):
        raise PostFeatureInputError(
            "E_EVAL_INCOMPATIBLE",
            "PRE metadata/context/history snapshot are inconsistent",
            context.game_id,
        )

    return context


def _require_slots(
    slots: tuple[ChampionSlot, ...],
    game: TargetGame | HistoricalGame,
    champion_reference: frozenset[str],
) -> tuple[ChampionSlot, ...]:
    if not isinstance(slots, tuple) or len(slots) != 10:
        raise PostFeatureInputError(
            "E_LINEUP_INCOMPLETE",
            "Champion lineup must be a tuple containing exactly ten slots",
            game.game_id,
        )

    side_roles: set[tuple[str, str]] = set()
    players: set[str] = set()
    champions: set[str] = set()

    for slot in slots:
        if not isinstance(slot, ChampionSlot):
            raise PostFeatureInputError(
                "E_POST_LINEUP_INVALID",
                "Every lineup entry must be ChampionSlot",
                game.game_id,
            )

        if (
            not isinstance(slot.side, str)
            or slot.side not in ("BLUE", "RED")
            or not isinstance(slot.role, str)
            or slot.role not in ROLES
        ):
            raise PostFeatureInputError(
                "E_POST_LINEUP_INVALID",
                "Lineup contains an unsupported side or role",
                game.game_id,
            )

        if (
            not _is_stable_id(slot.team_id, "team_id", 64)
            or not _is_stable_id(slot.player_id, "player_id", 64)
        ):
            raise PostFeatureInputError(
                "E_ID_INVALID",
                "Team and player IDs must be canonical stable string IDs",
                game.game_id,
            )

        if not _is_stable_id(slot.champion_id, "champion_id", 30):
            raise PostFeatureInputError(
                "E_CHAMPION_ID_INVALID",
                "Champion ID is missing or violates the existing ID contract",
                game.game_id,
            )

        if slot.champion_id not in champion_reference:
            raise PostFeatureInputError(
                "E_CHAMPION_UNKNOWN",
                "Champion ID is absent from the supplied reference",
                game.game_id,
            )

        key = (slot.side, slot.role)
        if (
            key in side_roles
            or slot.player_id in players
            or slot.champion_id in champions
        ):
            raise PostFeatureInputError(
                "E_SOURCE_CONFLICT",
                "Duplicate side-role, player ID or champion ID within the game",
                game.game_id,
            )

        side_roles.add(key)
        players.add(slot.player_id)
        champions.add(slot.champion_id)

    expected = {
        (side, player.role): (team_id, player.player_id)
        for side, team_id, roster in (
            ("BLUE", game.blue_team_id, game.blue_roster),
            ("RED", game.red_team_id, game.red_roster),
        )
        for player in roster
    }

    if side_roles != set(expected):
        raise PostFeatureInputError(
            "E_LINEUP_INCOMPLETE",
            "Each side must contain exactly TOP/JUNGLE/MID/BOT/SUPPORT",
            game.game_id,
        )

    for slot in slots:
        if expected[(slot.side, slot.role)] != (slot.team_id, slot.player_id):
            raise PostFeatureInputError(
                "E_EVAL_INCOMPATIBLE",
                "Champion lineup does not match the bound team/side/role/player",
                game.game_id,
            )

    return tuple(
        sorted(
            slots,
            key=lambda slot: (
                0 if slot.side == "BLUE" else 1,
                ROLES.index(slot.role),
            ),
        )
    )


def build_post_features(
    pre: PreFeatureResult,
    final_lineup: FinalLineup,
    history: Iterable[HistoricalChampionGame],
    champion_reference: frozenset[str],
) -> PostFeatureResult:
    """Bổ sung player-champion experience vào đúng PRE đã cung cấp.

    Caller cung cấp reference và lịch sử chuyên nghiệp đã xác minh bên ngoài.
    Hàm kiểm tra contract, không chứng nhận provenance hoặc temporal readiness.

    Chỉ dùng tập HistoricalGame đã chốt trong PRE. Mapping champion phải đầy đủ
    cho mọi game được chấp nhận; thiếu mapping là lỗi, không phải zero samples.
    Các game bị temporal filter loại không cần mapping champion.
    """
    context = _require_pre_context(pre)
    _require_reference(champion_reference)

    if not isinstance(final_lineup, FinalLineup):
        raise PostFeatureInputError(
            "E_POST_INPUT_INVALID",
            "final_lineup must be FinalLineup",
            context.game_id,
        )

    if (
        final_lineup.game_id != context.game_id
        or final_lineup.patch != context.patch
    ):
        raise PostFeatureInputError(
            "E_EVAL_INCOMPATIBLE",
            "Final lineup game or patch differs from PRE",
            context.game_id,
        )

    final_slots = _require_slots(
        final_lineup.slots,
        context,
        champion_reference,
    )

    try:
        records = tuple(history)
    except TypeError:
        raise PostFeatureInputError(
            "E_HISTORY_INPUT_INVALID",
            "history must be an iterable of HistoricalChampionGame",
        ) from None

    if any(not isinstance(record, HistoricalChampionGame) for record in records):
        raise PostFeatureInputError(
            "E_HISTORY_INPUT_INVALID",
            "Every POST history record must be HistoricalChampionGame",
        )

    # Reuse exactly the PRE validator and temporal filter.
    selection = select_pre_history(
        context,
        (record.game for record in records),
    )

    snapshot = pre.history_snapshot
    if snapshot is None or selection.games != snapshot.games:
        raise PostFeatureInputError(
            "E_PRE_HISTORY_MISMATCH",
            "Eligible historical games differ from the snapshot bound to PRE",
            context.game_id,
        )

    # The shared filter has already rejected duplicate IDs across the batch.
    by_game_id = {record.game.game_id: record for record in records}
    accepted = tuple(
        (
            game,
            _require_slots(
                by_game_id[game.game_id].slots,
                game,
                champion_reference,
            ),
        )
        for game in selection.games
    )

    features: list[PlayerChampionFeature] = []
    for target_slot in final_slots:
        matches = tuple(
            (game, historical_slot)
            for game, slots in accepted
            for historical_slot in slots
            if (
                historical_slot.player_id == target_slot.player_id
                and historical_slot.champion_id == target_slot.champion_id
            )
        )

        games_count = len(matches)
        wins_count = sum(
            historical_slot.team_id == game.winner_team_id
            for game, historical_slot in matches
        )

        features.append(
            PlayerChampionFeature(
                team_id=target_slot.team_id,
                side=target_slot.side,
                role=target_slot.role,
                player_id=target_slot.player_id,
                champion_id=target_slot.champion_id,
                games_count=games_count,
                wins_count=wins_count,
                win_rate=wins_count / games_count if games_count else None,
                missing=games_count == 0,
                game_ids=tuple(game.game_id for game, _slot in matches),
            )
        )

    return PostFeatureResult(
        pre=pre,
        metadata=PostFeatureMetadata(
            game_id=pre.metadata.game_id,
            patch=pre.metadata.patch,
            history_cutoff_at=pre.metadata.history_cutoff_at,
            pre_config_version=pre.metadata.config_version,
            cutoff_verification=pre.metadata.cutoff_verification,
        ),
        player_champion=tuple(features),
        accepted_game_ids=pre.accepted_game_ids,
        exclusions=selection.exclusions,
        counts=selection.counts,
    )
