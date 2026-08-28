"""Kiểm thử media local và upsert reference data."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import match_insight.data_processing.champions as champion_module
import match_insight.data_processing.players as player_module
import match_insight.data_processing.reference_common as common_module
from match_insight.data_processing.champions import (
    ChampionRecord,
    sync_champions,
)
from match_insight.data_processing.players import (
    PlayerRecord,
    sync_player_media,
    sync_player_metadata,
)
from match_insight.data_processing.teams import (
    TeamRecord,
    sync_teams,
)
from match_insight.database.engine import engine
from match_insight.database.models import Champion, Player, Team


class FakeImageResponse:
    """HTTP response giả chứa image bytes."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> FakeImageResponse:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.payload

        return self.payload[:size]


@pytest.fixture
def database_session() -> Iterator[Session]:
    """Session PostgreSQL nằm trong transaction được rollback."""
    with engine.connect() as connection:
        transaction = connection.begin()
        session = Session(
            bind=connection,
            join_transaction_mode="create_savepoint",
        )

        try:
            yield session
        finally:
            session.close()

            if transaction.is_active:
                transaction.rollback()


def test_download_image_is_local_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_payload = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    request_count = 0

    def fake_urlopen(
        request: object,
        timeout: int,
    ) -> FakeImageResponse:
        nonlocal request_count
        request_count += 1

        assert request is not None
        assert timeout == 45
        return FakeImageResponse(png_payload)

    monkeypatch.setattr(
        common_module,
        "urlopen",
        fake_urlopen,
    )

    asset_dir = tmp_path / "assets" / "champions"

    first_path, first_downloaded = common_module.download_image(
        url="https://media.example/Aatrox.png",
        asset_dir=asset_dir,
        stem="266-aatrox",
        project_root=tmp_path,
    )
    second_path, second_downloaded = common_module.download_image(
        url="https://media.example/Aatrox.png",
        asset_dir=asset_dir,
        stem="266-aatrox",
        project_root=tmp_path,
    )

    assert first_path == "assets/champions/266-aatrox.png"
    assert second_path == first_path
    assert first_downloaded is True
    assert second_downloaded is False
    assert request_count == 1
    assert (tmp_path / first_path).read_bytes() == png_payload


def test_champion_upsert_does_not_duplicate_record(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    champion_id = str(uuid4().int)[:20]
    image_file = f"assets/champions/{champion_id}-test-champion.png"

    def fake_download_image(
        **arguments: object,
    ) -> tuple[str, bool]:
        assert arguments
        return image_file, True

    monkeypatch.setattr(
        champion_module,
        "download_image",
        fake_download_image,
    )

    records = [
        ChampionRecord(
            champion_id=champion_id,
            canonical_name="TestChampion",
            display_name="Test Champion",
            image_url="https://media.example/champion.png",
        )
    ]

    first_stats = sync_champions(
        session=database_session,
        records=records,
        project_root=tmp_path,
    )
    second_stats = sync_champions(
        session=database_session,
        records=records,
        project_root=tmp_path,
    )

    record_count = database_session.scalar(
        select(func.count()).select_from(Champion).where(Champion.champion_id == champion_id)
    )
    champion = database_session.get(Champion, champion_id)

    assert first_stats.inserted == 1
    assert first_stats.updated == 0
    assert first_stats.skipped == 0
    assert second_stats.inserted == 0
    assert second_stats.updated == 0
    assert second_stats.skipped == 1
    assert record_count == 1
    assert champion is not None
    assert champion.image_file == image_file


def test_team_upsert_does_not_duplicate_record(
    database_session: Session,
    tmp_path: Path,
) -> None:
    team_id = f"test_team_{uuid4().hex[:12]}"
    records = [
        TeamRecord(
            team_id=team_id,
            canonical_name="Test Team",
            display_name="Test Team",
            record_source="automated_test",
            record_source_url="https://source.example/team",
            logo_url=None,
        )
    ]

    first_stats = sync_teams(
        session=database_session,
        records=records,
        project_root=tmp_path,
    )
    second_stats = sync_teams(
        session=database_session,
        records=records,
        project_root=tmp_path,
    )

    record_count = database_session.scalar(
        select(func.count()).select_from(Team).where(Team.team_id == team_id)
    )
    team = database_session.get(Team, team_id)

    assert first_stats.inserted == 1
    assert first_stats.updated == 0
    assert first_stats.skipped == 0
    assert second_stats.inserted == 0
    assert second_stats.updated == 0
    assert second_stats.skipped == 1
    assert record_count == 1
    assert team is not None
    assert team.logo_file is None


def test_player_upsert_does_not_duplicate_record(
    database_session: Session,
) -> None:
    player_id = f"test_player_{uuid4().hex[:12]}"
    records = [
        PlayerRecord(
            player_id=player_id,
            canonical_name="Test Player",
            display_name="Test Player",
            record_source="automated_test",
            record_source_url="https://source.example/player",
            photo_url=None,
        )
    ]

    first_stats = sync_player_metadata(
        session=database_session,
        records=records,
    )
    second_stats = sync_player_metadata(
        session=database_session,
        records=records,
    )

    record_count = database_session.scalar(
        select(func.count()).select_from(Player).where(Player.player_id == player_id)
    )
    player = database_session.get(Player, player_id)

    assert first_stats.inserted == 1
    assert first_stats.updated == 0
    assert first_stats.skipped == 0
    assert second_stats.inserted == 0
    assert second_stats.updated == 0
    assert second_stats.skipped == 1
    assert record_count == 1
    assert player is not None
    assert player.photo_file is None


def test_player_media_network_failure_does_not_abort_record(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player_id = f"test_player_{uuid4().hex[:12]}"

    def failing_download_image(
        **arguments: object,
    ) -> tuple[str, bool]:
        assert arguments
        raise RuntimeError("temporary media timeout")

    monkeypatch.setattr(
        player_module,
        "download_image",
        failing_download_image,
    )

    records = [
        PlayerRecord(
            player_id=player_id,
            canonical_name="Retry Player",
            display_name="Retry Player",
            record_source="automated_test",
            record_source_url=(
                "https://source.example/player"
            ),
            photo_url=(
                "https://static.wikia.nocookie.net/"
                "example/player.png"
            ),
        )
    ]
    metadata_stats = sync_player_metadata(
        session=database_session,
        records=records,
    )
    media_stats = sync_player_media(
        session=database_session,
        records=records,
        project_root=tmp_path,
    )

    player = database_session.get(Player, player_id)

    assert metadata_stats.inserted == 1
    assert media_stats.media_downloaded == 0
    assert media_stats.media_skipped == 1
    assert player is not None
    assert player.photo_file is None


def test_player_media_resume_links_existing_local_file(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player_id = f"test_player_{uuid4().hex[:12]}"
    stem = common_module.asset_stem(
        stable_id=player_id,
        canonical_name="Resume Player",
    )
    local_image = (
        tmp_path
        / "assets"
        / "players"
        / f"{stem}.png"
    )
    local_image.parent.mkdir(parents=True)
    local_image.write_bytes(b"existing-local-image")
    photo_file = f"assets/players/{stem}.png"
    records = [
        PlayerRecord(
            player_id=player_id,
            canonical_name="Resume Player",
            display_name="Resume Player",
            record_source="automated_test",
            record_source_url=(
                "https://source.example/player"
            ),
            photo_url=(
                "https://static.wikia.nocookie.net/"
                "example/player.png"
            ),
        )
    ]

    sync_player_metadata(
        session=database_session,
        records=records,
    )

    def unexpected_network(url: str) -> bytes:
        raise AssertionError(
            f"Không được gọi network khi file đã có: {url}"
        )

    monkeypatch.setattr(
        common_module,
        "_download_image_payload",
        unexpected_network,
    )

    stats = sync_player_media(
        session=database_session,
        records=records,
        project_root=tmp_path,
    )
    player = database_session.get(Player, player_id)

    assert stats.media_downloaded == 0
    assert stats.media_skipped == 1
    assert player is not None
    assert player.photo_file == photo_file


def test_player_metadata_preserves_existing_photo_file(
    database_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player_id = f"test_player_{uuid4().hex[:12]}"
    photo_file = f"assets/players/{player_id}-existing.png"
    player = Player(
        player_id=player_id,
        canonical_name="Existing Player",
        display_name="Old Display Name",
        photo_file=photo_file,
    )
    database_session.add(player)
    database_session.flush()

    def unexpected_download(
        **arguments: object,
    ) -> tuple[str, bool]:
        raise AssertionError(
            f"Metadata phase attempted media I/O: {arguments}"
        )

    monkeypatch.setattr(
        player_module,
        "download_image",
        unexpected_download,
    )

    stats = sync_player_metadata(
        session=database_session,
        records=[
            PlayerRecord(
                player_id=player_id,
                canonical_name="Existing Player",
                display_name="New Display Name",
                record_source="automated_test",
                record_source_url=(
                    "https://source.example/player"
                ),
                photo_url=(
                    "https://static.wikia.nocookie.net/"
                    "example/player.png"
                ),
            )
        ],
    )

    assert stats.updated == 1
    assert player.display_name == "New Display Name"
    assert player.photo_file == photo_file


def test_champion_upsert_updates_changed_record(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    champion_id = str(uuid4().int)[:20]
    image_file = f"assets/champions/{champion_id}-test-champion.png"

    def fake_download_image(
        **arguments: object,
    ) -> tuple[str, bool]:
        assert arguments
        return image_file, False

    monkeypatch.setattr(
        champion_module,
        "download_image",
        fake_download_image,
    )

    original_records = [
        ChampionRecord(
            champion_id=champion_id,
            canonical_name="TestChampion",
            display_name="Original Name",
            image_url="https://media.example/champion.png",
        )
    ]
    changed_records = [
        ChampionRecord(
            champion_id=champion_id,
            canonical_name="TestChampion",
            display_name="Updated Name",
            image_url="https://media.example/champion.png",
        )
    ]

    first_stats = sync_champions(
        session=database_session,
        records=original_records,
        project_root=tmp_path,
    )
    second_stats = sync_champions(
        session=database_session,
        records=changed_records,
        project_root=tmp_path,
    )

    record_count = database_session.scalar(
        select(func.count()).select_from(Champion).where(Champion.champion_id == champion_id)
    )
    champion = database_session.get(Champion, champion_id)

    assert first_stats.inserted == 1
    assert second_stats.inserted == 0
    assert second_stats.updated == 1
    assert second_stats.skipped == 0
    assert record_count == 1
    assert champion is not None
    assert champion.display_name == "Updated Name"


def test_team_upsert_updates_changed_record(
    database_session: Session,
    tmp_path: Path,
) -> None:
    team_id = f"test_team_{uuid4().hex[:12]}"
    original_records = [
        TeamRecord(
            team_id=team_id,
            canonical_name="Test Team",
            display_name="Original Team Name",
            record_source="automated_test",
            record_source_url="https://source.example/team",
            logo_url=None,
        )
    ]
    changed_records = [
        TeamRecord(
            team_id=team_id,
            canonical_name="Test Team",
            display_name="Updated Team Name",
            record_source="automated_test",
            record_source_url="https://source.example/team",
            logo_url=None,
        )
    ]

    first_stats = sync_teams(
        session=database_session,
        records=original_records,
        project_root=tmp_path,
    )
    second_stats = sync_teams(
        session=database_session,
        records=changed_records,
        project_root=tmp_path,
    )

    record_count = database_session.scalar(
        select(func.count()).select_from(Team).where(Team.team_id == team_id)
    )
    team = database_session.get(Team, team_id)

    assert first_stats.inserted == 1
    assert second_stats.inserted == 0
    assert second_stats.updated == 1
    assert second_stats.skipped == 0
    assert record_count == 1
    assert team is not None
    assert team.display_name == "Updated Team Name"


def test_player_upsert_updates_changed_record(
    database_session: Session,
) -> None:
    player_id = f"test_player_{uuid4().hex[:12]}"
    original_records = [
        PlayerRecord(
            player_id=player_id,
            canonical_name="Test Player",
            display_name="Original Player Name",
            record_source="automated_test",
            record_source_url="https://source.example/player",
            photo_url=None,
        )
    ]
    changed_records = [
        PlayerRecord(
            player_id=player_id,
            canonical_name="Test Player",
            display_name="Updated Player Name",
            record_source="automated_test",
            record_source_url="https://source.example/player",
            photo_url=None,
        )
    ]

    first_stats = sync_player_metadata(
        session=database_session,
        records=original_records,
    )
    second_stats = sync_player_metadata(
        session=database_session,
        records=changed_records,
    )

    record_count = database_session.scalar(
        select(func.count()).select_from(Player).where(Player.player_id == player_id)
    )
    player = database_session.get(Player, player_id)

    assert first_stats.inserted == 1
    assert second_stats.inserted == 0
    assert second_stats.updated == 1
    assert second_stats.skipped == 0
    assert record_count == 1
    assert player is not None
    assert player.display_name == "Updated Player Name"


def test_champion_media_failure_does_not_commit_partial_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier_seed = str(uuid4().int)
    first_id = f"{identifier_seed[:18]}1"
    second_id = f"{identifier_seed[:18]}2"
    download_count = 0

    def failing_download_image(
        **arguments: object,
    ) -> tuple[str, bool]:
        nonlocal download_count
        download_count += 1

        assert arguments

        if download_count == 2:
            raise ValueError("Ảnh champion test không hợp lệ.")

        return (
            f"assets/champions/{first_id}-first-champion.png",
            True,
        )

    monkeypatch.setattr(
        champion_module,
        "download_image",
        failing_download_image,
    )

    records = [
        ChampionRecord(
            champion_id=first_id,
            canonical_name="FirstChampion",
            display_name="First Champion",
            image_url="https://media.example/first.png",
        ),
        ChampionRecord(
            champion_id=second_id,
            canonical_name="SecondChampion",
            display_name="Second Champion",
            image_url="https://media.example/second.png",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Ảnh champion test không hợp lệ",
    ):
        with Session(engine) as session:
            sync_champions(
                session=session,
                records=records,
                project_root=tmp_path,
            )

    with Session(engine) as verification_session:
        record_count = verification_session.scalar(
            select(func.count())
            .select_from(Champion)
            .where(Champion.champion_id.in_([first_id, second_id]))
        )

    assert download_count == 2
    assert record_count == 0


def test_download_image_retries_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_payload = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    request_count = 0
    sleep_delays: list[int] = []

    def flaky_urlopen(
        request: object,
        timeout: int,
    ) -> FakeImageResponse:
        nonlocal request_count
        request_count += 1

        assert request is not None
        assert timeout == 45

        if request_count < 3:
            raise TimeoutError("Mạng test tạm thời chậm.")

        return FakeImageResponse(png_payload)

    def fake_sleep(seconds: int) -> None:
        sleep_delays.append(seconds)

    monkeypatch.setattr(
        common_module,
        "urlopen",
        flaky_urlopen,
    )
    monkeypatch.setattr(
        common_module.time,
        "sleep",
        fake_sleep,
    )

    relative_path, downloaded = common_module.download_image(
        url="https://media.example/retry.png",
        asset_dir=tmp_path / "assets" / "champions",
        stem="266-retry-champion",
        project_root=tmp_path,
    )

    assert downloaded is True
    assert relative_path == ("assets/champions/266-retry-champion.png")
    assert request_count == 3
    assert sleep_delays == [1, 2]
    assert (tmp_path / relative_path).is_file()


def test_download_image_retries_fandom_403_with_long_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_payload = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    media_url = (
        "https://lol.fandom.com/wiki/"
        "Special:Redirect/file/Testlogo_square.png"
    )
    request_count = 0
    sleep_delays: list[int] = []
    expected_path = (
        tmp_path
        / "assets"
        / "teams"
        / "lp_team_1-test-team.png"
    )

    def flaky_urlopen(
        request: object,
        timeout: int,
    ) -> FakeImageResponse:
        nonlocal request_count
        request_count += 1

        assert request is not None
        assert timeout == 45

        if request_count < 3:
            raise common_module.HTTPError(
                media_url,
                403,
                "Forbidden",
                None,
                None,
            )

        return FakeImageResponse(png_payload)

    def fake_sleep(seconds: int) -> None:
        if seconds == 1:
            assert expected_path.is_file()

        sleep_delays.append(seconds)

    monkeypatch.setattr(
        common_module,
        "urlopen",
        flaky_urlopen,
    )
    monkeypatch.setattr(
        common_module.time,
        "sleep",
        fake_sleep,
    )

    relative_path, downloaded = common_module.download_image(
        url=media_url,
        asset_dir=tmp_path / "assets" / "teams",
        stem="lp_team_1-test-team",
        project_root=tmp_path,
    )

    assert downloaded is True
    assert relative_path == (
        "assets/teams/lp_team_1-test-team.png"
    )
    assert request_count == 3
    assert sleep_delays == [30, 60, 1]
    assert (tmp_path / relative_path).is_file()


def test_download_image_does_not_retry_non_fandom_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_url = "https://media.example/forbidden.png"
    request_count = 0
    sleep_delays: list[int] = []

    def forbidden_urlopen(
        request: object,
        timeout: int,
    ) -> FakeImageResponse:
        nonlocal request_count
        request_count += 1

        assert request is not None
        assert timeout == 45

        raise common_module.HTTPError(
            media_url,
            403,
            "Forbidden",
            None,
            None,
        )

    monkeypatch.setattr(
        common_module,
        "urlopen",
        forbidden_urlopen,
    )
    monkeypatch.setattr(
        common_module.time,
        "sleep",
        sleep_delays.append,
    )

    with pytest.raises(
        RuntimeError,
        match="HTTP 403",
    ):
        common_module._download_image_payload(media_url)

    assert request_count == 1
    assert sleep_delays == []
