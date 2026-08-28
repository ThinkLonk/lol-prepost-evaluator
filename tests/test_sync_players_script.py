"""Kiểm thử điều phối nguồn của sync_players CLI."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.sync_players as sync_script
from match_insight.data_processing.players import (
    PlayerMediaSyncResult,
    PlayerPruneResult,
    PlayerRecord,
    PlayerSelectionResult,
    PlayerSelectionStats,
    load_players_csv,
    write_players_csv,
)
from match_insight.data_processing.reference_common import (
    SyncStats,
)


def _sample_player_record() -> PlayerRecord:
    return PlayerRecord(
        player_id="lp_player_123",
        canonical_name="Test Player",
        display_name="Test Player",
        record_source="leaguepedia_cargo",
        record_source_url=(
            "https://lol.fandom.com/wiki/"
            "Special:Redirect/page/123"
        ),
        photo_url=None,
    )


def _sample_selection(
    records: list[PlayerRecord] | None = None,
    *,
    media_urls_are_direct: bool = False,
) -> PlayerSelectionResult:
    selected_records = (
        [_sample_player_record()]
        if records is None
        else records
    )
    return PlayerSelectionResult(
        records=selected_records,
        stats=PlayerSelectionStats(
            catalog_total=len(selected_records),
            included=len(selected_records),
            included_by_game=len(selected_records),
            included_by_roster_only=0,
            excluded_personality=0,
            excluded_without_player_evidence=0,
            duplicate_handle_records=0,
            media_available=sum(
                record.photo_url is not None
                for record in selected_records
            ),
        ),
        media_urls_are_direct=media_urls_are_direct,
    )


def test_prepare_records_from_leaguepedia(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = (
        tmp_path / "reference" / "players.csv"
    )
    raw_output_path = (
        tmp_path
        / "raw"
        / "leaguepedia"
        / "players.json"
    )
    expected_record = _sample_player_record()
    observed_snapshot_path: Path | None = None
    observed_staging_path: Path | None = None

    def fake_fetch_players(
        snapshot_path: Path | None = None,
    ) -> PlayerSelectionResult:
        nonlocal observed_snapshot_path
        observed_snapshot_path = snapshot_path
        return _sample_selection([expected_record])

    def fake_write_players_csv(
        path: Path,
        records: list[PlayerRecord],
    ) -> None:
        nonlocal observed_staging_path
        observed_staging_path = path
        assert records == [expected_record]

    def fake_file_sha256(
        path: Path,
    ) -> str:
        if path == raw_output_path:
            return "raw-hash"

        if path == input_path:
            return "staging-hash"

        raise AssertionError(
            f"Unexpected path: {path}"
        )

    monkeypatch.setattr(
        sync_script,
        "fetch_players",
        fake_fetch_players,
    )
    monkeypatch.setattr(
        sync_script,
        "write_players_csv",
        fake_write_players_csv,
    )
    monkeypatch.setattr(
        sync_script,
        "file_sha256",
        fake_file_sha256,
    )

    records, source = sync_script.prepare_records(
        source="leaguepedia",
        input_path=input_path,
        raw_output_path=raw_output_path,
    )

    assert records == [expected_record]
    assert observed_snapshot_path == raw_output_path
    assert observed_staging_path == input_path
    assert source == {
        "type": "leaguepedia_cargo",
        "api_url": sync_script.LEAGUEPEDIA_API_URL,
        "client": sync_script.LEAGUEPEDIA_CLIENT_NAME,
        "media_url_resolution": (
            "mediawiki_imageinfo"
        ),
        "catalog_table": "Players",
        "evidence_tables": [
            "PlayerLeagueHistory",
            "TournamentPlayers",
            "ListplayerCurrent",
        ],
        "media_table": "PlayerImages",
        "selection": {
            "catalog_total": 1,
            "included": 1,
            "included_by_game": 1,
            "included_by_roster_only": 0,
            "excluded_personality": 0,
            "excluded_without_player_evidence": 0,
            "duplicate_handle_records": 0,
            "media_available": 0,
            "media_selected_by_manual_override": 0,
            "media_selected_by_playerimages_sort_date": 0,
            "media_selected_by_tournament_start_fuzzy": 0,
            "media_selected_by_tournament_date": 0,
            "media_selected_by_effective_sort_date": 0,
            "media_selected_by_filename_date": 0,
            "media_selected_by_revision_timestamp": 0,
            "media_selected_without_date": 0,
        },
        "raw_snapshot": str(raw_output_path),
        "raw_sha256": "raw-hash",
        "staging_file": str(input_path),
        "staging_sha256": "staging-hash",
        "record_count": 1,
    }


def test_prepare_records_from_csv(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "players.csv"
    raw_output_path = tmp_path / "unused.json"
    expected_record = _sample_player_record()

    write_players_csv(
        path=input_path,
        records=[expected_record],
    )

    records, source = sync_script.prepare_records(
        source="csv",
        input_path=input_path,
        raw_output_path=raw_output_path,
    )

    assert records == [expected_record]
    assert source["type"] == "curated_csv"
    assert source["input_file"] == str(input_path)
    assert source["media_url_resolution"] == (
        "mediawiki_imageinfo"
    )
    assert source["record_count"] == 1
    assert isinstance(
        source["input_sha256"],
        str,
    )


def test_prepare_records_from_browser_snapshot(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "reference" / "players.csv"
    raw_output_path = tmp_path / "raw" / "players.json"
    raw_output_path.parent.mkdir(parents=True)
    raw_output_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": {
                    "catalog_table": "Players",
                    "evidence_tables": [
                        "PlayerLeagueHistory",
                        "TournamentPlayers",
                        "ListplayerCurrent",
                    ],
                    "media_table": "PlayerImages",
                },
                "media_urls_are_direct": True,
                "catalog_row_count": 1,
                "game_evidence_row_count": 1,
                "roster_row_count": 0,
                "image_row_count": 0,
                "catalog_rows": [
                    {
                        "SourcePageId": "123",
                        "SourcePageName": "Test Player",
                        "PlayerHandle": "Test Player",
                        "Image": "Test.png",
                        "PhotoUrl": (
                            "https://static.wikia.nocookie.net/"
                            "example/Test.png"
                        ),
                        "IsPersonality": "0",
                    }
                ],
                "game_evidence_rows": [
                    {
                        "PlayerPage": "Test Player",
                        "Role": "Mid",
                    }
                ],
                "roster_rows": [],
                "image_rows": [],
            }
        ),
        encoding="utf-8",
    )
    records, source = sync_script.prepare_records(
        source="snapshot",
        input_path=input_path,
        raw_output_path=raw_output_path,
    )

    assert len(records) == 1
    assert records[0].photo_url == (
        "https://static.wikia.nocookie.net/"
        "example/Test.png"
    )
    assert input_path.is_file()
    assert source["type"] == (
        "leaguepedia_browser_snapshot"
    )
    assert source["client"] == "browser_fetch"
    assert source["media_url_resolution"] == (
        "browser_mediawiki_imageinfo"
    )
    assert source["selection"]["included"] == 1
    assert source["raw_snapshot"] == str(
        raw_output_path
    )
    assert source["record_count"] == 1


def test_prepare_records_rejects_empty_leaguepedia(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_players(
        snapshot_path: Path | None = None,
    ) -> PlayerSelectionResult:
        assert snapshot_path is not None
        return _sample_selection([])

    monkeypatch.setattr(
        sync_script,
        "fetch_players",
        fake_fetch_players,
    )

    with pytest.raises(
        RuntimeError,
        match="không trả tuyển thủ đủ điều kiện",
    ):
        sync_script.prepare_records(
            source="leaguepedia",
            input_path=tmp_path / "players.csv",
            raw_output_path=tmp_path / "players.json",
        )


def test_prepare_records_rejects_missing_csv(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Không tìm thấy player input file",
    ):
        sync_script.prepare_records(
            source="csv",
            input_path=tmp_path / "missing.csv",
            raw_output_path=tmp_path / "unused.json",
        )


def test_prepare_records_rejects_empty_csv(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "players.csv"
    input_path.write_text(
        (
            "player_id,canonical_name,display_name,"
            "record_source,record_source_url,photo_url\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="không có player record",
    ):
        sync_script.prepare_records(
            source="csv",
            input_path=input_path,
            raw_output_path=tmp_path / "unused.json",
        )


def test_main_commits_metadata_and_completed_batches_before_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        PlayerRecord(
            player_id=f"lp_player_{index}",
            canonical_name=f"Player {index}",
            display_name=f"Player {index}",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                f"Special:Redirect/page/{index}"
            ),
            photo_url=(
                "https://static.wikia.nocookie.net/"
                f"example/player-{index}.png"
            ),
        )
        for index in range(3)
    ]
    events: list[str] = []
    manifest_source: dict[str, object] | None = None
    manifest_stats: SyncStats | None = None

    class FakeSession:
        def __enter__(self) -> "FakeSession":
            events.append("session_open")
            return self

        def __exit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            events.append("session_close")

        def commit(self) -> None:
            events.append("commit")

    def fake_metadata(
        session: FakeSession,
        records: list[PlayerRecord],
    ) -> SyncStats:
        assert session is not None
        assert len(records) == 3
        events.append("metadata")
        return SyncStats(inserted=3)

    media_call_count = 0

    def fake_media(
        session: FakeSession,
        records: list[PlayerRecord],
        project_root: Path,
        refresh_media: bool,
    ) -> SyncStats:
        nonlocal media_call_count
        assert session is not None
        assert project_root == tmp_path
        assert refresh_media is False
        assert events.index("commit") < events.index(
            "session_open",
            2,
        )
        media_call_count += 1

        if media_call_count == 2:
            raise KeyboardInterrupt

        assert len(records) == 2
        events.append("media_batch")
        return SyncStats(media_skipped=2)

    def fake_manifest(
        asset_dir: Path,
        source: dict[str, object],
        stats: SyncStats,
    ) -> None:
        nonlocal manifest_source, manifest_stats
        assert asset_dir == tmp_path / "assets" / "players"
        manifest_source = source
        manifest_stats = stats
        events.append("manifest")

    def fake_prune(
        session: FakeSession,
        keep_player_ids: set[str],
    ) -> PlayerPruneResult:
        assert session is not None
        assert keep_player_ids == {
            record.player_id for record in records
        }
        events.append("prune")
        return PlayerPruneResult(
            deleted=1,
            protected=0,
            photo_files=(
                "assets/players/lp_player_stale.png",
            ),
        )

    def fake_delete_media(
        project_root: Path,
        relative_paths: tuple[str, ...],
    ) -> int:
        assert project_root == tmp_path
        assert relative_paths == (
            "assets/players/lp_player_stale.png",
        )
        events.append("delete_media")
        return 1

    args = SimpleNamespace(
        source="snapshot",
        input=tmp_path / "players.csv",
        raw_output=tmp_path / "players.json",
        refresh_media=False,
        audit_only=False,
        prune_missing=True,
        prune_media=True,
        print_browser_exporter=False,
        media_refresh_plan=tmp_path / "media-plan.json",
    )

    monkeypatch.setattr(sync_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sync_script, "PLAYER_MEDIA_BATCH_SIZE", 2)
    monkeypatch.setattr(sync_script, "parse_args", lambda: args)
    monkeypatch.setattr(
        sync_script,
        "prepare_records",
        lambda **arguments: (records, {"type": "test"}),
    )
    monkeypatch.setattr(
        sync_script,
        "Session",
        lambda engine: FakeSession(),
    )
    monkeypatch.setattr(
        sync_script,
        "sync_player_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        sync_script,
        "sync_player_media",
        fake_media,
    )
    monkeypatch.setattr(
        sync_script,
        "prune_missing_leaguepedia_players",
        fake_prune,
    )
    monkeypatch.setattr(
        sync_script,
        "delete_pruned_player_media",
        fake_delete_media,
    )
    monkeypatch.setattr(
        sync_script,
        "write_sync_manifest",
        fake_manifest,
    )
    monkeypatch.setattr(
        sync_script,
        "print_stats",
        lambda **arguments: None,
    )

    with pytest.raises(SystemExit) as raised:
        sync_script.main()

    assert raised.value.code == 130
    assert events.count("commit") == 2
    assert events.index("metadata") < events.index("commit")
    assert events.index("commit") < events.index("media_batch")
    assert "delete_media" in events
    assert events[-1] == "manifest"
    assert manifest_source is not None
    assert manifest_source["media_sync"] == {
        "status": "interrupted",
        "batch_size": 2,
        "total": 3,
        "processed": 2,
    }
    assert manifest_stats is not None
    assert manifest_stats.inserted == 3
    assert manifest_stats.media_downloaded == 0
    assert manifest_stats.media_skipped == 2
    assert manifest_source["cleanup"] == {
        "prune_missing": True,
        "prune_media": True,
        "deleted": 1,
        "protected": 0,
        "media_deleted": 1,
    }


def test_changed_media_finalization_resumes_after_manifest_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "data" / "reference" / "players.csv"
    target_path = baseline_path.with_name("players.next.csv")
    raw_path = tmp_path / "data" / "raw" / "players.json"
    plan_path = tmp_path / "assets" / "players" / "_plan.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text('{"snapshot": 1}', encoding="utf-8")
    baseline_record = _sample_player_record()
    target_record = PlayerRecord(
        player_id=baseline_record.player_id,
        canonical_name=baseline_record.canonical_name,
        display_name=baseline_record.display_name,
        record_source=baseline_record.record_source,
        record_source_url=baseline_record.record_source_url,
        photo_url="https://static.wikia.nocookie.net/example/new.png",
    )
    write_players_csv(baseline_path, [baseline_record])
    write_players_csv(target_path, [target_record])

    monkeypatch.setattr(sync_script, "PROJECT_ROOT", tmp_path)
    plan = sync_script._create_media_refresh_plan(
        plan_path=plan_path,
        baseline_path=baseline_path,
        target_path=target_path,
        raw_snapshot_path=raw_path,
        records=[target_record],
        source_details={"type": "test"},
    )
    assert plan["changed_count"] == 1
    baseline_before = baseline_path.read_bytes()

    args = SimpleNamespace(
        source="snapshot",
        input=baseline_path,
        raw_output=raw_path,
        refresh_media=False,
        refresh_changed_media=True,
        refresh_image_sort_dates=False,
        image_sort_output=tmp_path / "sort.json",
        media_refresh_plan=plan_path,
        audit_only=False,
        prune_missing=False,
        prune_media=False,
        print_browser_exporter=False,
    )

    class FakeSession:
        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            return None

        def commit(self) -> None:
            return None

    media_calls = 0

    def fake_media(**arguments: object) -> PlayerMediaSyncResult:
        nonlocal media_calls
        media_calls += 1
        assert arguments["records"] == [target_record]
        assert arguments["refresh_media"] is True
        return PlayerMediaSyncResult(
            media_downloaded=1,
            succeeded_player_ids=(target_record.player_id,),
        )

    manifest_calls = 0

    def flaky_manifest(**arguments: object) -> None:
        nonlocal manifest_calls
        manifest_calls += 1
        source = arguments["source"]
        assert isinstance(source, dict)
        archive_path = Path(
            source["media_selection_diff"]["refresh_plan"]
        )
        archived = json.loads(archive_path.read_text(encoding="utf-8"))
        assert archived["status"] == "complete"

        if manifest_calls == 1:
            raise RuntimeError("simulated manifest failure")

    monkeypatch.setattr(sync_script, "parse_args", lambda: args)
    monkeypatch.setattr(sync_script, "Session", lambda engine: FakeSession())
    monkeypatch.setattr(
        sync_script,
        "sync_player_metadata",
        lambda **arguments: SyncStats(updated=1),
    )
    monkeypatch.setattr(sync_script, "sync_player_media", fake_media)
    monkeypatch.setattr(sync_script, "write_sync_manifest", flaky_manifest)
    monkeypatch.setattr(
        sync_script,
        "print_stats",
        lambda **arguments: None,
    )

    with pytest.raises(RuntimeError, match="simulated manifest failure"):
        sync_script.main()

    interrupted_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert interrupted_plan["status"] == "finalizing"
    assert baseline_path.read_bytes() != baseline_before
    assert not target_path.exists()

    sync_script.main()

    completed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert completed_plan["status"] == "complete"
    assert media_calls == 1
    assert manifest_calls == 2
    assert load_players_csv(baseline_path) == [target_record]
    archive_path = Path(completed_plan["archive_path"])
    assert archive_path.is_file()
    assert archive_path.parent.name == "_media_refresh_plans"
