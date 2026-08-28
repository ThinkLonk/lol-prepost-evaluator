"""Kiểm thử adapter Leaguepedia Cargo dùng mwrogue."""

from __future__ import annotations

import json
from pathlib import Path

import mwrogue.esports_client as esports_client_module
import pytest

import match_insight.data_processing.leaguepedia as leaguepedia_module


class FakeCargoClient:
    """Cargo client giả không thực hiện network request."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        *,
        tables: str,
        fields: str,
        join_on: str | None = None,
        where: str | None = None,
        group_by: str | None = None,
        order_by: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        auto_continue: bool = True,
    ) -> object:
        call: dict[str, object] = {
            "tables": tables,
            "fields": fields,
            "join_on": join_on,
            "where": where,
            "order_by": order_by,
            "offset": offset,
            "limit": limit,
            "auto_continue": auto_continue,
        }

        if group_by is not None:
            call["group_by"] = group_by

        self.calls.append(call)
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


class FakeMediaWikiApiClient:
    """MediaWiki API client giả không thực hiện network request."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def api(
        self,
        action: str,
        http_method: str = "POST",
        **kwargs: object,
    ) -> object:
        self.calls.append(
            {
                "action": action,
                "http_method": http_method,
                **kwargs,
            }
        )
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


class FakeApiError(RuntimeError):
    """Lỗi API giả có mã tương tự mwclient.errors.APIError."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class FakeHttpResponse:
    """HTTP response giả chỉ mang status code."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeHttpError(RuntimeError):
    """HTTP error giả tương thích requests.HTTPError."""

    def __init__(self, status_code: int) -> None:
        self.response = FakeHttpResponse(status_code)
        super().__init__(f"HTTP {status_code}")


def test_create_site_fails_fast_for_http_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_delays: list[int] = []

    def fake_esports_client(
        wiki: str,
        *,
        credentials: object,
        user_agent: str,
    ) -> object:
        nonlocal attempts
        attempts += 1
        assert wiki == "lol"
        assert credentials is not None
        assert user_agent == (
            "MatchInsightReferenceSync/1.0 "
            "(ExampleAccount@ReferenceSync)"
        )

        raise FakeHttpError(403)

    monkeypatch.setattr(
        leaguepedia_module,
        "load_local_environment",
        lambda: None,
    )
    monkeypatch.setenv(
        "WIKI_USERNAME_MATCH_INSIGHT",
        "ExampleAccount@ReferenceSync",
    )
    monkeypatch.setenv(
        "WIKI_PASSWORD_MATCH_INSIGHT",
        "secret-token",
    )
    monkeypatch.setattr(
        esports_client_module,
        "EsportsClient",
        fake_esports_client,
    )
    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    with pytest.raises(
        RuntimeError,
        match="browser snapshot",
    ):
        leaguepedia_module._create_leaguepedia_site()

    assert attempts == 1
    assert sleep_delays == []


def test_fetch_cargo_rows_uses_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[
            [
                {
                    "SourcePageId": "1",
                    "Name": "Alpha",
                },
                {
                    "SourcePageId": "2",
                    "Name": "Beta",
                },
            ],
            [
                {
                    "SourcePageId": "3",
                    "Name": "Gamma",
                }
            ],
        ]
    )
    sleep_delays: list[int] = []

    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    rows = leaguepedia_module.fetch_cargo_rows(
        table="Teams",
        fields=(
            "Teams._pageID=SourcePageId",
            "Teams.Name=Name",
        ),
        order_by="Teams._pageID",
        page_size=2,
        cargo_client=cargo_client,
    )

    assert [row["SourcePageId"] for row in rows] == [
        "1",
        "2",
        "3",
    ]
    assert len(cargo_client.calls) == 2
    assert cargo_client.calls[0] == {
        "tables": "Teams",
        "fields": (
            "Teams._pageID=SourcePageId, "
            "Teams.Name=Name"
        ),
        "join_on": None,
        "where": None,
        "order_by": "Teams._pageID",
        "offset": 0,
        "limit": 2,
        "auto_continue": False,
    }
    assert cargo_client.calls[1]["offset"] == 2
    assert sleep_delays == [20]


def test_fetch_cargo_rows_passes_group_by() -> None:
    cargo_client = FakeCargoClient(
        responses=[
            [{"PlayerPage": "Faker", "Role": "Mid"}]
        ]
    )

    rows = leaguepedia_module.fetch_cargo_rows(
        table="ScoreboardPlayers",
        fields=(
            "ScoreboardPlayers.Link=PlayerPage",
            "ScoreboardPlayers.Role=Role",
        ),
        group_by=(
            "ScoreboardPlayers.Link, "
            "ScoreboardPlayers.Role"
        ),
        order_by=(
            "ScoreboardPlayers.Link, "
            "ScoreboardPlayers.Role"
        ),
        cargo_client=cargo_client,
    )

    assert rows == [
        {"PlayerPage": "Faker", "Role": "Mid"}
    ]
    assert cargo_client.calls[0]["group_by"] == (
        "ScoreboardPlayers.Link, "
        "ScoreboardPlayers.Role"
    )


def test_fetch_file_urls_uses_batched_imageinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_url = (
        "https://static.wikia.nocookie.net/"
        "test/images/a/ab/First_Logo.png"
    )
    api_client = FakeMediaWikiApiClient(
        responses=[
            {
                "query": {
                    "normalized": [
                        {
                            "from": "File:Old_Logo.png",
                            "to": "File:Old Logo.png",
                        }
                    ],
                    "redirects": [
                        {
                            "from": "File:Old Logo.png",
                            "to": "File:First Logo.png",
                        }
                    ],
                    "pages": [
                        {
                            "pageid": 1,
                            "title": "File:First Logo.png",
                            "imageinfo": [{"url": first_url}],
                        }
                    ]
                }
            },
            {
                "query": {
                    "pages": [
                        {
                            "ns": 6,
                            "title": "File:Missing.png",
                            "missing": True,
                        }
                    ]
                }
            },
        ]
    )
    sleep_delays: list[int] = []

    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    resolved = leaguepedia_module.fetch_file_urls(
        file_names=("Old_Logo.png", "Missing.png"),
        batch_size=1,
        api_client=api_client,
    )

    assert resolved == {"Old_Logo.png": first_url}
    assert len(api_client.calls) == 2
    assert api_client.calls[0] == {
        "action": "query",
        "http_method": "POST",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": "File:Old_Logo.png",
        "redirects": True,
        "formatversion": 2,
    }
    assert api_client.calls[1]["titles"] == (
        "File:Missing.png"
    )
    assert sleep_delays == [2]


def test_fetch_file_urls_retries_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_url = (
        "https://static.wikia.nocookie.net/"
        "test/images/a/ab/Test.png"
    )
    api_client = FakeMediaWikiApiClient(
        responses=[
            FakeApiError("ratelimited"),
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "title": "File:Test.png",
                            "imageinfo": [{"url": direct_url}],
                        }
                    ]
                }
            },
        ]
    )
    sleep_delays: list[int] = []

    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    resolved = leaguepedia_module.fetch_file_urls(
        file_names=("Test.png",),
        api_client=api_client,
    )

    assert resolved == {"Test.png": direct_url}
    assert len(api_client.calls) == 2
    assert sleep_delays == [30]


def test_fetch_cargo_rows_retries_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[
            TimeoutError("Leaguepedia tạm thời chậm."),
            [],
        ]
    )
    sleep_delays: list[int] = []

    def fake_sleep(seconds: int) -> None:
        sleep_delays.append(seconds)

    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        fake_sleep,
    )

    rows = leaguepedia_module.fetch_cargo_rows(
        table="Teams",
        fields=("Teams._pageID=SourcePageId",),
        order_by="Teams._pageID",
        cargo_client=cargo_client,
    )

    assert rows == []
    assert len(cargo_client.calls) == 2
    assert sleep_delays == [1]


def test_fetch_cargo_rows_waits_longer_when_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[
            FakeApiError("ratelimited"),
            [],
        ]
    )
    sleep_delays: list[int] = []

    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    rows = leaguepedia_module.fetch_cargo_rows(
        table="Teams",
        fields=("Teams._pageID=SourcePageId",),
        order_by="Teams._pageID",
        cargo_client=cargo_client,
    )

    assert rows == []
    assert len(cargo_client.calls) == 2
    assert sleep_delays == [30]


def test_fetch_cargo_rows_fails_fast_for_http_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[FakeHttpError(403)],
    )
    sleep_delays: list[int] = []
    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    with pytest.raises(
        RuntimeError,
        match="browser snapshot",
    ):
        leaguepedia_module.fetch_cargo_rows(
            table="Players",
            fields=("Players.ID=PlayerHandle",),
            order_by="Players.ID",
            cargo_client=cargo_client,
        )

    assert len(cargo_client.calls) == 1
    assert sleep_delays == []


def test_fetch_does_not_publish_partial_snapshot_after_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[
            [{"SourcePageId": "1"}],
            FakeApiError("ratelimited"),
            FakeApiError("ratelimited"),
            FakeApiError("ratelimited"),
            FakeApiError("ratelimited"),
        ]
    )
    sleep_delays: list[int] = []
    snapshot_path = tmp_path / "teams.json"

    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    with pytest.raises(
        RuntimeError,
        match="offset=1 sau 4 lần",
    ):
        leaguepedia_module.fetch_cargo_rows(
            table="Teams",
            fields=("Teams._pageID=SourcePageId",),
            order_by="Teams._pageID",
            page_size=1,
            snapshot_path=snapshot_path,
            cargo_client=cargo_client,
        )

    assert len(cargo_client.calls) == 5
    assert sleep_delays == [20, 30, 60, 120]
    assert not snapshot_path.exists()
    assert not snapshot_path.with_suffix(
        ".json.part"
    ).exists()


def test_fetch_cargo_rows_wraps_repeated_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[
            RuntimeError("badtoken"),
            RuntimeError("badtoken"),
            RuntimeError("badtoken"),
            RuntimeError("badtoken"),
        ]
    )
    sleep_delays: list[int] = []

    monkeypatch.setattr(
        leaguepedia_module.time,
        "sleep",
        sleep_delays.append,
    )

    with pytest.raises(
        RuntimeError,
        match="sau 4 lần",
    ):
        leaguepedia_module.fetch_cargo_rows(
            table="Teams",
            fields=("Teams._pageID=SourcePageId",),
            order_by="Teams._pageID",
            cargo_client=cargo_client,
        )

    assert len(cargo_client.calls) == 4
    assert sleep_delays == [1, 2, 4]


def test_normalize_cargo_rows_rejects_non_list() -> None:
    with pytest.raises(
        ValueError,
        match="phải trả về một list",
    ):
        leaguepedia_module._normalize_cargo_rows(
            {"cargoquery": []}
        )


def test_write_cargo_snapshot_preserves_raw_rows(
    tmp_path: Path,
) -> None:
    snapshot_path = (
        tmp_path
        / "raw"
        / "leaguepedia"
        / "teams.json"
    )
    rows = [
        {
            "SourcePageId": "123",
            "Name": "Đội Thử Nghiệm",
        }
    ]

    leaguepedia_module.write_cargo_snapshot(
        path=snapshot_path,
        table="Teams",
        fields=(
            "Teams._pageID=SourcePageId",
            "Teams.Name=Name",
        ),
        order_by="Teams._pageID",
        where=None,
        page_size=500,
        rows=rows,
    )

    snapshot = json.loads(
        snapshot_path.read_text(encoding="utf-8")
    )

    assert snapshot["row_count"] == 1
    assert snapshot["rows"] == rows
    assert snapshot["source"]["table"] == "Teams"
    assert snapshot["source"]["client"] == "mwrogue"
    assert snapshot["source"]["page_size"] == 500
    assert snapshot["source"]["where"] is None
    assert snapshot["source"]["join_on"] is None
    assert snapshot["fetched_at_utc"].endswith(
        "+00:00"
    )
    assert not snapshot_path.with_suffix(
        ".json.part"
    ).exists()


def test_fetch_cargo_rows_passes_and_snapshots_join_on(
    tmp_path: Path,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[
            [
                {
                    "PlayerPage": "Gumayusi",
                    "EffectiveSortDate": "2026-01-14",
                }
            ]
        ]
    )
    snapshot_path = tmp_path / "player-images.json"

    rows = leaguepedia_module.fetch_cargo_rows(
        table="PlayerImages=PI,Tournaments=T",
        fields=(
            "PI.Link=PlayerPage",
            "COALESCE(PI.SortDate,T.Date)=EffectiveSortDate",
        ),
        join_on="PI.Tournament=T.OverviewPage",
        where="PI.IsProfileImage=1",
        order_by="PI.Link",
        cargo_client=cargo_client,
        snapshot_path=snapshot_path,
    )

    assert rows[0]["EffectiveSortDate"] == "2026-01-14"
    assert cargo_client.calls[0]["join_on"] == (
        "PI.Tournament=T.OverviewPage"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["source"]["join_on"] == (
        "PI.Tournament=T.OverviewPage"
    )


def test_write_cargo_snapshot_rejects_empty_rows(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "teams.json"
    original_content = "existing snapshot\n"
    snapshot_path.write_text(
        original_content,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="rows rỗng",
    ):
        leaguepedia_module.write_cargo_snapshot(
            path=snapshot_path,
            table="Teams",
            fields=("Teams._pageID=SourcePageId",),
            order_by="Teams._pageID",
            where=None,
            page_size=500,
            rows=[],
        )

    assert snapshot_path.read_text(
        encoding="utf-8",
    ) == original_content


def test_fetch_cargo_rows_writes_requested_snapshot(
    tmp_path: Path,
) -> None:
    cargo_client = FakeCargoClient(
        responses=[
            [
                {
                    "SourcePageId": "123",
                    "Name": "Test Team",
                }
            ]
        ]
    )
    snapshot_path = tmp_path / "teams.json"

    rows = leaguepedia_module.fetch_cargo_rows(
        table="Teams",
        fields=(
            "Teams._pageID=SourcePageId",
            "Teams.Name=Name",
        ),
        order_by="Teams._pageID",
        snapshot_path=snapshot_path,
        cargo_client=cargo_client,
    )

    snapshot = json.loads(
        snapshot_path.read_text(encoding="utf-8")
    )

    assert len(rows) == 1
    assert snapshot["rows"] == rows
    assert snapshot["row_count"] == 1


def test_load_credentials_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        leaguepedia_module,
        "load_local_environment",
        lambda: None,
    )
    monkeypatch.setenv(
        "WIKI_USERNAME_MATCH_INSIGHT",
        "ExampleAccount@ReferenceSync",
    )
    monkeypatch.setenv(
        "WIKI_PASSWORD_MATCH_INSIGHT",
        "secret-token",
    )

    username, password = (
        leaguepedia_module._load_credentials()
    )

    assert username == "ExampleAccount@ReferenceSync"
    assert password == "secret-token"


def test_load_credentials_rejects_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        leaguepedia_module,
        "load_local_environment",
        lambda: None,
    )
    monkeypatch.delenv(
        "WIKI_USERNAME_MATCH_INSIGHT",
        raising=False,
    )
    monkeypatch.delenv(
        "WIKI_PASSWORD_MATCH_INSIGHT",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="WIKI_USERNAME_MATCH_INSIGHT",
    ):
        leaguepedia_module._load_credentials()


def test_load_credentials_requires_bot_username(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        leaguepedia_module,
        "load_local_environment",
        lambda: None,
    )
    monkeypatch.setenv(
        "WIKI_USERNAME_MATCH_INSIGHT",
        "ExampleAccount",
    )
    monkeypatch.setenv(
        "WIKI_PASSWORD_MATCH_INSIGHT",
        "secret-token",
    )

    with pytest.raises(
        ValueError,
        match="AccountName@BotName",
    ):
        leaguepedia_module._load_credentials()
