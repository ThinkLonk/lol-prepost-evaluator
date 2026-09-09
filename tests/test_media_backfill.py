"""Unit tests cho module đối soát và liên kết hình ảnh (media backfill)."""

from scripts.apply_media_backfill import (
    PlayerRecord,
    TeamRecord,
    generate_markdown_report,
    match_players,
    match_teams,
)


def test_match_teams_unique_success():
    oe_teams = [
        TeamRecord("oe:team:1", "T1", "T1", None),
        TeamRecord("oe:team:2", "Gen.G", "Gen.G", None),
    ]
    lp_teams = [
        TeamRecord("lp_team_1", "t1", "T1", "assets/teams/t1.png"),
        TeamRecord("lp_team_2", "gen.g", "Gen.G", "assets/teams/geng.png"),
    ]

    oe_up, unres = match_teams(oe_teams, lp_teams)

    assert len(oe_up) == 2
    assert len(unres) == 0
    assert oe_up[0] == {"team_id": "oe:team:1", "logo_file": "assets/teams/t1.png"}


def test_match_teams_ambiguous_remains_unresolved():
    oe_teams = [TeamRecord("oe:team:1", "Vanguard", "Vanguard", None)]
    lp_teams = [
        TeamRecord("lp_team_1", "vanguard", "Vanguard Esports", "assets/teams/v1.png"),
        TeamRecord("lp_team_2", "vanguard", "Vanguard Gaming", "assets/teams/v2.png"),
    ]

    oe_up, unres = match_teams(oe_teams, lp_teams)

    assert len(oe_up) == 0
    assert len(unres) == 1
    assert unres[0]["reason"].startswith("AMBIGUOUS_")


def test_match_teams_no_candidate():
    oe_teams = [TeamRecord("oe:team:99", "Unknown Team", "Unknown", None)]
    lp_teams = []

    oe_up, unres = match_teams(oe_teams, lp_teams)

    assert len(oe_up) == 0
    assert len(unres) == 1
    assert unres[0]["reason"] == "NO_LEAGUEPEDIA_CANDIDATE"


def test_match_players_unique_success():
    oe_players = [
        PlayerRecord("oe:player:1", "Faker", "Faker", None),
        PlayerRecord("oe:player:2", "Chovy", "Chovy", None),
    ]
    lp_players = [
        PlayerRecord("lp_player_1", "faker", "Faker", "assets/players/faker.webp"),
        PlayerRecord("lp_player_2", "chovy", "Chovy", "assets/players/chovy.webp"),
    ]

    oe_up, unres = match_players(oe_players, lp_players)

    assert len(oe_up) == 2
    assert len(unres) == 0
    assert oe_up[0] == {"player_id": "oe:player:1", "photo_file": "assets/players/faker.webp"}


def test_match_players_ambiguous_single_photo_resolves():
    oe_players = [PlayerRecord("oe:player:1", "ShowMaker", "ShowMaker", None)]
    lp_players = [
        PlayerRecord("lp_player_1", "showmaker", "ShowMaker", "assets/players/showmaker.webp"),
        PlayerRecord("lp_player_2", "showmaker", "ShowMaker (Sub)", None),
    ]

    oe_up, unres = match_players(oe_players, lp_players)

    assert len(oe_up) == 1
    assert len(unres) == 0
    assert oe_up[0]["photo_file"] == "assets/players/showmaker.webp"


def test_match_players_ambiguous_multiple_photos_remains_unresolved():
    oe_players = [PlayerRecord("oe:player:1", "Arthur", "Arthur", None)]
    lp_players = [
        PlayerRecord("lp_player_1", "arthur", "Arthur (KR)", "assets/players/arthur1.webp"),
        PlayerRecord("lp_player_2", "arthur", "Arthur (BR)", "assets/players/arthur2.webp"),
    ]

    oe_up, unres = match_players(oe_players, lp_players)

    assert len(oe_up) == 0
    assert len(unres) == 1
    assert unres[0]["reason"].startswith("AMBIGUOUS_")


def test_generate_markdown_report():
    report = generate_markdown_report(
        total_oe_teams=10,
        oe_teams_updated=9,
        unresolved_teams=[{"name": "TeamX", "oe_team_id": "oe:team:x", "reason": "NO_LEAGUEPEDIA_CANDIDATE"}],
        total_oe_players=100,
        oe_players_updated=85,
        unresolved_players=[{"name": "PlayerY", "oe_player_id": "oe:player:y", "reason": "AMBIGUOUS_2_CANDIDATES"}],
        mode="dry-run",
    )

    assert "# Báo cáo Đối soát và Liên kết Tài nguyên Hình ảnh" in report
    assert "90.0%" in report
    assert "85.0%" in report
    assert "DRY-RUN" in report
