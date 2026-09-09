"""Script an toàn đối soát và liên kết photo_file, logo_file từ Leaguepedia sang Oracle."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

REPORT_MD_PATH = (
    ROOT_DIR
    / "reports"
    / "data_quality"
    / "media_backfill_report.md"
)


@dataclass(frozen=True)
class TeamRecord:
    team_id: str
    canonical_name: str
    display_name: str
    logo_file: str | None


@dataclass(frozen=True)
class PlayerRecord:
    player_id: str
    canonical_name: str
    display_name: str
    photo_file: str | None


def match_teams(
    oe_teams: list[TeamRecord],
    lp_teams: list[TeamRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Đối soát và gán logo_file từ lp_team sang oe:team.
    
    Quy tắc:
    - Gom nhóm lp_team theo lower(canonical_name).
    - Nếu khớp duy nhất 1 đội lp_team có logo_file: gán logo cho oe:team.
    - Nếu trùng lặp mơ hồ (>1 ứng viên có logo): giữ unresolved để đảm bảo an toàn.
    """
    lp_by_name: dict[str, list[TeamRecord]] = defaultdict(list)
    for team in lp_teams:
        lp_by_name[team.canonical_name.strip().lower()].append(team)

    oe_updates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for oe in oe_teams:
        key = oe.canonical_name.strip().lower()
        candidates = lp_by_name.get(key, [])

        if len(candidates) == 1:
            matched = candidates[0]
            if matched.logo_file:
                oe_updates.append(
                    {"team_id": oe.team_id, "logo_file": matched.logo_file}
                )
        elif len(candidates) > 1:
            exact_cands = [
                c for c in candidates if c.canonical_name.strip() == oe.canonical_name.strip()
            ]
            if len(exact_cands) == 1 and exact_cands[0].logo_file:
                matched = exact_cands[0]
                oe_updates.append(
                    {"team_id": oe.team_id, "logo_file": matched.logo_file}
                )
            else:
                unresolved.append(
                    {
                        "oe_team_id": oe.team_id,
                        "name": oe.canonical_name,
                        "reason": f"AMBIGUOUS_{len(candidates)}_CANDIDATES",
                        "candidates": [c.team_id for c in candidates],
                    }
                )
        else:
            unresolved.append(
                {
                    "oe_team_id": oe.team_id,
                    "name": oe.canonical_name,
                    "reason": "NO_LEAGUEPEDIA_CANDIDATE",
                    "candidates": [],
                }
            )

    return oe_updates, unresolved


def match_players(
    oe_players: list[PlayerRecord],
    lp_players: list[PlayerRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Đối soát và gán photo_file từ lp_player sang oe:player.
    
    Quy tắc:
    - Gom nhóm lp_player theo lower(canonical_name).
    - Nếu khớp duy nhất 1 tuyển thủ lp_player có photo_file: gán photo cho oe:player.
    - Nếu có nhiều ứng viên (>1): kiểm tra khớp chính xác tên và chỉ 1 ứng viên có ảnh, nếu không thì giữ unresolved.
    """
    lp_by_name: dict[str, list[PlayerRecord]] = defaultdict(list)
    for player in lp_players:
        lp_by_name[player.canonical_name.strip().lower()].append(player)

    oe_updates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for oe in oe_players:
        key = oe.canonical_name.strip().lower()
        candidates = lp_by_name.get(key, [])

        if len(candidates) == 1:
            matched = candidates[0]
            if matched.photo_file:
                oe_updates.append(
                    {"player_id": oe.player_id, "photo_file": matched.photo_file}
                )
        elif len(candidates) > 1:
            exact_cands = [
                c for c in candidates if c.canonical_name.strip() == oe.canonical_name.strip()
            ]
            cands_with_photo = [c for c in candidates if c.photo_file is not None]

            if len(exact_cands) == 1 and exact_cands[0].photo_file:
                matched = exact_cands[0]
                oe_updates.append(
                    {"player_id": oe.player_id, "photo_file": matched.photo_file}
                )
            elif len(cands_with_photo) == 1:
                matched = cands_with_photo[0]
                oe_updates.append(
                    {"player_id": oe.player_id, "photo_file": matched.photo_file}
                )
            else:
                unresolved.append(
                    {
                        "oe_player_id": oe.player_id,
                        "name": oe.canonical_name,
                        "reason": f"AMBIGUOUS_{len(candidates)}_CANDIDATES",
                        "candidates": [c.player_id for c in candidates],
                    }
                )
        else:
            unresolved.append(
                {
                    "oe_player_id": oe.player_id,
                    "name": oe.canonical_name,
                    "reason": "NO_LEAGUEPEDIA_CANDIDATE",
                    "candidates": [],
                }
            )

    return oe_updates, unresolved


def generate_markdown_report(
    *,
    total_oe_teams: int,
    oe_teams_updated: int,
    unresolved_teams: list[dict[str, Any]],
    total_oe_players: int,
    oe_players_updated: int,
    unresolved_players: list[dict[str, Any]],
    mode: str,
) -> str:
    """Tạo báo cáo kiểm định chất lượng đối soát hình ảnh Markdown."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    team_cov = (oe_teams_updated / total_oe_teams * 100) if total_oe_teams > 0 else 0
    player_cov = (oe_players_updated / total_oe_players * 100) if total_oe_players > 0 else 0

    lines = [
        "# Báo cáo Đối soát và Liên kết Tài nguyên Hình ảnh (Media Backfill)",
        "",
        f"- **Thời điểm thực hiện**: `{now_utc}`",
        f"- **Chế độ**: `{mode.upper()}`",
        "",
        "## 1. Tổng quan Đội tuyển (Teams)",
        "",
        f"- Tổng số đội tuyển Oracle (`oe:team:`): **{total_oe_teams:,d}**",
        f"- Số đội tuyển liên kết thành công `logo_file`: **{oe_teams_updated:,d} ({team_cov:.1f}%)**",
        f"- Số đội tuyển chưa thể liên kết (unresolved): **{len(unresolved_teams):,d}**",
        "",
        "## 2. Tổng quan Tuyển thủ (Players)",
        "",
        f"- Tổng số tuyển thủ Oracle (`oe:player:`): **{total_oe_players:,d}**",
        f"- Số tuyển thủ liên kết thành công `photo_file`: **{oe_players_updated:,d} ({player_cov:.1f}%)**",
        f"- Số tuyển thủ chưa thể liên kết (unresolved): **{len(unresolved_players):,d}**",
        "",
        "## 3. Danh sách mẫu các trường hợp chưa liên kết (Unresolved Sample)",
        "",
        "### Đội tuyển:",
    ]

    for item in unresolved_teams[:10]:
        lines.append(f"- Đội: `{item['name']}` ({item['oe_team_id']}) — Lý do: `{item['reason']}`")

    lines.append("\n### Tuyển thủ:")
    for item in unresolved_players[:15]:
        lines.append(f"- Tuyển thủ: `{item['name']}` ({item['oe_player_id']}) — Lý do: `{item['reason']}`")

    lines.append("\n## 4. Kết luận")
    lines.append(
        "- Quy tắc bảo toàn dữ liệu tuân thủ nghiêm ngặt Mục 9 `AGENTS.md`: "
        "chỉ liên kết khi xác định được ứng viên duy nhất, giữ nguyên trạng thái nếu mơ hồ."
    )
    lines.append("- Các khóa chính và khóa ngoại trong bảng trận đấu không bị thay đổi.")
    lines.append("")

    return "\n".join(lines)


def fetch_database_records(conn: Connection) -> tuple[list[TeamRecord], list[TeamRecord], list[PlayerRecord], list[PlayerRecord]]:
    """Lấy dữ liệu từ các bảng team và player trong PostgreSQL."""
    teams_raw = conn.execute(text("""
        SELECT team_id, canonical_name, display_name, logo_file
        FROM team;
    """)).fetchall()

    oe_teams: list[TeamRecord] = []
    lp_teams: list[TeamRecord] = []
    for row in teams_raw:
        rec = TeamRecord(
            team_id=row[0],
            canonical_name=row[1],
            display_name=row[2],
            logo_file=row[3],
        )
        if rec.team_id.startswith("oe:team:"):
            oe_teams.append(rec)
        elif rec.team_id.startswith("lp_team_"):
            lp_teams.append(rec)

    players_raw = conn.execute(text("""
        SELECT player_id, canonical_name, display_name, photo_file
        FROM player;
    """)).fetchall()

    oe_players: list[PlayerRecord] = []
    lp_players: list[PlayerRecord] = []
    for row in players_raw:
        rec = PlayerRecord(
            player_id=row[0],
            canonical_name=row[1],
            display_name=row[2],
            photo_file=row[3],
        )
        if rec.player_id.startswith("oe:player:"):
            oe_players.append(rec)
        elif rec.player_id.startswith("lp_player_"):
            lp_players.append(rec)

    return oe_teams, lp_teams, oe_players, lp_players


def apply_backfill_updates(
    conn: Connection,
    *,
    oe_team_updates: list[dict[str, Any]],
    oe_player_updates: list[dict[str, Any]],
) -> None:
    """Thực thi các lệnh UPDATE trong transaction."""
    # 1. Update oe:team logo_file
    for item in oe_team_updates:
        conn.execute(
            text("UPDATE team SET logo_file = :logo_file WHERE team_id = :team_id"),
            item,
        )

    # 2. Update oe:player photo_file
    for item in oe_player_updates:
        conn.execute(
            text("UPDATE player SET photo_file = :photo_file WHERE player_id = :player_id"),
            item,
        )


def main() -> None:
    from match_insight.database.engine import engine

    parser = argparse.ArgumentParser(
        description="Đối soát và liên kết photo_file, logo_file vào PostgreSQL"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Thực hiện ghi thay đổi vào database (mặc định chỉ dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ đối soát và in báo cáo, không ghi database",
    )
    args = parser.parse_args()

    apply_mode = args.apply and not args.dry_run
    mode_str = "apply" if apply_mode else "dry-run"

    print(f"=== BẮT ĐẦU MEDIA BACKFILL ({mode_str.upper()}) ===")

    with engine.connect() as conn:
        oe_teams, lp_teams, oe_players, lp_players = fetch_database_records(conn)

    print(f"- Đã nạp: {len(oe_teams)} đội Oracle, {len(lp_teams)} đội Leaguepedia.")
    print(f"- Đã nạp: {len(oe_players)} tuyển thủ Oracle, {len(lp_players)} tuyển thủ Leaguepedia.")

    oe_team_up, unres_teams = match_teams(oe_teams, lp_teams)
    oe_player_up, unres_players = match_players(oe_players, lp_players)

    print("\n[KẾT QUẢ ĐỐI SOÁT]:")
    team_cov = (len(oe_team_up) / len(oe_teams) * 100) if oe_teams else 0
    player_cov = (len(oe_player_up) / len(oe_players) * 100) if oe_players else 0
    print(f"  + Đội tuyển Oracle sẽ được gán logo : {len(oe_team_up)} / {len(oe_teams)} ({team_cov:.1f}%)")
    print(f"  + Tuyển thủ Oracle sẽ được gán ảnh  : {len(oe_player_up)} / {len(oe_players)} ({player_cov:.1f}%)")

    report_content = generate_markdown_report(
        total_oe_teams=len(oe_teams),
        oe_teams_updated=len(oe_team_up),
        unresolved_teams=unres_teams,
        total_oe_players=len(oe_players),
        oe_players_updated=len(oe_player_up),
        unresolved_players=unres_players,
        mode=mode_str,
    )

    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text(report_content, encoding="utf-8")
    print(f"- Báo cáo đối soát đã được lưu tại: {REPORT_MD_PATH}")

    if apply_mode:
        print("\nĐang thực thi ghi dữ liệu vào PostgreSQL trong transaction...")
        with engine.begin() as conn:
            apply_backfill_updates(
                conn,
                oe_team_updates=oe_team_up,
                oe_player_updates=oe_player_up,
            )
        print(">>> GHI THÀNH CÔNG! Toàn bộ thay đổi đã được COMMIT an toàn.")
    else:
        print("\n>>> Chế độ DRY-RUN kết thúc. Chưa có thay đổi nào được ghi vào database.")
        print("    Để cập nhật chính thức, hãy chạy lại với tham số: --apply")


if __name__ == "__main__":
    main()
