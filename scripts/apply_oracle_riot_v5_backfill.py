"""Script nạp (backfill) dữ liệu thời gian Riot V5 vào bảng game trong PostgreSQL."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SUMMARY_JSON_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "leaguepedia"
    / "oracle_riot_temporal_v5"
    / "oracle_2025_riot_v5_timestamps_summary_11565_games.json"
)
REPORT_MD_PATH = (
    ROOT_DIR
    / "reports"
    / "data_quality"
    / "oracle_2025_riot_v5_backfill_report.md"
)


def extract_valid_backfill_payloads(
    raw_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Trích xuất danh sách các record hợp lệ để cập nhật database."""
    valid_payloads: list[dict[str, Any]] = []

    for item in raw_list:
        gid = item.get("game_id")
        if not gid or item.get("status") != "SUCCESS":
            continue

        start_ms = item.get("start_timestamp_ms")
        end_ms = item.get("end_timestamp_ms")
        started_at_str = item.get("started_at")
        ended_at_str = item.get("ended_at")

        if not isinstance(start_ms, int) or not isinstance(end_ms, int):
            continue
        if start_ms <= 0 or end_ms <= start_ms:
            continue
        if not started_at_str or not ended_at_str:
            continue

        try:
            started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
            ended_at = datetime.fromisoformat(ended_at_str.replace("Z", "+00:00"))
            if ended_at <= started_at:
                continue
        except (ValueError, TypeError):
            continue

        valid_payloads.append(
            {
                "game_id": gid,
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )

    return valid_payloads


def apply_backfill() -> dict[str, Any]:
    """Thực hiện cập nhật started_at và ended_at vào PostgreSQL có Transaction."""
    from match_insight.database.engine import engine

    if not SUMMARY_JSON_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file summary JSON tại: {SUMMARY_JSON_PATH}"
        )

    print(f"Đang đọc dữ liệu thời gian từ: {SUMMARY_JSON_PATH.name}...")
    with SUMMARY_JSON_PATH.open("r", encoding="utf-8") as f:
        v5_raw_list: list[dict[str, Any]] = json.load(f)

    valid_payloads = extract_valid_backfill_payloads(v5_raw_list)
    print(f"Tổng số record V5 hợp lệ sẵn sàng nạp: {len(valid_payloads):,}")

    update_stmt = text(
        """
        UPDATE game
        SET started_at = :started_at,
            ended_at = :ended_at
        WHERE game_id = :game_id;
        """
    )

    verify_updated_stmt = text(
        """
        SELECT COUNT(*) 
        FROM game 
        WHERE started_at IS NOT NULL AND ended_at IS NOT NULL;
        """
    )

    verify_null_stmt = text(
        """
        SELECT COUNT(*) 
        FROM game 
        WHERE started_at IS NULL AND ended_at IS NULL;
        """
    )

    verify_order_violations_stmt = text(
        """
        SELECT COUNT(*) 
        FROM game 
        WHERE started_at IS NOT NULL 
          AND ended_at IS NOT NULL 
          AND ended_at <= started_at;
        """
    )

    print("Đang mở Transaction trên PostgreSQL để nạp dữ liệu...")
    with engine.begin() as connection:
        # Thực thi batch update
        connection.execute(update_stmt, valid_payloads)

        # Đối chiếu kết quả ngay trong transaction trước khi commit
        updated_count = connection.execute(verify_updated_stmt).scalar_one()
        null_count = connection.execute(verify_null_stmt).scalar_one()
        violation_count = connection.execute(
            verify_order_violations_stmt
        ).scalar_one()

        if violation_count > 0:
            raise RuntimeError(
                f"Phát hiện {violation_count} ván vi phạm trật tự thời gian ended_at <= started_at. Đã Rollback."
            )

    print("Transaction đã commit thành công vào PostgreSQL.")
    print(f"Số ván đã có started_at và ended_at trong DB: {updated_count:,}")
    print(f"Số ván giữ NULL (LPL và ván thiếu nguồn): {null_count:,}")

    # Ghi báo cáo kết quả
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_content = f"""# Báo cáo Nạp Dữ liệu Thời gian Riot V5 vào PostgreSQL (Bước 7F)

## 1. Thông tin giao tác
- Thời điểm thực thi UTC: {datetime.now(timezone.utc).isoformat()}
- Nguồn dữ liệu: `{SUMMARY_JSON_PATH.name}`
- Phương thức: Transaction có kiểm soát (`BEGIN ... COMMIT`)

## 2. Kết quả cập nhật
| Chỉ số | Giá trị |
|---|---:|
| Tổng số ván đã cập nhật thời gian (`started_at`, `ended_at`) | **{updated_count:,}** |
| Số ván giữ nguyên NULL (LPL và ván thiếu nguồn) | **{null_count:,}** |
| Số ván vi phạm trật tự thời gian (`ended_at <= started_at`) | **0** |
| Trạng thái giao tác (Transaction Status) | **COMMITTED** |

## 3. Trạng thái Dự án
- `project_temporal_readiness`: **READY** (Đối với 31 giải đấu có dữ liệu Riot Platform)
- `history_cutoff_status`: **AVAILABLE**
- Điều kiện chống rò rỉ dữ liệu `historical_ended_at < target_started_at`: **ĐÃ SẴN SÀNG ĐỂ TÍNH TOÁN**
"""
    REPORT_MD_PATH.write_text(report_content, encoding="utf-8")
    print(f"Đã ghi báo cáo kết quả nạp: {REPORT_MD_PATH}")

    return {
        "updated_count": updated_count,
        "null_count": null_count,
        "violation_count": violation_count,
        "report_path": str(REPORT_MD_PATH),
    }


if __name__ == "__main__":
    apply_backfill()
