"""CLI đồng bộ player từ Leaguepedia hoặc staging CSV."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

from sqlalchemy.orm import Session

from match_insight.data_processing.leaguepedia import (
    LEAGUEPEDIA_API_URL,
    LEAGUEPEDIA_CLIENT_NAME,
)
from match_insight.data_processing.players import (
    PlayerRecord,
    delete_pruned_player_media,
    fetch_players,
    load_players_csv,
    load_players_snapshot,
    prune_missing_leaguepedia_players,
    refresh_players_snapshot_image_sort_dates,
    resolve_player_photo_urls,
    sync_player_media,
    sync_player_metadata,
    write_players_csv,
)
from match_insight.data_processing.reference_common import (
    file_sha256,
    write_sync_manifest,
)
from match_insight.database.engine import engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "players.csv"
)
DEFAULT_RAW_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "leaguepedia"
    / "players_reference.json"
)
PLAYER_MEDIA_BATCH_SIZE = 50
DEFAULT_IMAGE_SORT_SNAPSHOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "leaguepedia"
    / "player_image_sort_dates.json"
)
DEFAULT_MEDIA_REFRESH_PLAN = (
    PROJECT_ROOT
    / "assets"
    / "players"
    / "_media_refresh_plan.json"
)

_BROWSER_EXPORTER = r"""
(async () => {
  "use strict";

  const API_URL = "/api.php";
  const PAGE_SIZE = 500;
  const CARGO_REQUEST_INTERVAL_MS = 20000;
  const IMAGEINFO_BATCH_SIZE = 50;
  const IMAGEINFO_DELAY_MS = 2000;
  const MAX_REQUEST_ATTEMPTS = 8;
  const RATE_LIMIT_RETRY_MS = 65000;
  const TRANSIENT_RETRY_DELAYS_MS = [2000, 5000, 15000, 30000, 60000];
  const CHECKPOINT_KEY = "__matchInsightPlayersReferenceV2";
  const sleep = (milliseconds) => new Promise(
    (resolve) => setTimeout(resolve, milliseconds),
  );

  const existingCheckpoint = globalThis[CHECKPOINT_KEY];
  const checkpoint = existingCheckpoint?.version === 2
    ? existingCheckpoint
    : {
      version: 2,
      cargoQueries: {},
      fileUrls: {},
      completedFileKeys: {},
      lastCargoRequestAt: 0,
      startedAtUtc: new Date().toISOString(),
      running: false,
    };
  checkpoint.cargoQueries ||= {};
  checkpoint.fileUrls ||= {};
  checkpoint.completedFileKeys ||= {};
  checkpoint.lastCargoRequestAt ||= 0;
  checkpoint.startedAtUtc ||= new Date().toISOString();
  checkpoint.running ||= false;
  globalThis[CHECKPOINT_KEY] = checkpoint;

  function retryAfterMilliseconds(value) {
    const seconds = Number(value);

    if (Number.isFinite(seconds) && seconds > 0) {
      return seconds * 1000;
    }

    const retryDate = Date.parse(value || "");

    if (Number.isFinite(retryDate)) {
      return Math.max(0, retryDate - Date.now());
    }

    return 0;
  }

  async function waitForCargoRequestSlot(label) {
    const elapsed = Date.now() - checkpoint.lastCargoRequestAt;
    const waitMilliseconds = Math.max(
      0,
      CARGO_REQUEST_INTERVAL_MS - elapsed,
    );

    if (waitMilliseconds > 0) {
      console.log(
        `${label}: pacing wait ${Math.ceil(waitMilliseconds / 1000)}s`,
      );
      await sleep(waitMilliseconds);
    }

    checkpoint.lastCargoRequestAt = Date.now();
  }

  async function requestJson(
    parameters,
    label,
    { paceCargo = false } = {},
  ) {
    let lastError;

    for (let attempt = 1; attempt <= MAX_REQUEST_ATTEMPTS; attempt += 1) {
      if (paceCargo) await waitForCargoRequestSlot(label);

      try {
        const response = await fetch(
          `${API_URL}?${parameters.toString()}`,
          {
            credentials: "same-origin",
            headers: { Accept: "application/json" },
          },
        );
        const payload = await response.json();

        if (!response.ok || payload.error) {
          const errorCode = payload.error?.code || `HTTP_${response.status}`;
          const reason = payload.error
            ? `${errorCode}: ${payload.error.info || ""}`
            : `HTTP ${response.status}`;
          const requestError = new Error(reason);
          requestError.apiCode = errorCode;
          requestError.retryAfter = response.headers.get("Retry-After");
          throw requestError;
        }

        return payload;
      } catch (error) {
        lastError = error;

        if (attempt >= MAX_REQUEST_ATTEMPTS) break;

        const errorText = String(error).toLowerCase();
        const rateLimited = (
          error?.apiCode === "ratelimited"
          || error?.apiCode === "HTTP_429"
          || errorText.includes("ratelimited")
          || errorText.includes("rate limit")
          || errorText.includes("http_429")
        );
        const nonRetryable = (
          error?.apiCode === "db_error"
          || error?.apiCode === "badvalue"
          || error?.apiCode === "invalidparammix"
        );

        if (nonRetryable) break;

        const retryAfterDelay = retryAfterMilliseconds(
          error?.retryAfter,
        );
        const transientDelay = TRANSIENT_RETRY_DELAYS_MS[
          Math.min(attempt - 1, TRANSIENT_RETRY_DELAYS_MS.length - 1)
        ];
        const retryDelay = rateLimited
          ? Math.max(RATE_LIMIT_RETRY_MS, retryAfterDelay)
          : transientDelay;

        console.warn(
          `${label}: ${String(error)}; retry in ${retryDelay / 1000}s`,
        );
        await sleep(retryDelay);
      }
    }

    throw new Error(`${label} failed: ${String(lastError)}`);
  }

  async function cargoQuery({
    tables,
    fields,
    joinOn = "",
    where = "",
    groupBy = "",
    orderBy,
  }) {
    const queryKey = JSON.stringify({
      tables,
      fields,
      joinOn,
      where,
      groupBy,
      orderBy,
      pageSize: PAGE_SIZE,
    });
    const queryCheckpoint = checkpoint.cargoQueries[queryKey] || {
      complete: false,
      nextOffset: 0,
      rows: [],
    };
    checkpoint.cargoQueries[queryKey] = queryCheckpoint;

    if (queryCheckpoint.complete) {
      console.log(
        `${tables}: reused ${queryCheckpoint.rows.length} checkpoint rows`,
      );
      return [...queryCheckpoint.rows];
    }

    const rows = queryCheckpoint.rows;
    let offset = queryCheckpoint.nextOffset;

    if (rows.length > 0) {
      console.log(`${tables}: resume offset=${offset}, rows=${rows.length}`);
    }

    while (true) {
      const parameters = new URLSearchParams({
        action: "cargoquery",
        format: "json",
        formatversion: "2",
        tables,
        fields,
        order_by: orderBy,
        limit: String(PAGE_SIZE),
        offset: String(offset),
      });

      if (where) parameters.set("where", where);
      if (joinOn) parameters.set("join_on", joinOn);
      if (groupBy) parameters.set("group_by", groupBy);

      const payload = await requestJson(
        parameters,
        `${tables} offset=${offset}`,
        { paceCargo: true },
      );
      if (!Array.isArray(payload.cargoquery)) {
        throw new Error(`${tables} returned no cargoquery array.`);
      }

      const page = payload.cargoquery.map((item, index) => {
        if (!item || typeof item.title !== "object" || !item.title) {
          throw new Error(
            `${tables} returned an invalid row at page index ${index}.`,
          );
        }

        return item.title;
      });
      rows.push(...page);
      queryCheckpoint.nextOffset = offset + PAGE_SIZE;
      console.log(`${tables}: ${rows.length} rows`);

      if (page.length < PAGE_SIZE) {
        queryCheckpoint.complete = true;
        break;
      }

      offset += PAGE_SIZE;
    }

    return [...rows];
  }

  function fileKey(value) {
    let cleaned = String(value || "").trim();
    if (cleaned.toLowerCase().startsWith("file:")) {
      cleaned = cleaned.slice(5);
    }
    return cleaned.replaceAll("_", " ").replace(/\s+/g, " ").toLowerCase();
  }

  async function resolveFileUrls(fileNames) {
    const uniqueNames = [...new Set(fileNames.filter(Boolean))];
    const resolved = new Map(
      uniqueNames
        .map((name) => [
          fileKey(name),
          checkpoint.fileUrls[fileKey(name)],
        ])
        .filter(([, url]) => Boolean(url)),
    );

    for (
      let start = 0;
      start < uniqueNames.length;
      start += IMAGEINFO_BATCH_SIZE
    ) {
      const batch = uniqueNames.slice(start, start + IMAGEINFO_BATCH_SIZE);
      const pendingBatch = batch.filter(
        (name) => !checkpoint.completedFileKeys[fileKey(name)],
      );

      if (pendingBatch.length === 0) {
        console.log(
          `imageinfo: reused ${Math.min(
            start + batch.length,
            uniqueNames.length,
          )}/${uniqueNames.length}`,
        );
        continue;
      }

      const parameters = new URLSearchParams({
        action: "query",
        format: "json",
        formatversion: "2",
        prop: "imageinfo",
        iiprop: "url",
        redirects: "1",
        titles: pendingBatch.map((name) => `File:${name}`).join("|"),
      });
      const payload = await requestJson(
        parameters,
        `imageinfo batch=${start / IMAGEINFO_BATCH_SIZE + 1}`,
      );
      const aliases = new Map();

      for (const group of ["normalized", "redirects"]) {
        for (const item of payload.query?.[group] || []) {
          aliases.set(fileKey(item.from), fileKey(item.to));
        }
      }

      for (const page of payload.query?.pages || []) {
        const url = page.imageinfo?.[0]?.url;
        if (url) resolved.set(fileKey(page.title), url);
      }

      for (let pass = 0; pass < 2; pass += 1) {
        for (const [source, target] of aliases.entries()) {
          if (resolved.has(target)) {
            resolved.set(source, resolved.get(target));
          }
        }
      }

      for (const name of pendingBatch) {
        const key = fileKey(name);
        checkpoint.completedFileKeys[key] = true;

        if (resolved.has(key)) {
          checkpoint.fileUrls[key] = resolved.get(key);
        }
      }

      console.log(
        `imageinfo: ${Math.min(start + batch.length, uniqueNames.length)}`
        + `/${uniqueNames.length}`,
      );
      await sleep(IMAGEINFO_DELAY_MS);
    }

    return new Map(
      uniqueNames
        .filter((name) => resolved.has(fileKey(name)))
        .map((name) => [name, resolved.get(fileKey(name))]),
    );
  }

  if (checkpoint.running) {
    console.warn(
      "Player reference exporter is already running in this tab.",
    );
    return;
  }

  checkpoint.running = true;

  try {

  const catalogRows = await cargoQuery({
    tables: "Players",
    fields: [
      "Players._pageID=SourcePageId",
      "Players.OverviewPage=SourcePageName",
      "Players.ID=PlayerHandle",
      "Players.Image=Image",
      "Players.IsPersonality=IsPersonality",
    ].join(","),
    orderBy: "Players._pageID",
  });
  const gameEvidenceRows = await cargoQuery({
    tables: "PlayerLeagueHistory",
    fields: [
      "PlayerLeagueHistory.Player=PlayerPage",
    ].join(","),
    where: "PlayerLeagueHistory.TotalGames>0",
    groupBy: "PlayerLeagueHistory.Player",
    orderBy: "PlayerLeagueHistory.Player",
  });
  const tournamentRosterRows = [];
  const unavailableEvidenceTables = [
    {
      table: "TournamentPlayers",
      reason: "Leaguepedia Cargo returned db_error during grouped query.",
    },
  ];
  const currentRosterRows = await cargoQuery({
    tables: "ListplayerCurrent",
    fields: [
      "ListplayerCurrent.Link=PlayerPage",
      "ListplayerCurrent.Role=Role",
    ].join(","),
    groupBy: "ListplayerCurrent.Link,ListplayerCurrent.Role",
    orderBy: "ListplayerCurrent.Link,ListplayerCurrent.Role",
  });
  const imageRows = await cargoQuery({
    tables: "PlayerImages=PI,Tournaments=T",
    fields: [
      "PI.Link=PlayerPage",
      "PI.FileName=FileName",
      "PI.Tournament=Tournament",
      "PI.IsProfileImage=IsProfileImage",
      "PI.SortDate=SortDate",
      "T.DateStartFuzzy=TournamentDateStartFuzzy",
      "T.Date=TournamentDate",
      [
        "COALESCE(PI.SortDate,T.DateStartFuzzy,T.Date)",
        "EffectiveSortDate",
      ].join("="),
    ].join(","),
    joinOn: "PI.Tournament=T.OverviewPage",
    where: "PI.IsProfileImage=1",
    orderBy: [
      "PI.Link",
      "COALESCE(PI.SortDate,T.DateStartFuzzy,T.Date)",
      "PI.FileName",
    ].join(","),
  });

  const fileUrls = await resolveFileUrls([
    ...catalogRows.map((row) => row.Image),
    ...imageRows.map((row) => row.FileName),
  ]);

  for (const row of catalogRows) {
    row.PhotoUrl = fileUrls.get(row.Image) || "";
  }
  for (const row of imageRows) {
    row.PhotoUrl = fileUrls.get(row.FileName) || "";
  }

  const rosterRows = [
    ...tournamentRosterRows.map((row) => ({
      ...row,
      EvidenceSource: "TournamentPlayers",
    })),
    ...currentRosterRows.map((row) => ({
      ...row,
      EvidenceSource: "ListplayerCurrent",
    })),
  ];
  const snapshot = {
    schema_version: 2,
    fetched_at_utc: new Date().toISOString(),
    source: {
      type: "leaguepedia_browser_reference",
      api_url: `${location.origin}${API_URL}`,
      fetch_started_at_utc: checkpoint.startedAtUtc,
      catalog_table: "Players",
      evidence_tables: [
        "PlayerLeagueHistory",
        "TournamentPlayers",
        "ListplayerCurrent",
      ],
      unavailable_evidence_tables: unavailableEvidenceTables,
      media_table: "PlayerImages",
      media_sort_policy: [
        "COALESCE(PlayerImages.SortDate,",
        "Tournaments.DateStartFuzzy,Tournaments.Date)",
      ].join(""),
    },
    media_urls_are_direct: true,
    catalog_row_count: catalogRows.length,
    game_evidence_row_count: gameEvidenceRows.length,
    roster_row_count: rosterRows.length,
    image_row_count: imageRows.length,
    catalog_rows: catalogRows,
    game_evidence_rows: gameEvidenceRows,
    roster_rows: rosterRows,
    image_rows: imageRows,
  };
  const blob = new Blob(
    [JSON.stringify(snapshot, null, 2)],
    { type: "application/json" },
  );
  const anchor = document.createElement("a");
  const objectUrl = URL.createObjectURL(blob);
  anchor.href = objectUrl;
  anchor.download = "players_reference.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  console.log("Downloaded players_reference.json", snapshot);
  globalThis[CHECKPOINT_KEY] = undefined;
  } finally {
    checkpoint.running = false;
  }
})();
"""


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""

    parser = argparse.ArgumentParser(
        description=(
            "Đồng bộ player metadata và ảnh từ "
            "Leaguepedia hoặc staging players.csv."
        )
    )
    parser.add_argument(
        "--source",
        choices=("leaguepedia", "snapshot", "csv"),
        default="snapshot",
        help=(
            "Nguồn đầu vào; mặc định snapshot vì Fandom có thể "
            "chặn Python bằng HTTP 403. Leaguepedia/snapshot chỉ "
            "giữ hồ sơ có bằng chứng thi đấu hoặc roster hợp lệ."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Đường dẫn players.csv. Với Leaguepedia "
            "đây là staging output; với csv đây là input."
        ),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT,
        help=(
            "Đường dẫn bundle Leaguepedia player reference "
            "schema_version=2."
        ),
    )
    parser.add_argument(
        "--refresh-media",
        action="store_true",
        help=(
            "Tải lại ảnh ngay cả khi file local "
            "đã tồn tại."
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Chỉ kiểm tra và ghi staging CSV; không tải ảnh "
            "hoặc thay đổi PostgreSQL."
        ),
    )
    parser.add_argument(
        "--refresh-image-sort-dates",
        action="store_true",
        help=(
            "Làm giàu snapshot bằng ngày xếp ảnh chính thức "
            "từ PlayerImages và Tournaments trước khi chọn ảnh."
        ),
    )
    parser.add_argument(
        "--image-sort-output",
        type=Path,
        default=DEFAULT_IMAGE_SORT_SNAPSHOT,
        help=(
            "Raw snapshot riêng của truy vấn ngày xếp ảnh "
            "Leaguepedia."
        ),
    )
    parser.add_argument(
        "--reuse-image-sort-snapshot",
        action="store_true",
        help=(
            "Tái dùng --image-sort-output đã tải xong thay vì gọi "
            "lại Cargo; yêu cầu --refresh-image-sort-dates."
        ),
    )
    parser.add_argument(
        "--refresh-changed-media",
        action="store_true",
        help=(
            "Chỉ tải lại ảnh có photo_url thay đổi so với "
            "players.csv hiện tại và checkpoint để tiếp tục."
        ),
    )
    parser.add_argument(
        "--media-refresh-plan",
        type=Path,
        default=DEFAULT_MEDIA_REFRESH_PLAN,
        help=(
            "Kế hoạch checkpoint cho chế độ chỉ làm mới ảnh "
            "đã thay đổi."
        ),
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help=(
            "Sau upsert, xóa player Leaguepedia không còn "
            "trong tập hợp hợp lệ nếu không có khóa ngoại."
        ),
    )
    parser.add_argument(
        "--prune-media",
        action="store_true",
        help=(
            "Xóa ảnh local của các player đã prune; yêu cầu "
            "--prune-missing."
        ),
    )
    parser.add_argument(
        "--print-browser-exporter",
        action="store_true",
        help=(
            "In đoạn JavaScript thu thập snapshot v2 trong "
            "browser rồi thoát."
        ),
    )

    return parser.parse_args()


def _write_json_atomic(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Ghi checkpoint JSON nguyên tử trong cùng thư mục đích."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")

    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _next_staging_path(path: Path) -> Path:
    """Đặt staging mục tiêu cạnh baseline để promote nguyên tử."""

    return path.with_name(
        f"{path.stem}.next{path.suffix}"
    )


def _archive_previous_sync_manifest(
    *,
    manifest_path: Path,
    plan_path: Path,
) -> tuple[str | None, str | None]:
    """Lưu manifest cũ theo content hash trước khi ghi đè manifest hiện hành."""

    if not manifest_path.is_file():
        return None, None

    manifest_hash = file_sha256(manifest_path)
    archive_path = (
        plan_path.parent
        / "_sync_manifests"
        / f"{manifest_hash}.json"
    )

    if archive_path.is_file():
        if file_sha256(archive_path) != manifest_hash:
            raise ValueError(
                "Manifest archive trùng tên nhưng khác nội dung."
            )
    else:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive_path.with_suffix(
            f"{archive_path.suffix}.part"
        )

        try:
            temporary.write_bytes(manifest_path.read_bytes())
            temporary.replace(archive_path)
        finally:
            temporary.unlink(missing_ok=True)

    return manifest_hash, str(archive_path)


def _create_media_refresh_plan(
    *,
    plan_path: Path,
    baseline_path: Path,
    target_path: Path,
    raw_snapshot_path: Path,
    records: list[PlayerRecord],
    source_details: dict[str, object],
) -> dict[str, object]:
    """Chốt diff URL trước khi tải để có thể tiếp tục an toàn."""

    if not baseline_path.is_file():
        raise FileNotFoundError(
            "Chế độ changed-media yêu cầu players.csv "
            f"baseline: {baseline_path}"
        )

    baseline_records = load_players_csv(
        baseline_path
    )
    baseline_by_id = {
        record.player_id: record
        for record in baseline_records
    }
    target_by_id = {
        record.player_id: record
        for record in records
    }

    if len(target_by_id) != len(records):
        raise ValueError(
            "Target staging chứa player_id trùng."
        )

    if set(baseline_by_id) != set(target_by_id):
        raise ValueError(
            "Chế độ image-only yêu cầu tập player_id của baseline "
            "và target giống hệt nhau."
        )

    non_photo_changes = [
        player_id
        for player_id, target_record in target_by_id.items()
        if (
            target_record.canonical_name,
            target_record.display_name,
            target_record.record_source,
            target_record.record_source_url,
        )
        != (
            baseline_by_id[player_id].canonical_name,
            baseline_by_id[player_id].display_name,
            baseline_by_id[player_id].record_source,
            baseline_by_id[player_id].record_source_url,
        )
    ]

    if non_photo_changes:
        raise ValueError(
            "Chế độ image-only phát hiện thay đổi ngoài photo_url "
            f"ở {len(non_photo_changes)} player."
        )

    baseline_urls = {
        record.player_id: record.photo_url
        for record in baseline_records
    }
    changes = [
        {
            "player_id": record.player_id,
            "previous_photo_url": baseline_urls.get(
                record.player_id
            ),
            "target_photo_url": record.photo_url,
        }
        for record in records
        if record.photo_url is not None
        and baseline_urls.get(record.player_id)
        != record.photo_url
    ]
    removed_photo_ids = [
        record.player_id
        for record in records
        if record.photo_url is None
        and baseline_urls.get(record.player_id) is not None
    ]

    if removed_photo_ids:
        raise ValueError(
            "Bộ chọn mới làm mất URL của "
            f"{len(removed_photo_ids)} player; không tự xóa ảnh "
            "đang dùng nếu chưa có chính sách được xác nhận."
        )

    manifest_path = (
        PROJECT_ROOT
        / "assets"
        / "players"
        / "_sync_manifest.json"
    )
    previous_manifest_hash, previous_manifest_archive = (
        _archive_previous_sync_manifest(
            manifest_path=manifest_path,
            plan_path=plan_path,
        )
    )
    plan: dict[str, object] = {
        "schema_version": 1,
        "status": "active",
        "plan_id": str(uuid4()),
        "policy_version": (
            "leaguepedia_infobox_effective_date_v1"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "baseline_staging_path": str(baseline_path),
        "baseline_staging_sha256": file_sha256(
            baseline_path
        ),
        "target_staging_path": str(target_path),
        "target_staging_sha256": file_sha256(
            target_path
        ),
        "raw_snapshot_path": str(raw_snapshot_path),
        "raw_snapshot_sha256": file_sha256(
            raw_snapshot_path
        ),
        "previous_manifest_sha256": previous_manifest_hash,
        "previous_manifest_archive": previous_manifest_archive,
        "record_count": len(records),
        "changed_count": len(changes),
        "unchanged_count": len(records) - len(changes),
        "removed_photo_player_ids": removed_photo_ids,
        "changes": changes,
        "completed_player_ids": [],
        "failed_player_ids": [],
        "failed_attempts": 0,
        "source_details": source_details,
    }
    _write_json_atomic(plan_path, plan)
    return plan


def _load_active_media_refresh_plan(
    *,
    plan_path: Path,
    baseline_path: Path,
    raw_snapshot_path: Path,
) -> dict[str, object] | None:
    """Nạp plan dở dang và chặn resume nếu nguồn đã thay đổi."""

    if not plan_path.is_file():
        return None

    try:
        plan = json.loads(
            plan_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Media refresh plan không phải JSON hợp lệ."
        ) from error

    if not isinstance(plan, dict):
        raise ValueError(
            "Media refresh plan phải là JSON object."
        )

    status = plan.get("status")

    if status == "complete":
        return None

    if (
        plan.get("schema_version") != 1
        or status not in {"active", "finalizing"}
    ):
        raise ValueError(
            "Media refresh plan có schema/status không hỗ trợ."
        )

    target_value = plan.get("target_staging_path")

    if not isinstance(target_value, str):
        raise ValueError(
            "Media refresh plan thiếu target staging path."
        )

    target_path = Path(target_value)
    baseline_hash = plan.get("baseline_staging_sha256")
    target_hash = plan.get("target_staging_sha256")
    raw_hash = plan.get("raw_snapshot_sha256")

    if (
        not raw_snapshot_path.is_file()
        or not isinstance(raw_hash, str)
        or file_sha256(raw_snapshot_path) != raw_hash
    ):
        raise ValueError(
            "Không thể resume: raw snapshot đã thay đổi hoặc bị thiếu."
        )

    if not isinstance(baseline_hash, str) or not isinstance(
        target_hash, str
    ):
        raise ValueError(
            "Media refresh plan thiếu hash baseline/target."
        )

    current_baseline_hash = (
        file_sha256(baseline_path)
        if baseline_path.is_file()
        else None
    )
    current_target_hash = (
        file_sha256(target_path)
        if target_path.is_file()
        else None
    )

    if status == "active":
        staging_is_valid = (
            current_baseline_hash == baseline_hash
            and current_target_hash == target_hash
        )
    else:
        staging_is_valid = (
            current_baseline_hash == baseline_hash
            and current_target_hash == target_hash
        ) or (
            current_baseline_hash == target_hash
            and current_target_hash in {None, target_hash}
        )

    if not staging_is_valid:
        raise ValueError(
            "Không thể resume: baseline/target staging đã thay đổi "
            "hoặc bị thiếu."
        )

    return plan


def _has_unfinished_media_refresh_plan(plan_path: Path) -> bool:
    """Cho biết checkpoint hiện tại còn active/finalizing hay không."""

    if not plan_path.is_file():
        return False

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "Media refresh plan không phải JSON hợp lệ."
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "Media refresh plan phải là JSON object."
        )

    return payload.get("status") in {"active", "finalizing"}


def _refresh_plan_archive_path(
    *,
    plan_path: Path,
    plan: dict[str, object],
) -> Path:
    """Tạo đường dẫn bất biến để manifest không trỏ vào checkpoint tái dùng."""

    plan_id = plan.get("plan_id")

    if not isinstance(plan_id, str) or not plan_id.strip():
        raise ValueError("Media refresh plan thiếu plan_id.")

    return (
        plan_path.parent
        / "_media_refresh_plans"
        / f"{plan_id}.json"
    )


def _pending_changed_media_records(
    records: list[PlayerRecord],
    plan: dict[str, object],
) -> list[PlayerRecord]:
    """Lấy đúng target records chưa checkpoint thành công."""

    changes = plan.get("changes")
    completed = plan.get("completed_player_ids")

    if not isinstance(changes, list) or not isinstance(
        completed, list
    ):
        raise ValueError(
            "Media refresh plan thiếu changes/completed IDs."
        )

    ordered_ids: list[str] = []

    for change in changes:
        if (
            not isinstance(change, dict)
            or not isinstance(change.get("player_id"), str)
        ):
            raise ValueError(
                "Media refresh plan chứa change không hợp lệ."
            )

        ordered_ids.append(change["player_id"])

    completed_ids = {
        player_id
        for player_id in completed
        if isinstance(player_id, str)
    }
    records_by_id = {
        record.player_id: record
        for record in records
    }
    missing_ids = set(ordered_ids) - set(records_by_id)

    if missing_ids:
        raise ValueError(
            "Target staging thiếu player trong refresh plan: "
            f"{sorted(missing_ids)[:5]}"
        )

    return [
        records_by_id[player_id]
        for player_id in ordered_ids
        if player_id not in completed_ids
    ]


def _checkpoint_media_refresh_plan(
    *,
    plan_path: Path,
    plan: dict[str, object],
    succeeded_ids: tuple[str, ...],
    failed_ids: tuple[str, ...],
) -> None:
    """Checkpoint sau DB commit; lỗi vẫn để pending cho lần chạy sau."""

    completed = {
        player_id
        for player_id in plan.get(
            "completed_player_ids", []
        )
        if isinstance(player_id, str)
    }
    failed = {
        player_id
        for player_id in plan.get(
            "failed_player_ids", []
        )
        if isinstance(player_id, str)
    }
    completed.update(succeeded_ids)
    failed.difference_update(succeeded_ids)
    failed.update(failed_ids)
    plan["completed_player_ids"] = sorted(completed)
    plan["failed_player_ids"] = sorted(failed)
    plan["failed_attempts"] = int(
        plan.get("failed_attempts", 0)
    ) + len(failed_ids)
    plan["updated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    _write_json_atomic(plan_path, plan)


def _resolve_refresh_plan_redirects(
    *,
    plan_path: Path,
    plan: dict[str, object],
    records: list[PlayerRecord],
) -> list[PlayerRecord]:
    """Đổi transport URL redirect sang CDN mà không đổi file ảnh mục tiêu."""

    redirect_prefix = (
        "https://lol.fandom.com/wiki/Special:Redirect/file/"
    )
    redirect_records = [
        record
        for record in records
        if (record.photo_url or "").startswith(redirect_prefix)
    ]

    if not redirect_records:
        return records

    resolved_records = resolve_player_photo_urls(
        redirect_records
    )
    resolved_by_id = {
        record.player_id: record.photo_url
        for record in resolved_records
        if record.photo_url is not None
    }

    if not resolved_by_id:
        return records

    updated_records = [
        (
            PlayerRecord(
                player_id=record.player_id,
                canonical_name=record.canonical_name,
                display_name=record.display_name,
                record_source=record.record_source,
                record_source_url=record.record_source_url,
                photo_url=resolved_by_id[record.player_id],
            )
            if record.player_id in resolved_by_id
            else record
        )
        for record in records
    ]
    target_value = plan.get("target_staging_path")

    if not isinstance(target_value, str):
        raise ValueError(
            "Media refresh plan thiếu target staging path."
        )

    target_path = Path(target_value)
    write_players_csv(target_path, updated_records)
    target_hash = file_sha256(target_path)
    plan["target_staging_sha256"] = target_hash
    changes = plan.get("changes")

    if not isinstance(changes, list):
        raise ValueError("Media refresh plan thiếu changes.")

    for change in changes:
        if not isinstance(change, dict):
            raise ValueError(
                "Media refresh plan chứa change không hợp lệ."
            )

        player_id = change.get("player_id")

        if isinstance(player_id, str) and player_id in resolved_by_id:
            change["target_photo_url"] = resolved_by_id[player_id]

    source_details = plan.get("source_details")

    if isinstance(source_details, dict):
        source_details["staging_sha256"] = target_hash
        source_details["redirect_urls_resolved"] = len(
            resolved_by_id
        )

    plan["redirect_urls_resolved"] = len(resolved_by_id)
    plan["updated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    _write_json_atomic(plan_path, plan)
    print(
        "media_redirects_resolved="
        f"{len(resolved_by_id)}/{len(redirect_records)}"
    )
    return updated_records


def _resolve_player_media(
    records: list[PlayerRecord],
) -> list[PlayerRecord]:
    """Ánh xạ ảnh qua imageinfo và báo tiến độ."""

    print(
        "media_mapping_started="
        f"{len(records)} source=mediawiki_imageinfo"
    )

    resolved = resolve_player_photo_urls(records)

    print(
        "media_mapping_resolved="
        f"{sum(record.photo_url is not None for record in resolved)} "
        "media_mapping_missing="
        f"{sum(record.photo_url is None for record in resolved)}"
    )

    return resolved


def prepare_records(
    source: str,
    input_path: Path,
    raw_output_path: Path,
) -> tuple[list[PlayerRecord], dict[str, object]]:
    """Chuẩn bị records và metadata nguồn cho manifest."""

    if source == "leaguepedia":
        selection = fetch_players(
            snapshot_path=raw_output_path,
        )
        records = selection.records

        if not records:
            raise RuntimeError(
                "Leaguepedia không trả tuyển thủ đủ điều kiện nào."
            )

        if not selection.media_urls_are_direct:
            records = _resolve_player_media(records)

        write_players_csv(
            path=input_path,
            records=records,
        )

        source_details: dict[str, object] = {
            "type": "leaguepedia_cargo",
            "api_url": LEAGUEPEDIA_API_URL,
            "client": LEAGUEPEDIA_CLIENT_NAME,
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
            "selection": asdict(selection.stats),
            "raw_snapshot": str(raw_output_path),
            "raw_sha256": file_sha256(
                raw_output_path
            ),
            "staging_file": str(input_path),
            "staging_sha256": file_sha256(
                input_path
            ),
            "record_count": len(records),
        }

        return records, source_details

    if source == "snapshot":
        if not raw_output_path.is_file():
            raise FileNotFoundError(
                "Khong tim thay player snapshot file: "
                f"{raw_output_path}"
            )

        selection = load_players_snapshot(
            raw_output_path
        )
        records = selection.records

        if not records:
            raise RuntimeError(
                "players snapshot không có tuyển thủ đủ điều kiện."
            )

        if not selection.media_urls_are_direct or any(
            (record.photo_url or "").startswith(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/file/"
            )
            for record in records
        ):
            records = _resolve_player_media(records)

        write_players_csv(
            path=input_path,
            records=records,
        )

        source_details = {
            "type": "leaguepedia_browser_snapshot",
            "api_url": LEAGUEPEDIA_API_URL,
            "client": "browser_fetch",
            "media_url_resolution": (
                "browser_mediawiki_imageinfo"
                if selection.media_urls_are_direct
                else "mediawiki_imageinfo"
            ),
            "catalog_table": "Players",
            "evidence_tables": [
                "PlayerLeagueHistory",
                "TournamentPlayers",
                "ListplayerCurrent",
            ],
            "media_table": "PlayerImages",
            "selection": asdict(selection.stats),
            "raw_snapshot": str(raw_output_path),
            "raw_sha256": file_sha256(
                raw_output_path
            ),
            "staging_file": str(input_path),
            "staging_sha256": file_sha256(
                input_path
            ),
            "record_count": len(records),
        }

        return records, source_details

    if source == "csv":
        if not input_path.is_file():
            raise FileNotFoundError(
                "Không tìm thấy player input file: "
                f"{input_path}"
            )

        records = load_players_csv(input_path)

        if not records:
            raise RuntimeError(
                "players.csv không có player record nào."
            )

        records = _resolve_player_media(records)

        source_details = {
            "type": "curated_csv",
            "input_file": str(input_path),
            "input_sha256": file_sha256(
                input_path
            ),
            "media_url_resolution": (
                "mediawiki_imageinfo"
            ),
            "record_count": len(records),
        }

        return records, source_details

    raise ValueError(
        "Player source không được hỗ trợ: "
        f"{source}"
    )


def print_stats(
    inserted: int,
    updated: int,
    skipped: int,
    media_downloaded: int,
    media_skipped: int,
    deleted: int = 0,
    protected: int = 0,
    media_deleted: int = 0,
) -> None:
    """In kết quả theo hợp đồng chung của sync script."""

    print(
        f"inserted={inserted} "
        f"updated={updated} "
        f"skipped={skipped} "
        f"media_downloaded={media_downloaded} "
        f"media_skipped={media_skipped} "
        f"deleted={deleted} "
        f"protected={protected} "
        f"media_deleted={media_deleted}"
    )


def main() -> None:
    """Đồng bộ metadata trước, sau đó tải media theo transaction ngắn."""

    args = parse_args()
    refresh_changed_media = bool(
        getattr(args, "refresh_changed_media", False)
    )
    refresh_image_sort_dates = bool(
        getattr(
            args,
            "refresh_image_sort_dates",
            False,
        )
    )
    reuse_image_sort_snapshot = bool(
        getattr(
            args,
            "reuse_image_sort_snapshot",
            False,
        )
    )

    if args.print_browser_exporter:
        print(dedent(_BROWSER_EXPORTER).strip())
        return

    if args.prune_media and not args.prune_missing:
        raise ValueError(
            "--prune-media yêu cầu --prune-missing."
        )

    if args.audit_only and (
        args.prune_missing or args.prune_media
    ):
        raise ValueError(
            "--audit-only không được đi cùng cờ prune."
        )

    if args.prune_missing and args.source == "csv":
        raise ValueError(
            "Không prune bằng nguồn csv vì CSV có thể chỉ là "
            "một tập curated con."
        )

    if refresh_changed_media and args.refresh_media:
        raise ValueError(
            "--refresh-changed-media không đi cùng "
            "--refresh-media."
        )

    if refresh_changed_media and args.source != "snapshot":
        raise ValueError(
            "--refresh-changed-media chỉ áp dụng cho "
            "--source snapshot."
        )

    if refresh_changed_media and (
        args.audit_only
        or args.prune_missing
        or args.prune_media
    ):
        raise ValueError(
            "--refresh-changed-media không đi cùng audit/prune."
        )

    if refresh_image_sort_dates and args.source != "snapshot":
        raise ValueError(
            "--refresh-image-sort-dates chỉ áp dụng cho "
            "--source snapshot."
        )

    if reuse_image_sort_snapshot and not refresh_image_sort_dates:
        raise ValueError(
            "--reuse-image-sort-snapshot yêu cầu "
            "--refresh-image-sort-dates."
        )

    input_path = args.input.resolve()
    raw_output_path = args.raw_output.resolve()
    image_sort_output_path = Path(
        getattr(
            args,
            "image_sort_output",
            DEFAULT_IMAGE_SORT_SNAPSHOT,
        )
    ).resolve()
    media_refresh_plan_path = Path(
        getattr(
            args,
            "media_refresh_plan",
            DEFAULT_MEDIA_REFRESH_PLAN,
        )
    ).resolve()
    media_refresh_plan: dict[str, object] | None = None

    if (
        not refresh_changed_media
        and _has_unfinished_media_refresh_plan(
            media_refresh_plan_path
        )
    ):
        raise RuntimeError(
            "Đang có media refresh plan chưa hoàn tất; hãy resume "
            "bằng --refresh-changed-media trước khi chạy sync khác."
        )

    if refresh_changed_media:
        media_refresh_plan = (
            _load_active_media_refresh_plan(
                plan_path=media_refresh_plan_path,
                baseline_path=input_path,
                raw_snapshot_path=raw_output_path,
            )
        )

    if media_refresh_plan is not None:
        target_value = media_refresh_plan.get(
            "target_staging_path"
        )
        source_value = media_refresh_plan.get(
            "source_details"
        )

        if not isinstance(target_value, str) or not isinstance(
            source_value, dict
        ):
            raise ValueError(
                "Media refresh plan thiếu target/source details."
            )

        target_input_path = Path(target_value)
        records_path = (
            target_input_path
            if target_input_path.is_file()
            else input_path
        )
        records = load_players_csv(records_path)

        if media_refresh_plan.get("status") == "active":
            records = _resolve_refresh_plan_redirects(
                plan_path=media_refresh_plan_path,
                plan=media_refresh_plan,
                records=records,
            )
        source_details = dict(source_value)
        print(
            "media_refresh_resumed=1 "
            "completed="
            f"{len(media_refresh_plan.get('completed_player_ids', []))} "
            f"total={media_refresh_plan['changed_count']}"
        )
    else:
        if refresh_image_sort_dates:
            sort_stats = (
                refresh_players_snapshot_image_sort_dates(
                    raw_output_path,
                    query_snapshot_path=(
                        image_sort_output_path
                    ),
                    reuse_query_snapshot=(
                        reuse_image_sort_snapshot
                    ),
                )
            )
            print(
                "image_sort_rows_fetched="
                f"{sort_stats.fetched_rows} "
                "image_sort_rows_matched="
                f"{sort_stats.matched_rows} "
                "image_sort_rows_unmatched="
                f"{sort_stats.unmatched_snapshot_rows} "
                "image_sort_rows_new="
                f"{sort_stats.new_source_rows} "
                "image_sort_rows_duplicate_collapsed="
                f"{sort_stats.duplicate_source_rows_collapsed}"
            )

        target_input_path = (
            _next_staging_path(input_path)
            if refresh_changed_media
            else input_path
        )
        records, source_details = prepare_records(
            source=args.source,
            input_path=target_input_path,
            raw_output_path=raw_output_path,
        )

        if refresh_changed_media:
            media_refresh_plan = (
                _create_media_refresh_plan(
                    plan_path=media_refresh_plan_path,
                    baseline_path=input_path,
                    target_path=target_input_path,
                    raw_snapshot_path=raw_output_path,
                    records=records,
                    source_details=source_details,
                )
            )
            print(
                "media_selection_changed="
                f"{media_refresh_plan['changed_count']} "
                "media_selection_unchanged="
                f"{media_refresh_plan['unchanged_count']}"
            )

    selection = source_details.get("selection")

    if isinstance(selection, dict):
        print(
            "selection_catalog_total="
            f"{selection['catalog_total']} "
            "selection_included="
            f"{selection['included']} "
            "selection_game="
            f"{selection['included_by_game']} "
            "selection_roster_only="
            f"{selection['included_by_roster_only']} "
            "selection_excluded_personality="
            f"{selection['excluded_personality']} "
            "selection_excluded_unverified="
            f"{selection['excluded_without_player_evidence']} "
            "selection_duplicate_handles="
            f"{selection['duplicate_handle_records']} "
            "selection_media_available="
            f"{selection['media_available']}"
            " selection_media_manual="
            f"{selection.get('media_selected_by_manual_override', 0)}"
            " selection_media_sortdate="
            f"{selection.get('media_selected_by_playerimages_sort_date', 0)}"
            " selection_media_tournament_fuzzy="
            f"{selection.get('media_selected_by_tournament_start_fuzzy', 0)}"
            " selection_media_tournament_date="
            f"{selection.get('media_selected_by_tournament_date', 0)}"
            " selection_media_effective_date="
            f"{selection.get('media_selected_by_effective_sort_date', 0)}"
            " selection_media_filename_date="
            f"{selection.get('media_selected_by_filename_date', 0)}"
            " selection_media_revision_date="
            f"{selection.get('media_selected_by_revision_timestamp', 0)}"
        )

    if args.audit_only:
        print_stats(
            inserted=0,
            updated=0,
            skipped=len(records),
            media_downloaded=0,
            media_skipped=sum(
                record.photo_url is not None
                for record in records
            ),
        )
        return

    prune_result = None

    with Session(engine) as session:
        stats = sync_player_metadata(
            session=session,
            records=records,
        )

        if args.prune_missing:
            prune_result = prune_missing_leaguepedia_players(
                session=session,
                keep_player_ids={
                    record.player_id for record in records
                },
            )

        session.commit()

    print(
        "metadata_committed=1 "
        f"inserted={stats.inserted} "
        f"updated={stats.updated} "
        f"skipped={stats.skipped} "
        "deleted="
        f"{prune_result.deleted if prune_result is not None else 0} "
        "protected="
        f"{prune_result.protected if prune_result is not None else 0}"
    )

    if refresh_changed_media:
        if media_refresh_plan is None:
            raise RuntimeError(
                "Thiếu media refresh plan sau khi tính diff."
            )

        media_records = _pending_changed_media_records(
            records,
            media_refresh_plan,
        )
        media_total = int(
            media_refresh_plan["changed_count"]
        )
    else:
        media_records = [
            record
            for record in records
            if record.photo_url is not None
        ]
        media_total = len(media_records)
    media_processed = 0
    media_interrupted = False
    completed_plan_archive_path: Path | None = None

    try:
        for start in range(
            0,
            len(media_records),
            PLAYER_MEDIA_BATCH_SIZE,
        ):
            batch = media_records[
                start : start + PLAYER_MEDIA_BATCH_SIZE
            ]

            with Session(engine) as session:
                batch_stats = sync_player_media(
                    session=session,
                    records=batch,
                    project_root=PROJECT_ROOT,
                    refresh_media=(
                        True
                        if refresh_changed_media
                        else args.refresh_media
                    ),
                )
                session.commit()

            if refresh_changed_media:
                _checkpoint_media_refresh_plan(
                    plan_path=media_refresh_plan_path,
                    plan=media_refresh_plan,
                    succeeded_ids=getattr(
                        batch_stats,
                        "succeeded_player_ids",
                        (),
                    ),
                    failed_ids=getattr(
                        batch_stats,
                        "failed_player_ids",
                        (),
                    ),
                )

            stats.media_downloaded += (
                batch_stats.media_downloaded
            )
            stats.media_skipped += (
                batch_stats.media_skipped
            )
            media_processed += len(batch)

            progress = (
                len(
                    media_refresh_plan.get(
                        "completed_player_ids", []
                    )
                )
                if refresh_changed_media
                else media_processed
            )

            print(
                f"media_progress={progress}/{media_total} "
                "media_downloaded="
                f"{stats.media_downloaded} "
                "media_skipped="
                f"{stats.media_skipped}"
            )
    except KeyboardInterrupt:
        media_interrupted = True

    if refresh_changed_media:
        completed_ids = {
            player_id
            for player_id in media_refresh_plan.get(
                "completed_player_ids", []
            )
            if isinstance(player_id, str)
        }
        changed_count = int(
            media_refresh_plan["changed_count"]
        )
        pending_count = changed_count - len(completed_ids)

        if media_interrupted or pending_count:
            print_stats(
                inserted=stats.inserted,
                updated=stats.updated,
                skipped=stats.skipped,
                media_downloaded=(
                    stats.media_downloaded
                ),
                media_skipped=stats.media_skipped,
            )
            print(
                "media_refresh_incomplete=1 "
                f"completed={len(completed_ids)} "
                f"pending={pending_count} "
                "plan="
                f"{media_refresh_plan_path}"
            )
            raise SystemExit(
                130 if media_interrupted else 2
            )

        target_value = media_refresh_plan.get(
            "target_staging_path"
        )

        if not isinstance(target_value, str):
            raise ValueError(
                "Media refresh plan thiếu target staging path."
            )

        target_input_path = Path(target_value)
        target_hash_value = media_refresh_plan.get(
            "target_staging_sha256"
        )

        if not isinstance(target_hash_value, str):
            raise ValueError(
                "Media refresh plan thiếu target staging hash."
            )

        target_hash = target_hash_value
        completed_plan_archive_path = (
            _refresh_plan_archive_path(
                plan_path=media_refresh_plan_path,
                plan=media_refresh_plan,
            )
        )

        if media_refresh_plan.get("status") != "finalizing":
            media_refresh_plan["status"] = "finalizing"
            media_refresh_plan["finalizing_at_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )
            media_refresh_plan["archive_path"] = str(
                completed_plan_archive_path
            )
            _write_json_atomic(
                media_refresh_plan_path,
                media_refresh_plan,
            )

        current_input_hash = (
            file_sha256(input_path)
            if input_path.is_file()
            else None
        )

        if current_input_hash != target_hash:
            if (
                not target_input_path.is_file()
                or file_sha256(target_input_path) != target_hash
            ):
                raise ValueError(
                    "Không thể promote: target staging đã thay đổi "
                    "hoặc bị thiếu."
                )

            target_input_path.replace(input_path)
        elif target_input_path.is_file():
            target_input_path.unlink()

        source_details["staging_file"] = str(input_path)
        source_details["staging_sha256"] = target_hash
        source_details["previous_sync_manifest_sha256"] = (
            media_refresh_plan.get(
                "previous_manifest_sha256"
            )
        )
        source_details["previous_sync_manifest_archive"] = (
            media_refresh_plan.get(
                "previous_manifest_archive"
            )
        )
        source_details["media_selection_diff"] = {
            "policy_version": media_refresh_plan[
                "policy_version"
            ],
            "changed": changed_count,
            "unchanged": media_refresh_plan[
                "unchanged_count"
            ],
            "removed": len(
                media_refresh_plan.get(
                    "removed_photo_player_ids", []
                )
            ),
            "refresh_plan": str(
                completed_plan_archive_path
            ),
        }
        stats.media_downloaded = len(completed_ids)
        stats.media_skipped = 0

    media_deleted = 0

    if args.prune_media and prune_result is not None:
        media_deleted = delete_pruned_player_media(
            project_root=PROJECT_ROOT,
            relative_paths=prune_result.photo_files,
        )

    source_details["cleanup"] = {
        "prune_missing": args.prune_missing,
        "prune_media": args.prune_media,
        "deleted": (
            prune_result.deleted
            if prune_result is not None
            else 0
        ),
        "protected": (
            prune_result.protected
            if prune_result is not None
            else 0
        ),
        "media_deleted": media_deleted,
    }
    media_sync_details: dict[str, object] = {
        "status": (
            "interrupted"
            if media_interrupted
            else "complete"
        ),
        "batch_size": PLAYER_MEDIA_BATCH_SIZE,
        "total": media_total,
        "processed": (
            media_total
            if refresh_changed_media
            else media_processed
        ),
    }
    if refresh_changed_media:
        media_sync_details["mode"] = "changed_only"
        media_sync_details["failed_attempts"] = int(
            media_refresh_plan.get("failed_attempts", 0)
        )
    source_details["media_sync"] = media_sync_details

    if refresh_changed_media:
        if (
            media_refresh_plan is None
            or completed_plan_archive_path is None
        ):
            raise RuntimeError(
                "Thiếu refresh plan/archive trước khi hoàn tất."
            )

        completed_at_utc = datetime.now(
            timezone.utc
        ).isoformat()
        completed_plan = dict(media_refresh_plan)
        completed_plan["status"] = "complete"
        completed_plan["completed_at_utc"] = completed_at_utc
        completed_plan["promoted_staging_path"] = str(input_path)
        completed_plan["final_manifest_path"] = str(
            PROJECT_ROOT
            / "assets"
            / "players"
            / "_sync_manifest.json"
        )
        completed_plan["source_details"] = source_details
        _write_json_atomic(
            completed_plan_archive_path,
            completed_plan,
        )

    write_sync_manifest(
        asset_dir=PROJECT_ROOT / "assets" / "players",
        source=source_details,
        stats=stats,
    )

    if refresh_changed_media:
        media_refresh_plan.update(completed_plan)
        media_refresh_plan["archive_path"] = str(
            completed_plan_archive_path
        )
        _write_json_atomic(
            media_refresh_plan_path,
            media_refresh_plan,
        )

    print_stats(
        inserted=stats.inserted,
        updated=stats.updated,
        skipped=stats.skipped,
        media_downloaded=stats.media_downloaded,
        media_skipped=stats.media_skipped,
        deleted=(
            prune_result.deleted
            if prune_result is not None
            else 0
        ),
        protected=(
            prune_result.protected
            if prune_result is not None
            else 0
        ),
        media_deleted=media_deleted,
    )

    if media_interrupted:
        print(
            "media_interrupted=1 "
            f"media_processed={media_processed} "
            f"media_total={media_total}"
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
