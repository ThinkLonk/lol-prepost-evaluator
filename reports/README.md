# Mục lục và nguồn gốc báo cáo

Phân loại theo cách tạo file, không theo phần mở rộng hoặc văn phong. Audit ngày 10/09/2026; mốc UTC ghi trong [provenance.csv](provenance.csv).

## Đọc theo nhóm

| Vị trí | Nội dung | Mức xác minh nguồn tạo |
| --- | --- | --- |
| [data_quality/](data_quality/) | 17 báo cáo/bảng dữ liệu và một `.gitkeep` | 12 file có trình sinh hiện tại; 5 file có trình sinh tại checkpoint Git `05d63d4` |
| [model_runs/retrospective_dry_run_report.json](model_runs/retrospective_dry_run_report.json) | Output kiểm tra điều kiện dữ liệu cho huấn luyện hồi cứu | Cấu trúc phù hợp output runner; thao tác lưu file chưa xác minh |
| [authored/evaluation_persistence_acceptance.md](authored/evaluation_persistence_acceptance.md) | Báo cáo tổng hợp nghiệm thu lưu trữ PRE/POST | Chưa xác định người hay AI trực tiếp viết |

Không có file nào trong **20 file ban đầu** được xác minh là AI trực tiếp viết. Một chương trình có thể do AI hỗ trợ phát triển, nhưng output của nó vẫn được phân loại theo cơ chế sinh file.

Hai file tổ chức mới `README.md` và `provenance.csv` do **Codex (AI) tạo theo PLAN người dùng đã duyệt**, từ kết quả kiểm tra file, code và lịch sử Git. Chúng không phải output audit/ETL/huấn luyện; không nằm trong 20 dòng inventory gốc. CSV được tổng hợp từ metadata, SHA-256 và phân loại của lượt audit này.

## Danh mục báo cáo có trình sinh

| Chủ đề | File trong data_quality | Trình sinh được tìm thấy |
| --- | --- | --- |
| Audit Oracle | [oracle_2025_report.md](data_quality/oracle_2025_report.md), [oracle_2025_summary.json](data_quality/oracle_2025_summary.json), [oracle_2025_issues.csv](data_quality/oracle_2025_issues.csv) | `scripts/audit_oracle.py`; `oracle_audit.write_audit_outputs()` |
| ETL | [oracle_2025_etl_report.md](data_quality/oracle_2025_etl_report.md), [oracle_2025_etl_summary.json](data_quality/oracle_2025_etl_summary.json), [oracle_2025_rejected_records.csv](data_quality/oracle_2025_rejected_records.csv), [oracle_2025_unresolved_identities.csv](data_quality/oracle_2025_unresolved_identities.csv), [oracle_2025_unmapped_champions.csv](data_quality/oracle_2025_unmapped_champions.csv) | `match_insight/data_processing/oracle_etl_report.py` |
| Liên kết ảnh | [media_backfill_report.md](data_quality/media_backfill_report.md) | `scripts/apply_media_backfill.py::generate_markdown_report` |
| Coverage Riot V5 | [oracle_2025_riot_v5_coverage_audit.md](data_quality/oracle_2025_riot_v5_coverage_audit.md), [oracle_2025_riot_v5_coverage_summary.json](data_quality/oracle_2025_riot_v5_coverage_summary.json) | `scripts/audit_oracle_riot_v5_coverage.py` |
| Backfill thời gian | [oracle_2025_riot_v5_backfill_report.md](data_quality/oracle_2025_riot_v5_backfill_report.md) | `scripts/apply_oracle_riot_v5_backfill.py` |
| Khảo sát nguồn thời gian | [oracle_2025_temporal_source_report.md](data_quality/oracle_2025_temporal_source_report.md), [oracle_2025_temporal_source_sample.csv](data_quality/oracle_2025_temporal_source_sample.csv) | `oracle_temporal_source.write_audit_outputs()`, Git `05d63d4` |
| Audit mẫu Riot temporal | [oracle_2025_riot_temporal_report.md](data_quality/oracle_2025_riot_temporal_report.md), [oracle_2025_riot_temporal_sample.csv](data_quality/oracle_2025_riot_temporal_sample.csv) | `oracle_riot_temporal.write_outputs()`, Git `05d63d4` |
| Chính sách thời gian | [oracle_2025_temporal_policy_report.md](data_quality/oracle_2025_temporal_policy_report.md) | `oracle_temporal_policy.write_policy_report()`, Git `05d63d4` |

Tìm thấy trình sinh tương ứng **không chứng minh file hiện tại chưa được chỉnh tay**. Lượt phân loại không chạy lại generator hoặc so khớp bản tái sinh từng byte. Tác giả trực tiếp của các file cũ vẫn được ghi là chưa xác định.

`data_quality/.gitkeep` chỉ giữ thư mục; không phải báo cáo. File CSV chỉ có header như `oracle_2025_unmapped_champions.csv` vẫn được bảo toàn làm bằng chứng, không coi là file dư.

## Đường dẫn đã sắp xếp

Các đường dẫn dưới đây tính từ gốc project:

| Đường dẫn trước | Đường dẫn hiện tại |
| --- | --- |
| `reports/retrospective_dry_run_report.json` | `reports/model_runs/retrospective_dry_run_report.json` |
| `reports/evaluation_persistence_acceptance.md` | `reports/authored/evaluation_persistence_acceptance.md` |

Giữ nguyên byte và tên hai file. Các đường dẫn cũ nằm trong nội dung báo cáo được giữ làm dấu vết lịch sử; dùng bảng này để tra vị trí hiện tại. Không đổi các đường dẫn output của script trong `data_quality`.

Model `.joblib`, metadata model và input đã pin vẫn ở `artifacts/models` và `data/reference`; không chuyển vào `reports`. Báo cáo dry-run hiện có không đại diện tự động cho kết quả train lần 2 hoặc kết quả kiểm tra tái lập.

## Nội dung cần rà soát riêng

- `oracle_2025_riot_v5_coverage_audit.md` ghi 7.552/7.597, tương ứng 99,41%, nhưng có chú thích “Đạt chuẩn 100%”; dòng 45 trường hợp lỗi lại có chú thích “Không có trường hợp lỗi”. Các chú thích này có trong template của script. Đã ghi cờ `REVIEW_FIXED_TEXT_CONFLICT`, chưa sửa báo cáo hoặc code.
- Các trạng thái READY/BLOCKED và quy tắc trong báo cáo temporal là nội dung của từng thời điểm. Chúng không tự thay thế [AGENTS.md](../AGENTS.md) hoặc quyết định trực tiếp hiện hành.
- Báo cáo tổng hợp nghiệm thu chứa kết quả của một lượt trước. Việc xếp thư mục không xác nhận lại database, test, UI hoặc các số liệu đó hôm nay.

“Chương trình sinh” là thông tin về nguồn tạo, không phải chứng nhận mọi kết luận trong báo cáo đều đúng.

## Cách đọc provenance.csv

- `old_path`, `new_path`: đường dẫn tương đối từ gốc project trước/sau sắp xếp.
- `creation_kind`: `GENERATOR_CURRENT`, `GENERATOR_HISTORICAL`, `STRUCTURED_EXPORT`, `AUTHORED_UNVERIFIED` hoặc `DIRECTORY_PLACEHOLDER`.
- `producer_reference`, `producer_revision`: nơi tìm thấy trình sinh hoặc commit của tài liệu; để trống revision nếu chỉ đối chiếu working tree.
- `evidence_level`, `authorship`: mức bằng chứng và giới hạn xác định tác giả.
- `content_review`: nội dung chưa tái kiểm định hoặc cần rà soát riêng.
- `size_bytes`, `sha256`: số đo trước khi sắp xếp, dùng kiểm chứng bảo toàn.
- `audited_at_utc`, `notes`: mốc audit và ghi chú.

Inventory gốc có **20 file, 1.956.488 bytes**. Sau khi tạo mục lục và CSV, thư mục có 22 file nếu không có tác vụ khác sinh thêm output.

Có thể kiểm chứng 20 file gốc bằng PowerShell, từ bất kỳ thư mục làm việc nào; lệnh chỉ đọc:

```powershell
$projectRoot = 'C:\Đồ án ngành'
$rows = Import-Csv -LiteralPath (Join-Path $projectRoot 'reports\provenance.csv')
$checks = foreach ($row in $rows) {
    $path = Join-Path $projectRoot $row.new_path
    $exists = Test-Path -LiteralPath $path -PathType Leaf
    $sizeMatches = $false
    $hashMatches = $false
    if ($exists) {
        $sizeMatches = (Get-Item -LiteralPath $path).Length -eq [long]$row.size_bytes
        $hashMatches = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $row.sha256
    }
    [pscustomobject]@{
        Path = $row.new_path
        Exists = $exists
        SizeMatches = $sizeMatches
        HashMatches = $hashMatches
    }
}
$checks | Format-Table -AutoSize
```

Kết quả mong đợi: đủ 20 dòng, ba cột kiểm tra đều `True`. Đây là kiểm chứng file, không chạy test, ETL, model hoặc kết nối database.

