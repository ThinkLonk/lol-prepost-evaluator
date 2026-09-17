# Nghiệm thu lưu trữ PRE/POST trên PostgreSQL

Ngày thực hiện: 08/09/2026. Workspace: C:\Đồ án ngành.
HEAD giữ nguyên: edb670e631aff8e6b5fc0ef07c63faa179709914. Không stage, commit hoặc push.

**Kết quả: đã ghi và đọc lại từ PostgreSQL; import lặp lại không thay đổi dữ liệu.**

## 1. Số liệu thực thi

Trước nhiệm vụ: evaluation = 0, evaluation_warning = 0.

| Nguồn | PRE | POST | Warning |
|---|---:|---:|---:|
| RETROSPECTIVE_IMPORT | 1.150 | 1.150 | 1.329 |
| INTERACTIVE, thao tác nghiệm thu trên UI thật | 2 | 2 | 2 |
| Tổng cuối nhiệm vụ | 1.152 | 1.152 | 1.331 |

Cộng dồn đã nhập mới đủ 1.150 cặp hồi cứu, không từ chối cặp nào. Các lần dừng/tiếp tục trong quá trình hoàn thiện importer giữ nguyên những transaction đã commit. Lần hoàn tất cuối xử lý 832 cặp mới và xác nhận 318 cặp đã có.

**Lần chạy lại đầy đủ:** 1.150 cặp đã tồn tại, 0 evaluation mới, 0 cặp bị từ chối. Số liệu hồi cứu trước/sau đều 2.300 evaluation và 1.329 warning. Thời gian lần chạy lại: 123,05 giây.

Cảnh báo hồi cứu thực tế: W_H2H_MISSING = 460; W_PAIR_HISTORY_MISSING = 765; W_TEAM_HISTORY_SMALL = 104. Hai cảnh báo tương tác là W_PAIR_HISTORY_MISSING trên hai phiên bản POST; không tạo cảnh báo giả để lấp bảng.

## 2. Dữ liệu và contract đã lưu

- Migration hiện hành: c6e31a9d4b72. model_version/data_version dùng Text, không cắt version/identity.
- Phiên tương tác dùng analysis_session và analysis_id; game_id NULL. Kết quả hồi cứu dùng game_id thật; analysis_id NULL. Constraint giữ hai loại chủ thể tách biệt.
- evaluation lưu snapshot bất biến: target, roster/side/role/patch, cutoff và runtime context, feature values/columns/counts/missing/exclusions/used IDs, prediction, PRE seal, POST lineup/comparison, model contract và provenance.
- evaluation_history lưu toàn bộ payload lịch sử và champion reference, dùng chung theo hash nội dung. Hash là dấu kiểm tra; payload đầu vào vẫn được lưu đầy đủ.
- Warning thực tế có warning_code, group, message, severity, position và details chứa ID/counts; được ghi cùng transaction.
- Lưu PRE/POST chỉ trả evaluation ID sau COMMIT. save_pair atomic cho hai phía; retry kiểm nội dung bất biến. Thay context hoặc lineup quản lý is_active, không ghi đè snapshot cũ.
- Snapshot hồi cứu được tái dựng bằng core và model đã chốt; metadata ghi rõ reconstructed_from_existing_sources. Đã đối chiếu context signature, cutoff, feature/model contract và xác suất với artifact, tolerance 1e-12.
- Không có thời điểm suy luận gốc trong file hồi cứu: inferred_at NULL, trạng thái NOT_RECORDED; created_at là thời điểm nhập. PRE tương tác giữ clock thật đã chụp khi tạo PRE. Core POST không ghi clock suy luận riêng nên POST cũng giữ inferred_at NULL, tách khỏi thời điểm lưu.
- Model/context hồi cứu không được đổi thành strict PRE đã xác minh.

## 3. IDs mẫu và kiểm tra UI thật

| Nguồn | Chủ thể | PRE ID | POST ID |
|---|---|---:|---:|
| Hồi cứu | LOLTMNT03_288407 | 2 | 3 |
| Hồi cứu | LOLTMNT03_288421 | 4 | 5 |
| Hồi cứu | LOLTMNT03_288450 | 6 | 7 |
| Tương tác | analysis:f7ecabdb-859e-4929-866f-ab132ce92800 | 494 | 1177, sau đó 1658 |

Phiên tương tác chọn T1 BLUE và Hanwha Life Esports RED độc lập; roster được xác nhận, patch 15.18. Không chọn lại historical target và không chèn game giả.

- PRE 494: p_blue = 0.3702484289258291; T = 2026-09-08T15:46:13.559560Z; cutoff = 2026-09-07T00:00:00Z.
- POST 1177 và 1658: p_blue = 0.2474230840477847; delta_blue = -0.12282534487804439; cùng PRE 494/cutoff/history.
- History pool = 7.552, trước cutoff = 7.552, PRE thực dùng = 48 ván; ngày mới nhất thực dùng 2025-09-28T09:04:05.877Z. UI nêu độ cũ dữ liệu.
- Đổi riêng Lux → Leona: giữ PRE, POST 1177 mất hiệu lực và comparison biến mất. Đổi lại Lux rồi tạo POST: ID mới 1658; không hồi sinh/ghi đè POST 1177.
- Rerun thật không tăng số bản ghi. Đổi context làm mất xác nhận và kết quả cũ; PRE 494 cùng các POST liên quan đều inactive.
- Khởi động lại tiến trình Streamlit và mở phiên trình duyệt mới: đọc POST 1658 được từ DB ngay trước khi nạp model/runtime. Probability, warning, counts, comparison, cutoff và trạng thái inactive khớp. PRE ID 1 của phiên kiểm tra trước đó cũng đã được đọc lại sau restart.
- ID 1 là PRE tương tác có thật của phiên nghiệm thu ban đầu, chưa có POST và còn active. Không xóa bản ghi này để làm đẹp số liệu.
- Ảnh PRE/POST và màn đọc lại được ghi nhận trực tiếp trong hội thoại; không tạo file ảnh/report giả.

Đường đọc lại chỉ trình bày bản đã lưu, không biến bản đọc từ DB thành PRE đang hoạt động để tiếp tục POST.

## 4. SQL kiểm chứng và kết quả

Các câu dưới chạy trong transaction READ ONLY; kiểm tra cuối xác nhận transaction_read_only = on.

~~~sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SHOW transaction_read_only;

SELECT origin, evaluation_type, count(*) AS rows,
       count(*) FILTER (WHERE is_active) AS active_rows
FROM evaluation
GROUP BY origin, evaluation_type
ORDER BY origin, evaluation_type;

SELECT e.origin, w.warning_code, count(*) AS rows
FROM evaluation_warning w JOIN evaluation e USING (evaluation_id)
GROUP BY e.origin, w.warning_code
ORDER BY e.origin, w.warning_code;

SELECT evaluation_id, game_id, analysis_id, evaluation_type,
       pre_evaluation_id, blue_win_probability, history_cutoff_at,
       inferred_at, is_active
FROM evaluation
WHERE evaluation_id IN (1, 2, 3, 494, 1177, 1658)
ORDER BY evaluation_id;

SELECT count(*) AS orphan_warning
FROM evaluation_warning w LEFT JOIN evaluation e USING (evaluation_id)
WHERE e.evaluation_id IS NULL;

SELECT count(*) AS invalid_post_parent
FROM evaluation c LEFT JOIN evaluation p ON p.evaluation_id = c.pre_evaluation_id
WHERE c.evaluation_type = 'POST'
  AND (p.evaluation_id IS NULL OR p.evaluation_type <> 'PRE'
       OR c.game_id IS DISTINCT FROM p.game_id
       OR c.analysis_id IS DISTINCT FROM p.analysis_id
       OR c.context_key IS DISTINCT FROM p.context_key
       OR c.history_cutoff_at IS DISTINCT FROM p.history_cutoff_at
       OR c.model_version IS DISTINCT FROM p.model_version
       OR c.data_version IS DISTINCT FROM p.data_version
       OR c.history_sha256 IS DISTINCT FROM p.history_sha256
       OR c.inference_mode IS DISTINCT FROM p.inference_mode
       OR c.origin IS DISTINCT FROM p.origin
       OR (c.input_snapshot->>'pre_seal') IS DISTINCT FROM
          (p.input_snapshot->>'pre_seal')
       OR (c.is_active AND NOT p.is_active));

SELECT count(*) AS duplicate_request_groups
FROM (
  SELECT idempotency_key FROM evaluation
  GROUP BY idempotency_key HAVING count(*) > 1
) d;
ROLLBACK;
~~~

Kết quả kiểm tra cuối: bad_parent = 0; bad_post_contract = 0; bad_active_parent = 0; orphan_warning = 0; duplicate_request = 0; duplicate_active = 0; bad_subject = 0; snapshot_mismatch = 0; missing_history = 0; invented_inference_time = 0.

Đối chiếu toàn bộ hàng trước/sau replay, không chỉ counts: fingerprint của evaluation, evaluation_warning, evaluation_history và analysis_session đều không đổi. Fingerprint được tính bằng SHA-256 của danh sách khóa và MD5 JSONB từng hàng, theo thứ tự khóa cố định.

## 5. Kiểm thử thực chạy

Các lượt có phần trùng nhau; không cộng thành tổng test độc lập.

| Phạm vi | Kết quả | Warnings |
|---|---|---:|
| test_demo_ui + test_demo_service + test_evaluation_service | 333 PASS | 3.144 |
| test_demo_ui + test_evaluation_persistence sau vá phiên POST | 97 PASS | 1.428 |
| test_evaluation_persistence sau tối ưu validation không đổi packet | 36 PASS | 912 |
| test_import_evaluations, bản cuối | 23 PASS | 0 |
| test_database_schema, DB hiện hành | 13 PASS | 0 |

Đã chạy kiểm thử PostgreSQL trong schema riêng, gồm rollback khi warning/POST lỗi, giữ phiên active cũ khi rollback, FK/context mismatch, immutable guards, idempotency đồng thời và đổi lineup A → B → A. Schema thử nghiệm đã được xóa trong finally.

Warnings hiện hữu từ joblib/numpy_pickle.py:207 về thay shape của NumPy array trong NumPy 2.5; không ẩn warning hoặc đổi dependency.

Ruff exact 13 file code/test/migration: PASS. git diff --check: PASS. Kiểm whitespace cả file tracked/untracked trong phạm vi: PASS. Không chạy lại ETL hoặc train model thật.

## 6. File thay đổi

Sửa:
- C:\Đồ án ngành\match_insight\database\models.py — ORM lưu phiên/snapshot/version/warning.
- C:\Đồ án ngành\match_insight\ui\app.py — lưu trước khi công bố kết quả, retry/invalidation và đọc lại.
- C:\Đồ án ngành\tests\test_demo_ui.py — regression thao tác lưu/đọc/state.
- C:\Đồ án ngành\tests\test_database_schema.py — contract schema hiện hành.

Tạo:
- C:\Đồ án ngành\match_insight\database\evaluations.py — repository transaction và readback.
- C:\Đồ án ngành\match_insight\services\persistence.py — persistence chung từ snapshot nghiệp vụ.
- C:\Đồ án ngành\match_insight\services\import_evaluations.py — đối chiếu/tái dựng/import canonical.
- C:\Đồ án ngành\scripts\import_evaluations.py — CLI dry-run/apply tường minh.
- C:\Đồ án ngành\tests\test_evaluation_persistence.py — snapshot/transaction/rollback/idempotency.
- C:\Đồ án ngành\tests\test_import_evaluations.py — reconstruction/replay/preflight/spawn.
- C:\Đồ án ngành\alembic\versions\c6e31a9d4b72_persist_interactive_evaluations.py — migration storage.

Khôi phục nguyên văn hai migration còn thiếu từ local Git commit 05d63d4f28ad98ec04eeaa7729bb85a7798b2aeb để nối đúng revision DB đang có, không stamp hoặc chạy lại backfill:
- C:\Đồ án ngành\alembic\versions\a64d2f9c1b30_add_evaluation_warning_code.py
- C:\Đồ án ngành\alembic\versions\f2c8a7319d44_add_game_temporal_guard.py

Báo cáo mới: C:\Đồ án ngành\reports\evaluation_persistence_acceptance.md.

## 7. Bảo toàn và vận hành

- Model SHA-256: f83f8cb9190e126462d8f4fc1417c8b5e9edf5a2213c8a3d2d56811f1a31ae9b.
- Metadata JSON SHA-256: 10d8684ed28c591934ca8d441a08b325c5dbee328e5afb1e2c8947a47b9530cd.
- Pinned input SHA-256: 5f35e2784ff3fe83c8c70402a9878af5313cbc780b5aaf4e5317fa9553e630ff.
- Snapshot nguồn SHA-256: 344a5d43b8775c4e6afa61f6e5ccbadeb5e098bcedc53843835b4289b5ecf48d.
- 8.402 game / 16.804 game_team / 84.020 game_player không đổi.
- 15 file được bảo vệ đã kiểm hash cuối: artifact, input/report canonical, .env, AGENTS và backend feature/model/evaluation/demo/reader đều không đổi.
- Dataset/split/bundle identity giữ nguyên canonical; không thay metric validation/test hoặc provenance huấn luyện.
- Không còn schema evaluation_test_*. OS temporary directories riêng do tests tạo đã cleanup; giữ mọi file untracked/thay đổi có sẵn.
- Ứng dụng phục vụ kiểm tra đọc lại: http://127.0.0.1:8518, entrypoint match_insight/ui/app.py.

Lệnh import đã thực chạy, tái chạy an toàn trên cùng nguồn:

~~~powershell
Set-Location -LiteralPath 'C:\Đồ án ngành'
$previousEvaluationPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
try {
    $env:PYTHONPATH = 'C:\Đồ án ngành'
    & 'C:\Đồ án ngành\.venv\Scripts\python.exe' -X utf8 -B 'scripts/import_evaluations.py' --apply --workers 8
    if ($LASTEXITCODE -ne 0) { throw "Evaluation import failed: $LASTEXITCODE" }
} finally {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $previousEvaluationPythonPath, 'Process')
}
~~~

Không có blocker lưu trữ còn lại trong phạm vi đã kiểm. Kết quả chỉ xác nhận lưu/đọc và tính nhất quán, không chứng minh chất lượng model, hỗ trợ mọi patch, context verified pre-draft hoặc toàn bộ identity catalog đã resolve.
