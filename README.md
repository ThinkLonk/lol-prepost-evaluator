# Match Insight

Hệ thống hỗ trợ khán giả so sánh xác suất thắng của hai đội trong một ván Liên Minh Huyền Thoại chuyên nghiệp trước và sau cấm/chọn.

**Giao diện mới: React + TypeScript + FastAPI.** Xem [hướng dẫn cài đặt, chạy demo và kiến trúc](docs/React_FastAPI.md). Dùng chung dữ liệu PostgreSQL và ba mô hình đã huấn luyện; không cần thu thập dữ liệu hay huấn luyện lại khi chuyển giao diện.

Chạy lệnh tại **thư mục gốc dự án**, nơi có file `README.md`:

```powershell
cd "C:\Đồ án ngành"
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-web.txt
.\scripts\start_web.ps1
```

Thay đường dẫn ở lệnh `cd` nếu dự án nằm ở nơi khác. Mở **http://127.0.0.1:8000**; tài liệu API tại **http://127.0.0.1:8000/docs**. Script cài thư viện frontend bằng lockfile khi cần, build React và chạy FastAPI. Cần cấu hình dữ liệu và mô hình theo mục 3 trước khi phân tích.

## 1. Tóm tắt hệ thống

| Chức năng | Nội dung |
| --- | --- |
| Đánh giá PRE | Ước lượng xác suất thắng từ bối cảnh ván đấu và lịch sử: phong độ, thành tích theo bên, đối đầu và mức độ liên tục đội hình. |
| Đánh giá POST | Giữ nguyên thông tin nền của PRE, bổ sung đội hình mười tướng và lịch sử tuyển thủ–tướng để tính xác suất sau cấm/chọn. |
| So sánh | Ba hàng Baseline, Logistic Regression và Random Forest; các cột PRE, POST và thay đổi theo **điểm phần trăm**. Chuyển tên đội để xem xác suất của bên tương ứng. |
| Lưu và xem lại | Lưu đầu vào, kết quả và cảnh báo trong PostgreSQL; xem lại bằng mã đánh giá. |

Ứng dụng sử dụng Python, pandas và scikit-learn để xử lý dữ liệu và chạy mô hình; PostgreSQL để lưu trữ; React và FastAPI cho giao diện web mới. Bản Streamlit/Plotly vẫn có thể chạy để đối chiếu. Dữ liệu lịch sử đến từ Oracle’s Elixir; Data Dragon hỗ trợ danh mục và hình ảnh tướng. Giao diện hiển thị **Baseline, Logistic Regression và Random Forest** trên cùng đầu vào PRE/POST. Logistic Regression vẫn là mô hình canonical đã được chọn; hai bộ pipeline bổ sung dùng nguyên dữ liệu, split, seed và cách tiền xử lý của lần thực nghiệm đó. Giao diện chỉ nạp pipeline đã fit, không huấn luyện khi mở trang hoặc tạo đánh giá.

Người dùng cung cấp thông tin ván đấu và đội hình cuối cùng. Ứng dụng không tự lấy diễn biến cấm/chọn trực tiếp, không cập nhật xác suất trong ván và không đưa ra khuyến nghị cá cược hoặc chọn tướng. Chênh lệch PRE–POST là thay đổi dự báo của mô hình, không chứng minh tác động nhân quả của đội hình.

## 2. Khởi động và cập nhật giao diện

Khi đã có `.venv`, `.env`, PostgreSQL, mô hình và bản build React:

```powershell
.\scripts\start_web.ps1 -SkipBuild
```

Giữ terminal mở khi sử dụng; nhấn **Ctrl+C** để dừng. Sau khi sửa mã frontend hoặc lấy mã mới từ Git, cần build lại:

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\scripts\start_web.ps1 -SkipBuild
```

Nếu máy chủ đang chạy, chỉ cần hai lệnh npm rồi tải lại trình duyệt bằng **Ctrl+F5** để nhận bản giao diện mới. Nếu mã Python thay đổi, dừng và khởi động lại máy chủ.

Mô hình và lịch sử được nạp một lần mỗi tiến trình. Ứng dụng chạy một worker; khởi động lại máy chủ làm mất phiên phân tích trong RAM, nhưng bản đánh giá đã lưu trong PostgreSQL vẫn tra cứu được bằng mã.

Bản Streamlit cũ vẫn có thể chạy riêng để đối chiếu:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m streamlit run streamlit_app.py
```

Địa chỉ mặc định của bản Streamlit là `http://localhost:8501`.

## 3. Cài đặt trên máy mới

Giải nén bản bàn giao vào thư mục trên máy, mở **Windows PowerShell** tại thư mục chứa `README.md` và `streamlit_app.py`. Chạy lần lượt các lệnh bên dưới; không sao chép phần dấu nhắc `PS ...>` hoặc `>>` từ ảnh terminal. Nếu một bước báo lỗi, xử lý lỗi trước khi chạy bước tiếp theo.

Bản bàn giao cần có mã nguồn, mô hình đã huấn luyện, các file dữ liệu đi kèm và bản sao PostgreSQL được liệt kê ở mục 3.4. Nếu thiếu thành phần nào, cần yêu cầu sinh viên cung cấp bổ sung trước khi chạy.

### 3.1. Chuẩn bị Python, Node.js và PostgreSQL

Cần Node.js và npm để cài/build frontend (môi trường đã chạy thử dùng Node.js 24), Python, PostgreSQL đang hoạt động và tài khoản có quyền đọc dữ liệu, lưu đánh giá. Nếu phục hồi dữ liệu bằng dòng lệnh, cần thêm các công cụ PostgreSQL `psql` và `pg_restore` trong `PATH`; cũng có thể dùng pgAdmin để tạo cơ sở dữ liệu và phục hồi bản sao.

Bộ nạp mô hình kiểm tra chính xác sáu phiên bản dưới đây, lấy từ `artifacts/models/retrospective_pre_post.json`:

| Thành phần | Phiên bản của bộ mô hình đi kèm |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.1 |
| joblib | 1.5.3 |

Kiểm tra Python trước khi tạo môi trường:

```powershell
python --version
```

Để dùng bộ mô hình này, kết quả cần là `Python 3.12.14`. Nếu máy có nhiều bản Python, dùng đường dẫn tới đúng `python.exe` ở bước tạo môi trường. Không thay số phiên bản trong metadata để bỏ qua kiểm tra tương thích.

### 3.2. Tạo môi trường và cài thư viện

Trên máy mới, tạo môi trường riêng bằng Python đã kiểm tra ở mục 3.1. Không dùng lại thư mục `.venv` chép từ máy khác. Nếu máy hiện tại đã có `.venv` đúng phiên bản, bỏ qua lệnh tạo môi trường:

```powershell
python -m venv .venv
```

Kiểm tra Python của môi trường và chuẩn bị pip để cài thư viện:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip --version
```

Kết quả Python cần là `3.12.14`; lệnh cuối cần hiển thị phiên bản pip. Nếu trước đó gặp lỗi `No module named pip`, lệnh `ensurepip` dùng để bổ sung pip vào chính môi trường này. Nếu cả `ensurepip` cũng không có, cần dùng bản Python đầy đủ có hỗ trợ `venv` và `ensurepip` để tạo môi trường.

Cài các thư viện chạy ứng dụng và kiểm tra phụ thuộc:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m pip check
```

Kết quả mong đợi của lệnh kiểm tra là `No broken requirements found.` Đây là kiểm tra phụ thuộc thư viện; cần hoàn thành cấu hình dữ liệu và mở ứng dụng ở các bước tiếp theo để kiểm tra bản bàn giao.

`requirements-web.txt` cài FastAPI, Uvicorn và bao gồm `requirements.txt`. File `requirements.txt` đã cố định phiên bản NumPy, pandas, scikit-learn, SciPy và joblib theo metadata của mô hình, đồng thời khai báo Pillow và `psycopg[binary]`. File này không cài hoặc thay đổi phiên bản Python; môi trường `.venv` vẫn cần dùng Python 3.12.14.

`requirements-dev.txt` dành cho việc phát triển mã nguồn, không cần dùng để cài bản chạy chấm bài.

`psycopg[binary]` cung cấp thành phần kết nối PostgreSQL cho Python trên Windows. Các lệnh gọi trực tiếp Python trong `.venv`, nên không cần chạy `Activate.ps1` hoặc đổi `ExecutionPolicy`. Nếu pip báo không tìm thấy một phiên bản đã cố định, cần gửi lỗi cho sinh viên để kiểm tra và cung cấp môi trường tương thích; không tự đổi phiên bản hay sửa metadata để bỏ qua kiểm tra.

### 3.3. Cấu hình kết nối

Tạo `.env` từ file mẫu nếu chưa có:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

Mở `.env` và sửa dòng kết nối theo PostgreSQL trên máy:

```dotenv
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/lol_prepost
```

Thay `user`, `password`, địa chỉ máy chủ, cổng và tên cơ sở dữ liệu bằng thông tin thực tế. Nếu mật khẩu chứa ký tự đặc biệt như `@`, `#` hoặc `%`, cần mã hóa ký tự đó theo dạng URL. Không đặt dấu nháy bao quanh giá trị trong `.env` vì bộ đọc hiện tại không tự bỏ dấu nháy.

Ứng dụng tự đọc `.env` ở thư mục gốc. Nếu PowerShell đã có biến môi trường `DATABASE_URL`, giá trị đó được ưu tiên hơn `.env`. Hai biến `WIKI_USERNAME_MATCH_INSIGHT` và `WIKI_PASSWORD_MATCH_INSIGHT` chỉ phục vụ các script lấy dữ liệu Leaguepedia; có thể để trống khi chạy giao diện với dữ liệu đã chuẩn bị. Không đưa `.env` chứa mật khẩu lên Git.

### 3.4. Chuẩn bị dữ liệu và bộ mô hình

**Chỉ có mã nguồn và cơ sở dữ liệu rỗng là chưa đủ để tạo PRE/POST.** Cần bộ dữ liệu và mô hình được bàn giao cùng nhau:

| Thành phần | Vị trí hoặc yêu cầu |
| --- | --- |
| Dữ liệu PostgreSQL | Lịch sử ván đấu, danh mục đội/tuyển thủ/tướng, thông tin thời gian và các bảng lưu đánh giá. Dữ liệu lịch sử phải khớp bộ đầu vào đã cố định. |
| Mô hình đã huấn luyện | `artifacts/models/retrospective_pre_post.joblib` |
| Metadata mô hình | `artifacts/models/retrospective_pre_post.json` |
| Manifest bộ ba mô hình | `artifacts/models_comparison/manifest.json` |
| Pipeline Baseline bổ sung | `artifacts/models_comparison/baseline_pre_post.joblib` |
| Pipeline Random Forest bổ sung | `artifacts/models_comparison/random_forest_pre_post.joblib` |
| Hồ sơ thời gian và đầu vào thực nghiệm | `data/reference/retrospective_pre_inputs.json` |
| Hình ảnh | Các thư mục `assets/champions/`, `assets/teams/`, `assets/players/`. Thiếu ảnh không chặn việc phân tích. |

Các file `.joblib`, dữ liệu thô và một số tài nguyên bị loại khỏi Git bằng `.gitignore`. Khi chuyển máy, cần nhận thêm những thành phần này và bản sao PostgreSQL; dự án không có sẵn file sao lưu cơ sở dữ liệu trong mã nguồn.

Nếu đã nhận **bản sao PostgreSQL dạng custom (`pg_dump -Fc`)**, có thể phục hồi vào một cơ sở dữ liệu mới bằng các lệnh sau. Thay tài khoản và đường dẫn mẫu trước khi chạy:

```powershell
psql -h localhost -U postgres -c "CREATE DATABASE lol_prepost;"
pg_restore -h localhost -U postgres -d lol_prepost --no-owner --no-privileges --exit-on-error "C:\duong-dan\lol_prepost.backup"
```

Hai lệnh này tạo và ghi dữ liệu vào PostgreSQL; dùng cho cơ sở dữ liệu mới, chưa có dữ liệu. Với bản sao dạng `.sql`, dùng công cụ phục hồi phù hợp của PostgreSQL hoặc pgAdmin thay cho `pg_restore`.

Kiểm tra phiên bản lược đồ sau khi phục hồi:

```powershell
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic heads
```

Revision mới nhất trong mã nguồn hiện tại là `c6e31a9d4b72`. Bản sao PostgreSQL bàn giao cần có cùng revision này. Nếu kết quả không khớp, yêu cầu sinh viên cung cấp bản sao đúng phiên bản; người chấm không cần tự nâng lược đồ hoặc xử lý chuyển dữ liệu.

Bộ nạp kiểm tra mã băm của dữ liệu lịch sử, hồ sơ thời gian và mô hình. Nếu một thành phần không khớp, ứng dụng sẽ chặn khởi tạo. Việc cập nhật lịch sử hoặc thay mô hình cần thực hiện đồng bộ, không chỉ chép đè một file hay sửa mã băm.

### 3.5. Kiểm tra kết nối và mở ứng dụng

Lệnh sau chỉ thử kết nối và chạy `SELECT 1`, không ghi dữ liệu:

```powershell
.\.venv\Scripts\python.exe -c "from match_insight.database.engine import check_database_connection; print(check_database_connection())"
```

Kết quả mong đợi là `1`. Sau đó chạy `.\scripts\start_web.ps1` để build và mở bản React + FastAPI. Khi nạp thành công, trang **Phân tích ván đấu** hiển thị danh sách đội và tuyển thủ để thiết lập đầu vào. Lần nạp đầu cần đọc lịch sử và mô hình nên có thể lâu hơn các thao tác tiếp theo.

## 4. Luồng demo trên giao diện React

### Bước 1. Thiết lập thông tin ván đấu

1. Chọn hai đội khác nhau ở bên xanh và bên đỏ. Có thể gõ tắt: `HLE` → Hanwha Life Esports; `Gen`, `geng` → Gen.G. Tìm kiếm không phân biệt hoa/thường và bỏ qua dấu câu, khoảng trắng.
2. Ảnh tuyển thủ hiển thị cạnh ô chọn; thiếu ảnh sẽ có chữ viết tắt thay thế. Kiểm tra năm tuyển thủ mỗi đội theo các vị trí Đường trên, Đi rừng, Đường giữa, Xạ thủ và Hỗ trợ. Gợi ý được lấy từ lịch sử, nên cần sửa lại nếu đội hình thực tế khác.
3. Nếu không thấy đội hoặc tuyển thủ cần chọn, bỏ chọn **Chỉ hiện tên có lịch sử**. Bản ghi trong danh mục vẫn có thể chưa có lịch sử liên kết.
4. Nhập **Patch**. Ô **Giải / bối cảnh ván (tùy chọn)** dùng để ghi thêm thông tin ván đấu.
5. Đánh dấu ô xác nhận hai đội, bên thi đấu, patch và tuyển thủ/vị trí do người dùng cung cấp.

Cần đủ mười tuyển thủ khác nhau, mỗi người ở một vị trí. Đây là một phiên phân tích mới, không bắt buộc ván mục tiêu đã tồn tại trong cơ sở dữ liệu. Giao diện hiện không có ô để người dùng tự đặt mốc cắt lịch sử.

### Bước 2. Tạo và xem PRE

Nhấn **Tạo đánh giá PRE**. Hệ thống tổng hợp lịch sử hợp lệ, tính xác suất và lưu bản đánh giá. Bảng hiển thị xác suất PRE của ba mô hình theo đội đang được chọn; các thống kê lịch sử, số mẫu và cảnh báo dùng chung nằm bên dưới. Mở **Thông tin bản lưu và thời gian xử lý** để ghi lại mã PRE nếu muốn tra cứu sau.

Mốc giới hạn lịch sử trong phiên tương tác được tính tại lúc tạo PRE: **00:00 UTC của ngày liền trước ngày tạo PRE theo UTC**. POST giữ nguyên mốc và thông tin nền này. Mốc trên là quy tắc lọc lịch sử, không phải thời điểm bắt đầu cấm/chọn được lấy trực tiếp từ giải đấu.

### Bước 3. Nhập đội hình cuối cùng và tạo POST

Sau khi cấm/chọn hoàn tất, chọn tướng cho từng tuyển thủ ở đủ mười vị trí. Danh sách phải có mười tướng hợp lệ và không trùng nhau. Nhấn **Tạo POST & so sánh** để tính và lưu đánh giá sau cấm/chọn.

Kết quả POST hiển thị xác suất mới, lịch sử tuyển thủ–tướng và cảnh báo. Một cặp chưa có lịch sử được ghi **Chưa ghi nhận**, không tự xem là tỷ lệ thắng 0% hoặc 50%. Cảnh báo thiếu lịch sử liệt kê ngay tên **tuyển thủ — tướng**, đội và vị trí của từng cặp chưa ghi nhận. Mở **Thông tin bản lưu và thời gian xử lý** để lấy mã POST khi cần xem lại.

### Bước 4. Xem so sánh và xử lý khi đổi đầu vào

Sau POST, mỗi hàng mô hình có xác suất PRE, POST và thay đổi có dấu của đội đang chọn. Nhấn tên đội còn lại để đổi hướng xem. Ví dụ cách đọc: từ 54% lên 58% là **tăng 4 điểm phần trăm**. Đây chỉ là ví dụ giải thích đơn vị, không phải kết quả thực nghiệm. Baseline dùng tỷ lệ thắng BLUE học từ tập train nên PRE và POST có thể bằng nhau. Xác suất cao hơn ở một ván không chứng minh mô hình tốt hơn; Brier Score, Log Loss, ROC-AUC và calibration cần được đối chiếu riêng trên cùng tập đánh giá.

| Thao tác thay đổi | Cách tiếp tục |
| --- | --- |
| Chỉ đổi tướng | Nhấn **Chỉnh sửa đội hình**; PRE được giữ, POST cũ mất hiệu lực; chọn lại rồi tạo POST mới. |
| Đổi đội, bên thi đấu, tuyển thủ, vị trí, patch hoặc bối cảnh | Nhấn **Chỉnh sửa bối cảnh**, xác nhận lại đầu vào và tạo PRE mới trước khi tạo POST. |
| Lưu kết quả bị lỗi | Kiểm tra kết nối/quyền ghi rồi thử lại thao tác. Kết quả chỉ được công bố sau khi lưu thành công. |

Số mẫu ít là hạn chế của dữ liệu, không phải bằng chứng tuyển thủ hoặc đội yếu. Cần đọc số mẫu và cảnh báo cùng với xác suất.

### Bước 5. Xem lại đánh giá đã lưu

Mở **Bản đã lưu**, nhập **Mã đánh giá** rồi nhấn **Tra cứu bản lưu**. Mã đánh giá là số được cấp sau khi lưu thành công.

Phần này chỉ hiển thị lại đầu vào, kết quả, cảnh báo và trạng thái bản lưu; không khôi phục phiên để tiếp tục tạo POST. Để phân tích lại, thiết lập đầu vào trong phiên hiện tại. Kết quả ba mô hình và identity từng bộ pipeline được lưu chung trong `input_snapshot.provenance.model_comparison` của đánh giá canonical, cùng transaction. Bản lưu cũ chỉ có một mô hình vẫn được đọc nguyên trạng; ứng dụng không tính thêm hoặc ghi đè dự đoán cũ.

## 5. Lỗi thường gặp

| Hiện tượng | Kiểm tra và xử lý |
| --- | --- |
| `No module named pip` | Chạy `.\.venv\Scripts\python.exe -m ensurepip --upgrade`, sau đó thực hiện lại bước cài `requirements-web.txt` ở mục 3.2. Kích hoạt `.venv` không tự cài pip. |
| `No module named ensurepip` | Bản Python đang dùng thiếu thành phần tạo pip. Cần bản Python đầy đủ đúng phiên bản ở mục 3.1 rồi tạo môi trường riêng trên máy. |
| `No matching distribution found` khi cài thư viện | Gửi lỗi cho sinh viên để kiểm tra bản Python, kho gói và bộ môi trường bàn giao. Không tự sửa các phiên bản đã cố định. |
| Thiếu `DATABASE_URL` hoặc không kết nối được | Kiểm tra `.env`, tài khoản, tên database, cổng và dịch vụ PostgreSQL; chú ý biến môi trường có thể ghi đè `.env`. |
| Chưa có đủ tệp model / không nạp được model | Kiểm tra cặp `.joblib` và `.json` trong `artifacts/models/`, manifest cùng hai `.joblib` trong `artifacts/models_comparison/`, và file `retrospective_pre_inputs.json`. Không thay mô hình bằng số dự đoán giả. |
| Model không tương thích với môi trường | Đối chiếu đủ sáu phiên bản ở mục 3.1, bao gồm phiên bản vá của Python. |
| Dữ liệu hoặc file đầu vào không khớp bản đã duyệt | Dùng đúng bộ PostgreSQL, hồ sơ thời gian và mô hình được bàn giao cùng nhau. Không bỏ qua kiểm tra mã băm. |
| Lần nạp dữ liệu chưa hoàn tất | Khắc phục nguyên nhân rồi tải lại trang. |
| Nút PRE bị khóa | Kiểm tra hai đội khác nhau, đủ mười tuyển thủ khác nhau, patch và ô xác nhận đầu vào. |
| Nút POST bị khóa | Tạo PRE thành công, chọn đủ mười tướng khác nhau; nếu POST đã tồn tại và đầu vào không đổi thì không cần tạo lại. |
| Chưa thể cập nhật trạng thái bản lưu | Khôi phục kết nối/quyền ghi rồi thử lại thao tác chỉnh sửa. |
| Không đọc được đánh giá đã lưu | Kiểm tra mã số và cơ sở dữ liệu đang kết nối; bản ghi có thể nằm ở database khác. |
| Không có ảnh đội, tuyển thủ hoặc tướng | Kiểm tra file trong `assets/` và đường dẫn media đã lưu. Thiếu ảnh không ngăn tạo PRE/POST. |
| Không tìm thấy `start_web.ps1` | Chuyển vào thư mục gốc dự án bằng `cd` trước khi chạy. Dấu nhắc `(.venv)` không có nghĩa đang đứng đúng thư mục. |
| `Unexpected token`, tiếng Việt thành `ChÆ...` trong script | Dùng bản script mới; thông báo trong script đã chuyển sang ASCII để tương thích Windows PowerShell 5.1. |
| Giao diện chưa có thay đổi mới | Build lại frontend rồi nhấn **Ctrl+F5**. `-SkipBuild` chỉ dùng bản build đã tồn tại. |
| Cổng 8000 đang được sử dụng | Nếu ứng dụng đã chạy, mở địa chỉ hiện có; hoặc dùng `.\scripts\start_web.ps1 -SkipBuild -Port 8001` rồi mở `http://127.0.0.1:8001`. |

## 6. Các thư mục chính

| Đường dẫn | Vai trò |
| --- | --- |
| `frontend/src/` | Giao diện React + TypeScript, ô tìm kiếm, ảnh và bảng so sánh. |
| `match_insight/api/` | FastAPI, phiên phân tích, tài nguyên dùng chung và phục vụ ảnh. |
| `scripts/start_web.ps1` | Build frontend và khởi động ứng dụng web. |
| `docs/React_FastAPI.md` | Hướng dẫn kiến trúc, phát triển và vận hành web. |
| `streamlit_app.py` | Điểm khởi động bản Streamlit cũ. |
| `match_insight/ui/` | Nhập thông tin và trình bày kết quả. |
| `match_insight/services/` | Điều phối PRE, POST, so sánh và lưu đánh giá. |
| `match_insight/features/` | Tổng hợp đặc trưng và lọc lịch sử. |
| `match_insight/ml/` | Tạo tập dữ liệu, huấn luyện, nạp và sử dụng mô hình. |
| `match_insight/database/` | Mô hình bảng, đọc lịch sử và lưu kết quả. |
| `match_insight/data_processing/` | Kiểm tra và chuẩn hóa dữ liệu nguồn. |
| `alembic/` | Các phiên bản thay đổi cấu trúc PostgreSQL. |
| `scripts/` | Các lệnh chuẩn bị dữ liệu và thực nghiệm ngoại tuyến. |
| `data/`, `artifacts/`, `assets/` | Dữ liệu đầu vào, mô hình và tài nguyên hình ảnh. |
| `reports/` | Báo cáo kiểm tra dữ liệu và thực nghiệm đã lưu. |

Các script như `apply_oracle_etl.py`, `apply_oracle_riot_v5_backfill.py` và `train_real_models.py` phục vụ chuẩn bị dữ liệu hoặc thực nghiệm, không phải các bước khởi động giao diện. Nếu cần xây lại bộ dữ liệu/mô hình, đọc tham số và điều kiện đầu vào của từng script; các bước này có thể ghi database hoặc tạo file mới. Hướng dẫn trên tập trung chạy và sử dụng bộ hệ thống đã được chuẩn bị.

## 7. Kiểm tra mã nguồn

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -X utf8 -m pytest
npm.cmd --prefix frontend run build
```

Xem [biên bản kiểm chứng React + FastAPI](reports/authored/react_fastapi_acceptance.md) để đọc phạm vi kiểm tra, số đo tốc độ và giới hạn của phép đo.
