# Giao diện React và FastAPI

Quyết định công nghệ được người dùng xác nhận ngày 20/09/2026: React + TypeScript + Vite cho giao diện, FastAPI cho HTTP API. Quyết định này thay yêu cầu giao diện Streamlit trong tài liệu thiết kế cũ. Python, pandas, scikit-learn, PostgreSQL và định nghĩa PRE/POST tiếp tục sử dụng các lớp hiện có. Bộ ba mô hình được nạp từ artifact đã chuẩn bị; ứng dụng không huấn luyện lại hoặc thay công thức tiền xử lý.

## Chạy bản trình diễn

Từ thư mục gốc dự án, dùng môi trường Python và PostgreSQL đã cấu hình theo README. Không tạo lại cơ sở dữ liệu nếu đã có dữ liệu bàn giao.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-web.txt
cd frontend
npm.cmd ci
cd ..
.\scripts\start_web.ps1
```

Mở **http://127.0.0.1:8000**. Script build React và phục vụ giao diện cùng API từ một địa chỉ. Sau khi đã build, có thể chạy nhanh bằng:

```powershell
.\scripts\start_web.ps1 -SkipBuild
```

Node.js chỉ cần cho bước cài và build. Dùng Node.js 22.12 trở lên thuộc nhánh còn hỗ trợ. Khi trình diễn bản đã build, chỉ cần tiến trình FastAPI và PostgreSQL. Không cần chạy Streamlit hoặc gọi dịch vụ Internet. Ảnh được đọc từ tài nguyên cục bộ; thiếu ảnh có khung thay thế, không ảnh hưởng dự đoán.

Lần đầu nạp cần có bộ `artifacts/models`, `artifacts/models_comparison`, `data/reference/retrospective_pre_inputs.json`, dữ liệu PostgreSQL và `.env` như README. Nếu thiếu hoặc không khớp, UI báo lỗi và cho thử lại; không thay bằng xác suất giả.

## Luồng trình diễn

1. Chọn hai đội. Mỗi bên gợi ý lực lượng từ một ván lịch sử hoàn chỉnh, có ngày nguồn. Kiểm tra và sửa đủ mười tuyển thủ theo vị trí; gợi ý không xác minh đội hình hiện tại.
2. Nhập patch, ghi bối cảnh nếu cần, đánh dấu ô xác nhận. Nhấn **Tạo đánh giá PRE**.
3. Xem xác suất của ba mô hình trong một bảng. Chuyển tên đội để xem hướng xanh/đỏ. Các giá trị đều do dịch vụ Python trả về.
4. Chọn đủ mười tướng, đúng vị trí cuối cùng. Nhấn **Tạo POST & so sánh**. PRE và mốc lịch sử được giữ nguyên.
5. Đọc PRE, POST và chênh lệch bằng điểm phần trăm. Baseline là tỷ lệ nền của train. Mức thay đổi không được diễn giải như tác động nhân quả của draft.
6. Mở phần lịch sử để xem số thắng/số ván, cặp tuyển thủ–tướng, trạng thái thiếu và độ mới của lịch sử. Mở phần chất lượng để xem Brier, Log Loss và AUC trên validation; số liệu được lấy trực tiếp từ model bundle.
7. Ghi mã PRE/POST, chuyển **Bản đã lưu**, nhập mã để đọc lại. Bản lưu không được dùng để khôi phục một PRE đang thực thi.
8. **Chỉnh sửa đội hình** vô hiệu hóa POST đang giữ, bảo toàn PRE. **Chỉnh sửa bối cảnh** vô hiệu hóa cả PRE và POST. Bản ghi cũ vẫn được giữ trong PostgreSQL để đối chiếu.

## Phân lớp

- `frontend/src`: biểu mẫu, lựa chọn, trạng thái chờ, bảng so sánh và các mục chi tiết. Chỉ định dạng số; không tạo đặc trưng, tính xác suất hay tính chênh lệch nghiệp vụ.
- `match_insight/api/schemas.py`: hợp đồng HTTP, giới hạn trường được nhập. Không nhận thời gian cắt lịch sử, xác suất, đặc trưng hoặc dữ liệu draft trong yêu cầu PRE.
- `match_insight/api/service.py`: giữ tài nguyên đã nạp, phiên phân tích độc lập, gửi lại yêu cầu an toàn, lưu và trình bày bằng các dịch vụ cũ.
- `match_insight/api/app.py`: HTTP, lỗi dễ hiểu, tài nguyên ảnh, bản build React. Tài liệu API tự động tại `/docs`.
- Các mô-đun `services/demo.py`, `services/evaluation.py`, `services/model_comparison.py`, `services/persistence.py`: tiếp tục chịu trách nhiệm nghiệp vụ, dự đoán, xác thực và lưu dữ liệu.

## Trạng thái và lưu kết quả

Mỗi tab giữ một mã phiên ngẫu nhiên trong `sessionStorage`, gửi bằng header `X-Analysis-Session`. Snapshot PRE/POST nằm phía máy chủ; trình duyệt không được truyền một snapshot tự tạo. Trạng thái xem có thể đọc lại sau refresh trong cùng phiên. Chỉ công bố kết quả sau khi lưu thành công.

Mỗi lệnh tạo PRE/POST có `operation_id`. Gửi lại cùng yêu cầu trả kết quả cũ; nếu việc lưu bị gián đoạn, máy chủ giữ nguyên snapshot và mốc thời gian để thử lưu lại, không dự đoán lại. Khi vô hiệu hóa, operation cũ được ghi nhớ để không tái kích hoạt đánh giá đã bỏ.

Mô hình và lịch sử được nạp một lần mỗi tiến trình, dùng cho các phiên. Khóa theo phiên tuần tự hóa thao tác thay đổi; hai phiên không được dùng PRE của nhau. Cache chỉ dùng cho tài nguyên và dữ liệu hiển thị, không bỏ qua kiểm tra nghiệp vụ tại biên PRE/POST/lưu kết quả.

**Phạm vi vận hành của bản này:** ứng dụng trình diễn cục bộ, bind `127.0.0.1`, một worker FastAPI. Phiên đang thao tác lưu trong RAM, hết hạn sau 4 giờ không hoạt động; giới hạn 128 phiên. Khởi động lại máy chủ tạo phiên phân tích mới, trong khi kết quả đã lưu vẫn đọc được từ PostgreSQL bằng mã. Chưa có tài khoản hay phân quyền, nên không mở cổng này trực tiếp ra Internet. Nếu cần nhiều worker hoặc phục hồi phiên đang thực thi sau restart, phải bổ sung cơ chế lưu snapshot phiên tương thích; không chỉ tăng `--workers`.

## Phát triển giao diện

Terminal 1, tại thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m uvicorn match_insight.api.app:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm.cmd run dev
```

Mở `http://127.0.0.1:5173`. Vite chuyển tiếp `/api` về FastAPI. Bản build trình diễn phục vụ cùng origin nên không cần cấu hình CORS mở rộng.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_web_api.py
.\.venv\Scripts\python.exe -X utf8 -m pytest
.\.venv\Scripts\python.exe -X utf8 -m ruff check match_insight/api tests/test_web_api.py
cd frontend
npm.cmd run build
```

Kiểm thử HTTP dùng dữ liệu tổng hợp nhỏ chỉ trong fixture, thực thi các lớp tạo đặc trưng và dự đoán hiện có; giả lập riêng I/O PostgreSQL. Có đối chiếu kết quả ba mô hình với dịch vụ cũ, kiểm tra chặn thiếu/trùng tướng, từ chối clock/feature do client gửi, cách ly phiên, vô hiệu hóa, lỗi lưu và retry, và đọc bản lưu sau restart. Bộ mô hình và dữ liệu thật không bị ghi đè khi chạy test.

## Đo tốc độ

Mỗi phản hồi có `Server-Timing`; PRE/POST trả thời gian phân tích, chuẩn bị chi tiết và lưu dữ liệu ở `timing_ms`, hiển thị trong phần thông tin bản lưu. Thời gian nạp lần đầu nằm trong bootstrap. Đây là thời gian máy chủ, không gồm mạng và vẽ giao diện. Đo riêng lần đầu và các lần đã nạp; không suy ra mô hình tính nhanh hơn chỉ vì đổi React.

Ứng dụng giữ nguyên pipeline mô hình để kiểm chứng chuyển giao diện độc lập với cải thiện chất lượng. Thử tiền xử lý riêng hoặc đổi tập train là thí nghiệm khác, chưa thực hiện trong lần chuyển công nghệ này.

Phần tuần tự hóa dùng cho mã kiểm tra đã được tối ưu thành một lượt duyệt, thay cho việc `asdict` sao chép sâu rồi duyệt lại. Dữ liệu JSON và mã kiểm tra giữ nguyên; kiểm tra tính toàn vẹn vẫn được thực hiện, không dùng cache để bỏ qua xác thực. Bootstrap và phản hồi lớn được nén gzip; ảnh có cache trình duyệt. Xem [biên bản chạy thử](../reports/authored/react_fastapi_acceptance.md) để biết số đo thực tế và giới hạn của phép đo.
