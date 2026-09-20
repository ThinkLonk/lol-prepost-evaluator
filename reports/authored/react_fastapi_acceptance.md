# Chạy thử chuyển giao diện React / FastAPI

Ngày thực hiện: 20/09/2026. Phạm vi: giao diện React + TypeScript + Vite, API FastAPI, dữ liệu PostgreSQL và bộ ba mô hình đã có. Không huấn luyện lại, thay đặc trưng, thay phép chia dữ liệu hoặc thay cách tiền xử lý.

## Kết quả tự động

- Bản build production React/TypeScript hoàn tất; JavaScript khoảng 248 kB, khoảng 78 kB sau gzip, trước truyền HTTP.
- Toàn bộ 76 kiểm thử Python qua sau tối ưu tuần tự hóa; bao gồm 11 trường hợp HTTP mới và 6 trường hợp kiểm tra tính tương đương của tuần tự hóa.
- Ruff qua trên API, phần mô hình được sửa và các test mới.
- `pip check`: không có phụ thuộc xung đột. Có cảnh báo deprecation từ Starlette/httpx và joblib/NumPy; không có test thất bại.
- Kiểm thử HTTP chạy lớp tạo đặc trưng và các mô hình thật của fixture tổng hợp; chỉ giả lập I/O PostgreSQL. Không nhầm số liệu fixture với chất lượng trên dữ liệu esports thật.

## Kiểm tra trực tiếp trên trình duyệt và PostgreSQL

Đã thao tác qua giao diện: chọn T1 và Gen.G, kiểm tra mười tuyển thủ gợi ý, nhập patch 15.18, tạo PRE, chọn mười tướng, tạo POST, đọc ba hàng kết quả và đọc lại bản đã lưu sau khi máy chủ khởi động lại. Đây là bối cảnh nhập để kiểm tra phần mềm, không phải xác nhận một trận sắp diễn ra hoặc một đội hình đang thi đấu thực tế.

Đội hình kiểm tra theo TOP, JUNGLE, MID, BOT, SUPPORT:

- T1: Doran/Aatrox, Oner/Vi, Faker/Ahri, Gumayusi/Jinx, Keria/Nautilus.
- Gen.G: Kiin/Gnar, Canyon/Lee Sin, Chovy/Azir, Ruler/Ashe, Duro/Leona.

Bản trước tối ưu: PRE **2332**, POST **2333**. Bản chạy lại sau tối ưu: PRE **2334**, POST **2335**. Các bản này là đầu ra kiểm tra trong PostgreSQL, không thay đổi lịch sử nguồn hoặc artifact mô hình. Sáu xác suất PRE/POST và tất cả chênh lệch khớp chính xác giữa hai lượt.

| Mô hình | Xác suất T1 thắng PRE | Xác suất T1 thắng POST | Chênh lệch (điểm %) |
| --- | ---: | ---: | ---: |
| Baseline | 53,645038% | 53,645038% | 0 |
| Logistic Regression | 28,445855% | 19,784790% | −8,661064 |
| Random Forest | 43,822145% | 50,766683% | +6,944538 |

Đây là xác suất của một bối cảnh minh họa chạy bằng mô hình thật, không phải chỉ số chất lượng mô hình hoặc kết quả thật của trận. Dữ liệu lịch sử của các đội kết thúc trong năm 2025; UI hiển thị ngày nguồn và không khẳng định phong độ hiện tại.

Kiểm tra giao diện ở khung trình duyệt mặc định và chiều rộng 390 px. Bảng so sánh trên điện thoại chuyển thành từng hàng mô hình với ba cột PRE, POST, Δ, không tràn ngang trang; biểu mẫu xếp hai đội theo chiều dọc. Không ghi nhận lỗi JavaScript trong console tại thời điểm chạy thử. Đã trả viewport về mặc định sau kiểm tra.

## Thời gian quan sát

Số liệu của một lần chạy từng cấu hình trên máy hiện tại, không phải benchmark lặp nhiều lần hoặc p95. Tổng thời gian bước POST trước tối ưu được đọc từ UI; sau tối ưu được lấy từ phản hồi API. Cùng bộ dữ liệu, cặp đội và tướng; phép đo không bao gồm vẽ màn hình, và có thể chịu ảnh hưởng bởi tải máy hoặc cache hệ điều hành.

| Bước POST, thời gian máy chủ | Trước tối ưu | Sau tối ưu |
| --- | ---: | ---: |
| Phân tích và dự đoán | 28.064,4 ms | 12.019,4 ms |
| Chuẩn bị chi tiết | 14.633,3 ms | 6.702,3 ms |
| Kiểm tra và lưu | 16.125,3 ms | 6.583,7 ms |
| **Tổng** | **58.823,0 ms** | **25.305,4 ms** |

Tổng POST giảm khoảng **57% trong ca kiểm tra này**. Lợi ích đến từ tối ưu tuần tự hóa cấu trúc lịch sử trong Python; không quy kết cho React hoặc cho việc đổi thuật toán học máy. POST vẫn mất khoảng 25 giây trên máy đo, chưa thể coi là tức thời.

Các số đo bổ sung sau tối ưu:

- Nạp tài nguyên khi máy chủ khởi động: 4.670,5 ms.
- GET bootstrap sau khi tài nguyên đã nạp: 724,0 ms, đo từ HTTP client cục bộ trước bổ sung gzip.
- PRE, từ gửi yêu cầu đến nhận kết quả: 13.234,2 ms.
- POST, từ gửi yêu cầu đến nhận kết quả: 25.310,5 ms.
- Gửi lại đúng yêu cầu POST đã hoàn thành: 4,3 ms, trả kết quả đã giữ; không dự đoán hoặc lưu thêm.

Đo riêng một lần tạo mã kiểm tra toàn bộ kho lịch sử bằng cProfile: 3.626,2 ms trước và 1.407,3 ms sau; số lời gọi hàm giảm từ khoảng 13,1 triệu xuống 4,1 triệu. Profiler có chi phí riêng, nên không cộng số này vào các thời gian HTTP.

## Bảo toàn nghiệp vụ và giới hạn vận hành

- PRE không nhận champion, feature, xác suất hoặc mốc lịch sử tự đặt từ client.
- POST cần PRE hiện tại của đúng phiên và đủ mười tướng khác nhau.
- Sửa đội hình giữ PRE, vô hiệu hóa POST; sửa bối cảnh vô hiệu hóa cả hai. Bản ghi cũ không bị ghi đè.
- Gửi lại sau lỗi lưu giữ đúng snapshot và mốc cắt ban đầu; yêu cầu đã vô hiệu hóa không được tái kích hoạt.
- Tài nguyên mô hình dùng chung trong tiến trình; trạng thái từng phiên được cách ly.
- Chỉ dùng một worker trên loopback cho bản demo. Trạng thái đang phân tích trong RAM không phục hồi sau restart; các bản đã lưu vẫn đọc từ PostgreSQL.
- Chưa công bố mức tăng chất lượng dự đoán; bộ mô hình và xác suất không đổi.
