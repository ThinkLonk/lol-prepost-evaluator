# Báo cáo kết quả huấn luyện lại và kiểm tra tính tái lập của mô hình

**Nội dung báo cáo:** Đối chiếu kết quả của hai lượt huấn luyện theo phương pháp mô phỏng hồi cứu.  
**Ngày lập:** 10/09/2026.

## 1. Mục tiêu

Lượt huấn luyện thứ hai nhằm kiểm tra chương trình có tạo lại kết quả của lượt đầu hay không. Hai lượt sử dụng cùng dữ liệu, cách chia tập, tham số mô hình và giá trị khởi tạo ngẫu nhiên. Việc chạy lại trong cùng điều kiện được gọi là kiểm tra tính tái lập.

Trong đề tài, PRE là kết quả ước lượng trước cấm/chọn. POST là kết quả cập nhật sau khi đã xác định đội hình cuối cùng. Cả hai cùng ước lượng xác suất đội bên xanh thắng trong một ván.

Lượt chạy này không bổ sung dữ liệu mới và không thay đổi cấu hình để tìm mô hình tốt hơn. Báo cáo tập trung trả lời ba câu hỏi:

1. Hai lượt có tạo ra cùng tệp mô hình và cùng dự đoán hay không?
2. Chất lượng dự đoán của PRE và POST khác nhau như thế nào?
3. Kết quả hiện có đủ để xác nhận những nội dung nào?

## 2. Phạm vi và nguồn số liệu

### 2.1. Các tệp được đối chiếu

Bộ kết quả ban đầu nằm trong thư mục `artifacts/models/`. Bộ kết quả của lượt huấn luyện thứ hai nằm trong `artifacts/models_reproducibility_run1/`. Tên `run1` chỉ lần kiểm tra tái lập đầu tiên, tức lượt huấn luyện thứ hai nếu tính cả lượt tạo bộ gốc.

Mỗi bộ gồm hai tệp:

- `retrospective_pre_post.joblib`: lưu mô hình đã huấn luyện.
- `retrospective_pre_post.json`: lưu thông tin mô tả lần chạy, cấu hình và kết quả đánh giá. Tệp này được gọi là tệp thông tin mô hình trong báo cáo.

Tệp đầu vào `data/reference/retrospective_pre_inputs.json` cũng được kiểm tra. Tệp này cùng hai tệp mô hình gốc đã được chọn làm mốc đối chiếu và cần giữ nguyên nội dung.

Số liệu được lấy từ [tệp thông tin mô hình gốc](../../artifacts/models/retrospective_pre_post.json) và [tệp thông tin của lượt hai](../../artifacts/models_reproducibility_run1/retrospective_pre_post.json). Cách thực hiện được đối chiếu với [mã huấn luyện](../../match_insight/ml/real_training.py), [mã xây dựng mô hình](../../match_insight/ml/models.py) và [mã tạo tập dữ liệu](../../match_insight/ml/dataset.py).

### 2.2. Cách kiểm tra

Phần đối chiếu chỉ đọc các tệp đã có. Mô hình không được huấn luyện lại trong bước này. Cơ sở dữ liệu PostgreSQL cũng không được truy vấn hoặc thay đổi.

Kích thước tệp và mã SHA-256 được tính trực tiếp từ nội dung tệp. SHA-256 là mã băm dùng để kiểm tra nội dung có thay đổi hay không. Hai tệp có cùng kích thước và cùng mã SHA-256 là bằng chứng mạnh cho thấy nội dung của chúng giống nhau.

Số mẫu, tham số và điểm đánh giá được đọc từ tệp thông tin mô hình. Các bảng chênh lệch trong báo cáo được tính từ những số liệu đó. Phần chưa có nhật ký hoặc kết quả kiểm thử được ghi rõ là chưa xác nhận.

## 3. Dữ liệu và cách chia tập

### 3.1. Phạm vi mô phỏng hồi cứu

Mô phỏng hồi cứu là cách dùng dữ liệu đã xảy ra để dựng lại tình huống đánh giá trong quá khứ. Lượt huấn luyện sử dụng quy tắc có mã `retrospective-utc-day-minus1-v1`.

Quy tắc này xác định mốc đánh giá tại 00:00 của ngày liền trước ngày bắt đầu ván, theo giờ quốc tế UTC. Đây không phải phép trừ đúng 24 giờ từ thời điểm bắt đầu ván.

Tệp thông tin ghi nhận 7.552 ván có bối cảnh PRE được dựng lại từ dữ liệu. Số ván có bối cảnh PRE đã được xác minh là **0**. Vì vậy, báo cáo không khẳng định toàn bộ thông tin đầu vào đã thực sự có sẵn trước cấm/chọn tại thời điểm lịch sử.

Chương trình cũng ghi nhận rằng các thông tin xác nhận từ nguồn bên ngoài chưa được mã nguồn tự xác thực độc lập. Kết quả trong báo cáo chỉ áp dụng cho phương pháp mô phỏng hồi cứu đang xét. Các điều kiện còn thiếu về thời gian của luồng chính vẫn được giữ theo [quy định dự án, mục 5](../../AGENTS.md).

### 3.2. Số lượng mẫu

Theo tệp thông tin mô hình, bản dữ liệu được đọc từ PostgreSQL có 8.402 ván. Trong đó, 7.552 ván lịch sử có trường thời điểm kết thúc `ended_at`. Quá trình chuẩn bị tạo được 7.552 cặp mẫu PRE và POST.

Dữ liệu được chia theo thời gian với tỷ lệ mục tiêu 70%/15%/15%:

- **Tập huấn luyện:** dùng để học tham số mô hình.
- **Tập lựa chọn mô hình:** dùng để so sánh và chọn phương án. Trong mã nguồn, tập này có tên `validation`.
- **Tập kiểm tra cuối:** dùng để báo cáo kết quả sau khi đã chọn phương án. Trong mã nguồn, tập này có tên `test`.

Các ván có cùng mốc đánh giá được giữ trong cùng một phần. Chương trình còn loại những mẫu chưa có kết quả thắng/thua trước ranh giới của phần tiếp theo. Vì vậy, số mẫu cuối cùng không đúng tuyệt đối tỷ lệ mục tiêu.

| Phần dữ liệu | Trước khi loại | Số mẫu bị loại | Số mẫu giữ lại | Bên xanh thắng | Bên đỏ thắng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Huấn luyện | 5.291 | 51 | 5.240 | 2.811 | 2.429 |
| Lựa chọn mô hình | 1.111 | 39 | 1.072 | 565 | 507 |
| Kiểm tra cuối | 1.150 | 0 | 1.150 | 607 | 543 |
| Tổng | 7.552 | 90 | 7.462 | 3.983 | 3.479 |

Có 51 mẫu bị loại khỏi tập huấn luyện vì nhãn kết quả chưa sẵn có trước tập lựa chọn mô hình. Có 39 mẫu bị loại khỏi tập lựa chọn vì lý do tương tự tại ranh giới tập kiểm tra cuối.

Như vậy, **7.552 là số cặp mẫu được tạo ban đầu**, còn **7.462 là số cặp được giữ lại trong ba tập**. Các số lượng này được đọc từ kết quả đã lưu, không phải số đếm mới từ cơ sở dữ liệu.

| Phần dữ liệu | Mốc đánh giá đầu tiên | Mốc đánh giá cuối cùng |
| --- | --- | --- |
| Huấn luyện | 10/01/2025 00:00 UTC | 23/07/2025 00:00 UTC |
| Lựa chọn mô hình | 25/07/2025 00:00 UTC | 21/08/2025 00:00 UTC |
| Kiểm tra cuối | 23/08/2025 00:00 UTC | 15/11/2025 00:00 UTC |

Các ngày trong bảng là mốc đánh giá do quy tắc hồi cứu xác định. Chúng không được hiểu trực tiếp là ngày thi đấu của từng ván.

## 4. Mô hình và cấu hình thực nghiệm

### 4.1. Các phương án so sánh

Thực nghiệm gồm ba phương án:

- **Mô hình cơ sở:** luôn dùng tỷ lệ bên xanh thắng trong tập huấn luyện làm xác suất dự đoán.
- **Hồi quy logistic (Logistic Regression):** học mối liên hệ giữa các đặc trưng đầu vào và xác suất bên xanh thắng.
- **Rừng ngẫu nhiên (Random Forest):** kết hợp nhiều cây quyết định để đưa ra dự đoán.

Đặc trưng là các thông tin được chuyển thành đầu vào cho mô hình. PRE sử dụng nhóm thông tin nền theo cấu hình đã chọn. POST giữ phần nền PRE và bổ sung đội hình cuối cùng cùng thông tin lịch sử tuyển thủ–tướng.

### 4.2. Tham số được ghi nhận

| Thành phần | Cấu hình của cả hai lượt |
| --- | --- |
| Giá trị khởi tạo ngẫu nhiên | 1729 |
| Hồi quy logistic | `C=1.0`, tối đa 1.000 vòng lặp |
| Rừng ngẫu nhiên | 32 cây; độ sâu tối đa 4; mỗi nút lá có ít nhất 2 mẫu |
| Phong độ gần đây trong PRE | 10 ván |
| Thành tích theo bên trong PRE | 20 ván |
| Lịch sử đối đầu trong PRE | 10 ván |
| Số khoảng dùng để kiểm tra xác suất | 10 |

Giá trị khởi tạo ngẫu nhiên giúp lặp lại các thao tác có yếu tố ngẫu nhiên trong cùng điều kiện. Tham số `C` điều khiển mức điều chuẩn của hồi quy logistic, tức mức hạn chế độ lớn của các hệ số đã học. Bảng trên ghi cấu hình thực tế của hai lượt. Nó không chứng minh đây là cấu hình tốt nhất trong mọi phương án.

Hai tệp thông tin mô hình cùng ghi các phiên bản sau:

| Thành phần môi trường | Phiên bản |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.1 |
| joblib | 1.5.3 |

Đây là các phiên bản được chương trình lưu lại. Chưa có kết quả xác nhận việc cài mới toàn bộ môi trường và chạy lại trên máy khác.

## 5. Các chỉ số đánh giá

Báo cáo sử dụng ba chỉ số để đánh giá xác suất bên xanh thắng:

| Chỉ số | Ý nghĩa | Chiều đánh giá |
| --- | --- | --- |
| Brier Score | Trung bình bình phương sai lệch giữa xác suất dự đoán và kết quả thực tế. Kết quả thực tế được mã hóa bằng 1 nếu bên xanh thắng, bằng 0 nếu thua. | Càng thấp càng tốt |
| Log Loss | Đo mức sai lệch của xác suất theo hàm mất mát logarit. Chỉ số phạt mạnh khi mô hình dự đoán rất chắc chắn nhưng sai. | Càng thấp càng tốt |
| ROC-AUC | Diện tích dưới đường cong ROC, phản ánh khả năng xếp các ván bên xanh thắng cao hơn các ván bên xanh thua theo xác suất dự đoán. Đường cong ROC thể hiện tỷ lệ phát hiện đúng và báo nhầm khi thay đổi ngưỡng phân loại. | Càng cao càng tốt |

ROC-AUC không phải tỷ lệ số ván dự đoán đúng. Vì vậy, ROC-AUC bằng 0,654593 không có nghĩa mô hình đạt độ chính xác 65,46%.

Các bảng điểm bên dưới giữ dấu chấm thập phân như tệp kết quả và làm tròn đến chín chữ số. Dấu mũi tên nhắc chiều đánh giá của từng chỉ số.

## 6. Kết quả kiểm tra tính tái lập

### 6.1. Tệp mô hình và thông tin lần chạy

Hai tệp mô hình `.joblib` đều có kích thước 45.086 byte và cùng mã SHA-256. Mã băm lưu trong tệp thông tin cũng khớp mã băm tính trực tiếp từ tệp mô hình tương ứng.

Hai tệp thông tin `.json` đều có kích thước 13.509.878 byte nhưng khác mã SHA-256. Khi so sánh toàn bộ nội dung, chỉ có một trường khác nhau là `run.snapshot_read_at`.

Trường này lưu thời điểm chương trình đọc bản dữ liệu dùng cho lần chạy. Bản dữ liệu được ghi nhận tại một thời điểm còn được gọi là `snapshot` trong mã nguồn.

| Nội dung | Lượt đầu | Lượt thứ hai |
| --- | --- | --- |
| Thời điểm đọc dữ liệu, theo UTC | `2026-09-07T08:37:55.795397+00:00` | `2026-09-10T11:31:31.476289+00:00` |

Sự khác nhau về thời điểm đọc dữ liệu giải thích vì sao hai tệp JSON có mã băm khác nhau. Trường này không cho biết thời gian bắt đầu, thời gian kết thúc hoặc tổng thời lượng huấn luyện.

Ba tệp được chọn làm mốc đối chiếu vẫn khớp các mã băm đã ghi nhận trước đó. Danh sách đường dẫn, kích thước và mã băm đầy đủ nằm ở Phụ lục A.

### 6.2. Dự đoán và điểm đánh giá

Các bản ghi dự đoán được ghép theo mã ván `game_id` trước khi so sánh. Mỗi bộ có 1.150 mã ván riêng biệt và hai bộ có cùng tập mã ván.

| Nội dung kiểm tra | Kết quả |
| --- | --- |
| Số bản ghi dự đoán của mỗi lượt | 1.150 |
| Các bản ghi sau khi ghép theo mã ván | Khớp chính xác |
| Sai khác tuyệt đối lớn nhất của xác suất PRE | 0 |
| Sai khác tuyệt đối lớn nhất của xác suất POST | 0 |
| Sai khác tuyệt đối lớn nhất của mức thay đổi POST − PRE | 0 |
| Cấu hình và các mã nhận diện dữ liệu | Giống nhau |
| Điểm trên tập lựa chọn và tập kiểm tra cuối | Giống nhau |
| Bảng đối chiếu xác suất với tỷ lệ thắng | Giống nhau |

Các giá trị được đối chiếu là dự đoán đã lưu trong tệp thông tin mô hình. Bước kiểm tra này không nạp mô hình để tính lại dự đoán.

Kết quả cho thấy hai bộ tệp lưu cùng mô hình và cùng kết quả đánh giá. Tuy nhiên, việc các tệp giống nhau chưa xác nhận được toàn bộ quá trình chạy trên một môi trường độc lập.

## 7. Kết quả lựa chọn mô hình

Mỗi phương án được đánh giá trên 1.072 mẫu của tập lựa chọn mô hình. Cả hai lượt có cùng kết quả sau:

| Họ mô hình | Cấu hình | Brier Score ↓ | Log Loss ↓ | ROC-AUC ↑ |
| --- | --- | ---: | ---: | ---: |
| Mô hình cơ sở | PRE | 0.249356501 | 0.691860260 | 0.500000000 |
| Mô hình cơ sở | POST | 0.249356501 | 0.691860260 | 0.500000000 |
| Hồi quy logistic | PRE | 0.216487800 | 0.622184942 | 0.708146131 |
| Hồi quy logistic | POST | 0.230702510 | 0.658319381 | 0.674926254 |
| Rừng ngẫu nhiên | PRE | 0.226129713 | 0.643847679 | 0.684809132 |
| Rừng ngẫu nhiên | POST | 0.232658047 | 0.657957425 | 0.701558709 |

Chương trình lựa chọn dựa trên trung bình Brier Score của PRE và POST. Quy tắc còn có ngưỡng cải thiện 0,001 khi xét rừng ngẫu nhiên so với hồi quy logistic. Ngưỡng này nhằm tránh chọn mô hình phức tạp hơn khi mức cải thiện Brier quá nhỏ.

| Họ mô hình | Trung bình Brier PRE/POST |
| --- | ---: |
| Mô hình cơ sở | 0.249356501 |
| Hồi quy logistic | 0.223595155 |
| Rừng ngẫu nhiên | 0.229393880 |

Hồi quy logistic có trung bình Brier Score thấp nhất trong ba phương án. Vì vậy, chương trình chọn họ mô hình này cho bộ kết quả được lưu.

Hồi quy logistic không tốt hơn ở mọi chỉ số riêng lẻ. Với POST, rừng ngẫu nhiên có ROC-AUC cao hơn và Log Loss thấp hơn một chút trên tập lựa chọn. Tuy nhiên, trung bình Brier Score của rừng ngẫu nhiên vẫn cao hơn.

## 8. Kết quả PRE và POST trên tập kiểm tra cuối

### 8.1. So sánh ba chỉ số

Sau khi chọn hồi quy logistic, chương trình đánh giá PRE và POST trên 1.150 ván của tập kiểm tra cuối. Hai lượt có cùng các điểm sau:

| Chỉ số | PRE | POST | POST − PRE | Nhận xét |
| --- | ---: | ---: | ---: | --- |
| Brier Score ↓ | 0.230574573 | 0.245137443 | +0.014562869 | POST kém PRE |
| Log Loss ↓ | 0.651923292 | 0.688146713 | +0.036223422 | POST kém PRE |
| ROC-AUC ↑ | 0.654592978 | 0.629018116 | -0.025574862 | POST kém PRE |

Brier Score của POST tăng khoảng 0,014563 so với PRE. Log Loss tăng khoảng 0,036223. ROC-AUC giảm khoảng 0,025575.

Cả ba thay đổi đều cho thấy POST kém PRE trong cấu hình đang xét. Việc thêm nhóm đặc trưng về đội hình cuối cùng và lịch sử tuyển thủ–tướng chưa giúp cải thiện kết quả trên tập kiểm tra cuối này.

Kết quả trên chưa đủ để kết luận nhóm thông tin POST không có giá trị trong mọi trường hợp. Nguyên nhân có thể liên quan đến số mẫu lịch sử của từng cặp tuyển thủ–tướng hoặc cách biểu diễn đầu vào. Đây mới là hướng cần kiểm tra, chưa phải nguyên nhân đã được chứng minh.

### 8.2. Đối chiếu xác suất với tỷ lệ thắng quan sát được

Chương trình chia dự đoán thành 10 khoảng xác suất. Trong mỗi khoảng, xác suất trung bình được so với tỷ lệ bên xanh thắng thực tế. Cách kiểm tra này cho biết xác suất mô hình đưa ra có gần tỷ lệ quan sát được hay không.

Trong bảng dưới, `n` là số ván và “Xác suất TB” là xác suất trung bình. Tỷ lệ phần trăm được làm tròn đến hai chữ số thập phân.

| Khoảng xác suất | PRE: n | PRE: Xác suất TB | PRE: Tỷ lệ thắng | POST: n | POST: Xác suất TB | POST: Tỷ lệ thắng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–10% | 1 | 9.90% | 0.00% | 16 | 6.75% | 12.50% |
| 10–20% | 29 | 15.50% | 10.34% | 80 | 15.64% | 28.75% |
| 20–30% | 78 | 25.53% | 32.05% | 119 | 24.88% | 45.38% |
| 30–40% | 148 | 35.47% | 39.86% | 159 | 35.46% | 45.28% |
| 40–50% | 289 | 45.42% | 46.37% | 178 | 45.20% | 47.75% |
| 50–60% | 270 | 55.20% | 57.41% | 194 | 54.81% | 55.67% |
| 60–70% | 222 | 64.55% | 64.41% | 181 | 64.80% | 61.88% |
| 70–80% | 89 | 74.49% | 74.16% | 129 | 75.20% | 61.24% |
| 80–90% | 24 | 85.08% | 91.67% | 69 | 84.56% | 73.91% |
| 90–100% | 0 | Không áp dụng | Không áp dụng | 25 | 92.57% | 84.00% |

Với POST, nhóm xác suất 20–30% có 119 ván. Xác suất trung bình là 24,88%, trong khi bên xanh thắng 45,38% số ván. Mô hình đã ước lượng thấp hơn tỷ lệ quan sát được ở nhóm này.

Ở nhóm POST 70–80%, xác suất trung bình là 75,20%. Tỷ lệ thắng thực tế chỉ đạt 61,24% trên 129 ván. Mô hình đã ước lượng cao hơn tỷ lệ quan sát được.

Các nhóm có ít ván cần được đọc thận trọng. Nhóm PRE 90–100% không có mẫu, nên các tỷ lệ được ghi là “Không áp dụng”. Không thể thay giá trị thiếu này bằng 0%.

Bảng trên mô tả kết quả xác suất đã lưu. Nó chưa chứng minh chương trình đã thực hiện một bước điều chỉnh riêng để đưa xác suất gần tỷ lệ thực tế hơn.

## 9. Kết quả kiểm thử phần mềm còn cần bổ sung

Nhật ký kiểm thử được cung cấp cùng quá trình thực hiện ghi nhận:

| Công cụ hoặc nội dung | Kết quả đã có |
| --- | --- |
| Ruff, công cụ kiểm tra quy tắc mã nguồn | Đạt |
| pytest, công cụ chạy kiểm thử tự động | 80 trường hợp đạt, 2 trường hợp không đạt |
| Số cảnh báo | 532 |
| Thời lượng lần chạy kiểm thử | 235,66 giây |

Hai trường hợp không đạt liên quan đến tùy chọn chọn thư mục lưu mô hình khi chạy bằng dòng lệnh. Cả hai đều nhận lỗi `E_TIME_SOURCE_CHANGED`, nghĩa là bước kiểm tra phát hiện nguồn dữ liệu thời gian không khớp.

Một trường hợp mong đợi việc huấn luyện thành công trong thư mục riêng. Trường hợp còn lại mong đợi chương trình chặn khi đã có tệp mô hình trong thư mục đích. Tuy nhiên, cả hai đều bị dừng ở bước kiểm tra nguồn thời gian trước đó.

Chẩn đoán hiện có cho rằng nguồn dữ liệu tạm dùng trong kiểm thử chưa được nối đúng vào đường chạy của chương trình. Cách xử lý cần tập trung vào dữ liệu kiểm thử và đường dẫn nguồn. Không nên bỏ bước kiểm tra nguồn hoặc bỏ cơ chế chống ghi đè để làm kiểm thử đạt.

Trong số cảnh báo có thông báo của joblib về cách đặt hình dạng mảng NumPy không còn được khuyến khích sử dụng. Nhật ký không cho thấy cảnh báo này là nguyên nhân gây ra hai trường hợp kiểm thử không đạt.

Bộ bằng chứng dùng cho báo cáo chưa có kết quả chạy lại xác nhận cả hai trường hợp đã đạt. Vì vậy, báo cáo chưa kết luận phần kiểm thử này đã hoàn tất. Thời lượng 235,66 giây cũng chỉ thuộc lần chạy kiểm thử, không phải thời lượng huấn luyện mô hình.

## 10. Giới hạn của kết quả

Báo cáo xác nhận hai bộ tệp có cùng mô hình và cùng dự đoán đã lưu. Phạm vi xác nhận còn các giới hạn sau:

| Nội dung | Mức độ xác nhận |
| --- | --- |
| Hai tệp mô hình giống nhau | Đã đối chiếu kích thước và SHA-256 |
| 1.150 dự đoán của hai lượt giống nhau | Đã đối chiếu theo mã ván |
| Ba tệp gốc được giữ nguyên tại thời điểm kiểm tra | Đã đối chiếu mã băm với mốc đã ghi nhận |
| Chương trình ghi nhận huấn luyện hoàn tất | Tệp thông tin có trạng thái `TRAINED`, nghĩa là đã huấn luyện |
| Toàn bộ quá trình chạy lượt hai | Chưa có đầy đủ nhật ký, mã kết thúc tiến trình và phiên bản mã nguồn tại thời điểm chạy |
| Huấn luyện lại trên máy hoặc môi trường cài mới | Chưa kiểm tra |
| Cơ sở dữ liệu hiện tại khớp bản dữ liệu đã huấn luyện | Chưa truy vấn lại |
| Luồng ứng dụng PRE → POST → so sánh với bộ lượt hai | Chưa kiểm tra trong lần đối chiếu này |
| Hai trường hợp kiểm thử từng không đạt | Chưa có kết quả chạy lại xác nhận |
| Điều kiện thời gian của luồng chính | Vẫn còn giới hạn do bối cảnh hồi cứu được dựng lại |

Tệp kết quả ghi nhận dữ liệu và phiên bản thư viện giống nhau ở hai lượt. Tuy nhiên, các tệp này chưa đủ để xác minh độc lập mọi thao tác đã thực hiện khi huấn luyện.

Báo cáo cũng chưa có khoảng tin cậy hoặc phép kiểm định thống kê cho chênh lệch PRE và POST. Vì vậy, kết luận chỉ mô tả kết quả quan sát được trên tập dữ liệu đang xét. Mức thay đổi xác suất không được hiểu là tác động nhân quả của đội hình, tướng hoặc tuyển thủ.

Việc mô hình cho kết quả giống nhau chưa chứng minh toàn bộ phần mềm đã đủ điều kiện bàn giao. Yêu cầu tệp nén dưới 1.000.000.000 byte và việc phục hồi dữ liệu trên môi trường nhận vẫn cần được kiểm tra riêng.

## 11. Kết luận và hướng thực hiện tiếp theo

Hai lượt huấn luyện cho cùng mô hình đã lưu và cùng 1.150 dự đoán trên tập kiểm tra cuối. Tệp thông tin chỉ khác thời điểm đọc dữ liệu. Kết quả này xác nhận tính tái lập ở mức các tệp đầu ra đã đối chiếu.

Hồi quy logistic được chọn theo kết quả trên tập lựa chọn mô hình. Trên tập kiểm tra cuối, POST kém PRE ở cả Brier Score, Log Loss và ROC-AUC. Báo cáo giữ kết quả này để phản ánh đúng thực nghiệm.

Bộ mô hình gốc nên tiếp tục được dùng làm mốc của ứng dụng. Bộ lượt hai được giữ làm bằng chứng kiểm tra tái lập. Chưa có lợi ích về chất lượng được chứng minh từ việc thay bộ gốc bằng bộ lượt hai.

Các công việc tiếp theo gồm:

1. Bổ sung nhật ký huấn luyện và phiên bản mã nguồn nếu còn lưu. Phần không có bằng chứng cần được ghi rõ, không suy ngược từ trạng thái hiện tại.
2. Hoàn tất kiểm thử tùy chọn thư mục lưu mô hình và cung cấp kết quả chạy lại.
3. Kiểm tra riêng luồng ứng dụng từ PRE đến POST và phép so sánh.
4. Nếu tiếp tục tìm nguyên nhân POST kém, lập thực nghiệm riêng trên tập huấn luyện và tập lựa chọn mô hình. Không liên tục điều chỉnh theo tập kiểm tra cuối rồi coi tập đó vẫn là đánh giá độc lập.

## Phụ lục A. Tệp và mã SHA-256

Bảng sau lưu thông tin kỹ thuật để kiểm tra lại nội dung tệp. Dấu phẩy trong cột kích thước dùng để tách hàng nghìn.

| Đường dẫn tính từ thư mục gốc dự án | Kích thước (byte) | SHA-256 |
| --- | ---: | --- |
| `artifacts/models/retrospective_pre_post.joblib` | 45,086 | `f83f8cb9190e126462d8f4fc1417c8b5e9edf5a2213c8a3d2d56811f1a31ae9b` |
| `artifacts/models/retrospective_pre_post.json` | 13,509,878 | `10d8684ed28c591934ca8d441a08b325c5dbee328e5afb1e2c8947a47b9530cd` |
| `artifacts/models_reproducibility_run1/retrospective_pre_post.joblib` | 45,086 | `f83f8cb9190e126462d8f4fc1417c8b5e9edf5a2213c8a3d2d56811f1a31ae9b` |
| `artifacts/models_reproducibility_run1/retrospective_pre_post.json` | 13,509,878 | `7fac56bee9664a9f3e6fab25d607d84a45b1220869df89ff3e4a347d5ed49e00` |
| `data/reference/retrospective_pre_inputs.json` | 12,284,666 | `5f35e2784ff3fe83c8c70402a9878af5313cbc780b5aaf4e5317fa9553e630ff` |

Các tệp JSON gốc được giữ nguyên cách mã hóa và nội dung. Mã băm khác nhau giữa hai tệp thông tin được giữ đúng theo kết quả đo.

## Phụ lục B. Mã nhận diện dữ liệu và cấu hình

Các giá trị dưới đây được giữ nguyên như trong tệp thông tin mô hình để tiện tra cứu mã nguồn. Mã của bản dữ liệu, bộ dữ liệu huấn luyện và cách chia tập đều giống nhau ở hai lượt.

| Tên trường trong chương trình | Ý nghĩa |
| --- | --- |
| `snapshot_sha256` | Mã băm của bản dữ liệu được ghi nhận cho lần chạy |
| `evidence_sha256` | Mã băm của tệp đầu vào làm căn cứ |
| `evidence_document_sha256` | Mã kiểm tra nội dung tài liệu đầu vào do chương trình ghi riêng |
| `dataset_id` | Mã nhận diện bộ dữ liệu |
| `split_id` | Mã nhận diện cách chia tập |
| `bundle_version` | Mã phiên bản của bộ mô hình |
| `dataset_signature`, `fit_signature` | Các mã kiểm tra dữ liệu và quá trình học do chương trình lưu |

```text
snapshot_sha256:
344a5d43b8775c4e6afa61f6e5ccbadeb5e098bcedc53843835b4289b5ecf48d

evidence_sha256:
5f35e2784ff3fe83c8c70402a9878af5313cbc780b5aaf4e5317fa9553e630ff

evidence_document_sha256:
141ff30a943af2564e6f97ff3495b29bd84ff11c34bf60c5a268c96b6f353f2e

dataset_id:
retrospective:postgresql:05f268bf0295dfdc79456e14f1a5ea89a79bdadc0713fb1752aa71afce9032b4

split_id:
retrospective:split:fe4467dca8420e39346e9888fb1e99fed32b4ea75c2001cfb4c5d78cd21180d4

bundle_version:
retrospective:7l:c1c82ad44fbe40ada209e6bb3a1d1a741819afdfa5f8152c92bc34b991144613

dataset_signature:
cad2c9d5600778cd7136edd86603fae38149d9a3ca6581d553028697480c4533

fit_signature:
d5dbca1e13bf13fdfb1875371c6b87b53f2e8a1bc42cd5428a0fb44da5426713
```

Hai trường `evidence_sha256` và `evidence_document_sha256` được lưu riêng. Chúng không được dùng thay thế cho nhau khi kiểm tra.

| Nội dung | Giá trị gốc trong chương trình |
| --- | --- |
| Loại dữ liệu: mô phỏng hồi cứu từ dữ liệu thật | `REAL_RETROSPECTIVE_SIMULATION` |
| Quy tắc đánh giá hồi cứu | `retrospective-utc-day-minus1-v1` |
| Cấu trúc tệp mô hình | `real-model-artifact-v1` |
| Nguồn dữ liệu được ghi nhận | `postgresql:official-project` |
| Trạng thái xác thực thông tin từ nguồn ngoài | `EXTERNAL_ASSERTIONS_NOT_AUTHENTICATED_BY_CODE` |
| Phiên bản đặc trưng | `7j-pre-post-from-7i-v1` |
| Phiên bản xử lý đầu vào | `median-empty-zero-standardscale-flags-onehot-ignore-v1` |
| Cấu hình PRE | `pre-v1-r10-s20-h10-roster-latest1` |
| Cấu hình POST | `post-v1-player-champion-all-pre-history` |
| Quy tắc chọn mô hình | `validation-mean-brier-logloss-rf-margin-v1` |

## Phụ lục C. Nguồn kiểm thử và cách biên soạn

Hai trường hợp kiểm thử được nêu ở mục 9 có tên đầy đủ:

- `test_cli_train_with_custom_output_directory`.
- `test_cli_custom_output_directory_preserves_overwrite_guard`.

Trong tên kiểm thử, `CLI` chỉ cách sử dụng chương trình bằng dòng lệnh. Mã lỗi chống ghi đè được mong đợi ở trường hợp thứ hai là `E_MODEL_ARTIFACT_EXISTS`. Mã lệnh liên quan nằm trong [chương trình huấn luyện bằng dòng lệnh](../../scripts/train_real_models.py).