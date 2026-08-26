# Quy tắc thi công bắt buộc của dự án

## Mục đích của tệp

Tệp này là bộ quy tắc bắt buộc đối với mọi hoạt động phân tích, thiết kế chi tiết, viết mã, xử lý dữ liệu, huấn luyện mô hình, kiểm thử và xây dựng giao diện của dự án. Mọi thay đổi trong dự án phải tuân thủ các quy tắc dưới đây.

Không được tự ý mở rộng phạm vi, thay đổi ý nghĩa nghiệp vụ hoặc thêm công nghệ chỉ vì thuận tiện cho việc thi công.

## Thứ tự ưu tiên nguồn sự thật

Khi cần xác định yêu cầu, phải đọc và áp dụng các tài liệu theo thứ tự ưu tiên sau:

1. `Thiết Kế.docx` — đặc tả thi công chính.
2. `PhânTích_Lần2.docx` — yêu cầu chức năng, quy tắc nghiệp vụ và mô hình khái niệm.
3. `KhảoSát_Lần3.docx` — nhu cầu người dùng, phạm vi và bối cảnh dữ liệu.
4. `Mô Tả.docx` — mục tiêu tổng quát của đề tài.

Nếu phát hiện nội dung mâu thuẫn giữa các tài liệu:

- Không tự ý chọn một phương án.
- Không âm thầm sửa, bỏ qua hoặc diễn giải lại yêu cầu.
- Phải liệt kê rõ từng nội dung mâu thuẫn, nêu tài liệu và vị trí liên quan, mô tả ảnh hưởng đến thi công rồi yêu cầu người dùng xác nhận.
- Chỉ được tiếp tục phần bị ảnh hưởng sau khi có xác nhận. Những phần độc lập, không phụ thuộc vào mâu thuẫn, vẫn có thể tiếp tục nếu không làm phát sinh quyết định ngầm.

Quyết định đã được người dùng xác nhận trực tiếp là bắt buộc. Riêng cơ sở dữ liệu, phải dùng PostgreSQL thay cho đề xuất SQLite trong `Thiết Kế.docx`.

## Mục tiêu và đơn vị xử lý

- Sản phẩm là hệ thống hỗ trợ khán giả đánh giá tương quan lợi thế giữa hai đội trong một ván Liên Minh Huyền Thoại chuyên nghiệp trước và sau cấm/chọn.
- Đơn vị xử lý là một ván đấu, không phải toàn bộ loạt BO1, BO3 hoặc BO5.
- Hai kết quả PRE và POST phải cùng ước lượng xác suất đội bên xanh thắng để có thể so sánh trực tiếp. Khi hiển thị, phải chuyển bên xanh và bên đỏ thành tên hai đội cụ thể.
- Kết quả là thông tin tham khảo, không phải lời khẳng định chắc chắn đội thắng.
- Hệ thống không thay thế huấn luyện viên hoặc nhà phân tích chiến thuật và không đưa ra khuyến nghị cá cược hay khuyến nghị chọn tướng.

## Luồng nghiệp vụ cốt lõi

Luồng bắt buộc phải được giữ đúng thứ tự:

1. Chọn hoặc xác nhận một ván.
2. Kiểm tra bối cảnh, phạm vi hỗ trợ, phiên bản, hai đội, bên thi đấu, lực lượng và vị trí.
3. Tổng hợp lịch sử hợp lệ có trước mốc đánh giá và tạo đánh giá PRE.
4. Sau cấm/chọn, tiếp nhận hoặc xác nhận đội hình cuối cùng.
5. Kiểm tra đủ mười cặp tuyển thủ–tướng–vị trí, đúng năm cặp cho mỗi đội.
6. Tổng hợp kinh nghiệm tuyển thủ–tướng trong thi đấu chuyên nghiệp và tạo đánh giá POST tham chiếu đúng bản PRE nền.
7. Kiểm tra tính tương thích của PRE và POST.
8. Tính và hiển thị mức thay đổi giữa POST và PRE, thống kê liên quan, số lượng mẫu, phạm vi lịch sử và cảnh báo.

Không được tách PRE, POST và phép so sánh thành các luồng nghiệp vụ không liên quan. PRE là mốc nền; POST là bản cập nhật của đúng ván và đúng mốc nền đó.

## Quy tắc thời gian và chống rò rỉ dữ liệu

- PRE chỉ được sử dụng thông tin đã tồn tại trước khi cấm/chọn bắt đầu.
- POST phải giữ nguyên toàn bộ thông tin nền hợp lệ của PRE và chỉ bổ sung đội hình cuối cùng cùng lịch sử tuyển thủ–tướng chuyên nghiệp.
- Mọi đặc trưng lịch sử của một ván mục tiêu chỉ được tính từ các ván đã kết thúc trước mốc đánh giá của ván mục tiêu.
- Phải dùng thời điểm kết thúc của ván lịch sử để xác định tính hợp lệ; không được chỉ dựa vào thứ tự dòng hoặc tên tệp.
- Tuyệt đối không dùng kết quả hoặc chỉ số diễn biến trong chính ván đang đánh giá làm đầu vào mô hình, bao gồm vàng, mạng hạ gục, sát thương, trụ, rồng, Baron và các mục tiêu khác.
- Kết quả thật của ván mục tiêu chỉ được dùng làm nhãn sau trận, phục vụ đánh giá phương pháp và trở thành lịch sử cho các ván xảy ra sau.
- Không được dùng bất kỳ thống kê nào đã vô tình tổng hợp cả ván mục tiêu hoặc dữ liệu tương lai.
- Việc chia dữ liệu, tính lịch sử, thay thế dữ liệu thiếu, mã hóa, chuẩn hóa, lựa chọn đặc trưng, chọn tham số và hiệu chỉnh xác suất đều phải ngăn thông tin từ validation hoặc test truyền ngược vào train.
- Bên thi đấu và thứ tự lựa chọn là hai thông tin khác nhau khi giải áp dụng First Selection. Không được suy rằng bên xanh luôn chọn trước.
- Các quy tắc thời gian phải được tập trung ở lớp xử lý hoặc tạo đặc trưng có thể kiểm thử; không được rải rác dưới dạng điều kiện tạm thời trong giao diện.

## Điều kiện và nội dung của PRE

PRE chỉ được tạo chính thức khi đã xác nhận tối thiểu:

- ván mục tiêu và thời điểm đánh giá;
- giải, giai đoạn và phiên bản;
- đúng hai đội và bên thi đấu;
- đủ năm tuyển thủ cùng vị trí cho mỗi đội;
- phạm vi áp dụng của phương pháp;
- lịch sử đầu vào không chứa dữ liệu tương lai.

Nhóm thông tin PRE có thể gồm phong độ gần đây, thành tích theo bên, lịch sử đối đầu và mức độ liên tục lực lượng. Các thống kê này phải do hệ thống tính từ dữ liệu lịch sử đã kiểm tra; người dùng không được nhập thủ công tỷ lệ hoặc phong độ.

Các cửa sổ 5, 10 và 20 ván chỉ là phương án thử nghiệm cho phong độ gần đây. Việc chọn cửa sổ phải dựa trên tập phát triển hoặc validation và phải được cố định trước khi đánh giá trên test cuối.

Không được đưa tướng cấm, tướng chọn, đội hình cuối cùng hoặc bất kỳ dữ liệu phát sinh trong ván mục tiêu vào PRE.

Nếu lực lượng hoặc thông tin nền bắt buộc thay đổi sau khi PRE được tạo nhưng trước khi so sánh, phải vô hiệu hóa PRE cũ và tạo PRE mới. Không được dùng POST mới với PRE có nền thông tin cũ.

## Điều kiện và nội dung của POST

- POST chỉ được tạo sau khi cấm/chọn kết thúc nhưng trước khi ván bắt đầu.
- POST chỉ được tạo khi xác định đủ mười tướng, mười tuyển thủ sử dụng, đội sở hữu và vị trí cuối cùng.
- Việc gắn tướng với tuyển thủ phải dựa trên phân công cuối cùng, không được suy hoàn toàn từ thứ tự chọn.
- POST phải giữ nguyên thông tin nền, lực lượng, phạm vi lịch sử, biến kết quả, hướng tham chiếu và thang đo của PRE nền.
- POST chỉ bổ sung đội hình cuối cùng và lịch sử chuyên nghiệp của các cặp tuyển thủ–tướng.
- Chuỗi cấm/chọn chi tiết chỉ được dùng để đối chiếu hoặc minh họa khi nguồn đầy đủ; không phải đầu vào bắt buộc và không được biến thành phân tích chiến thuật sâu trong phạm vi cốt lõi.
- Nếu thiếu bất kỳ cặp tuyển thủ–tướng–vị trí nào, phải giữ PRE, chặn POST và không tính chênh lệch.
- Chưa có lịch sử của một cặp tuyển thủ–tướng không đồng nghĩa tuyển thủ có năng lực bằng không. Pipeline có thể tiếp tục nếu phương pháp đã được thiết kế để xử lý thiếu, đồng thời phải ghi cờ thiếu, số mẫu và cảnh báo phù hợp.

## Quy tắc so sánh PRE và POST

Chỉ được so sánh khi hai bản đánh giá có cùng:

- ván mục tiêu;
- hai đội và bên thi đấu;
- lực lượng và vị trí;
- phiên bản và phạm vi lịch sử nền;
- biến kết quả, hướng tham chiếu và thang đo;
- phiên bản dữ liệu và phương pháp tương thích;
- dấu vết đầu vào nền tương thích.

Mức thay đổi được tính bằng xác suất POST trừ xác suất PRE của bên xanh. Khi hiển thị, phải diễn đạt hướng thay đổi theo tên đội cụ thể.

Chỉ dùng ba nhóm diễn giải: gần như không thay đổi, nghiêng thêm về đội bên xanh hoặc nghiêng thêm về đội bên đỏ. Ngưỡng “gần như không thay đổi” phải được lựa chọn trên validation và cố định trước khi mở test cuối.

Không được diễn giải chênh lệch PRE–POST như tác động nhân quả của draft, của một tướng, của một tuyển thủ hoặc của một lựa chọn cụ thể. Chỉ được nói rằng đánh giá của hệ thống thay đổi khi có thêm nhóm thông tin sau cấm/chọn.

## Dữ liệu thiếu, giá trị bằng không và không áp dụng

- Dữ liệu thiếu, giá trị bằng 0 và trạng thái không áp dụng là ba trạng thái khác nhau và phải được lưu, xử lý, kiểm thử và hiển thị khác nhau.
- Chưa có lịch sử đối đầu phải được ghi là chưa có dữ liệu đối đầu, với số mẫu bằng 0 và cờ thiếu; không được thay bằng tỷ lệ 50%.
- Chưa có lịch sử tuyển thủ–tướng phải được ghi là chưa ghi nhận lịch sử; không được thay bằng 0%, 50% hoặc một tỷ lệ giả định khác.
- Khi số ván bằng 0, không được áp dụng công thức làm trơn tỷ lệ thắng cho cặp tuyển thủ–tướng.
- Mọi tỷ lệ lịch sử phải đi kèm số thắng, số ván hoặc phạm vi mẫu phù hợp.
- Mẫu nhỏ là vấn đề độ bao phủ, không phải bằng chứng về chất lượng hay năng lực. Ngưỡng mẫu nhỏ phải được chọn bằng train/validation và cố định trước test cuối.
- Xác suất gần 50% chỉ biểu thị hệ thống chưa nhận thấy lợi thế rõ; không tự động đồng nghĩa dữ liệu kém chất lượng.
- Xác suất xa 50% không chứng minh dữ liệu đầy đủ hoặc dự đoán đáng tin cậy.
- Trên giao diện, dùng các nhãn rõ như “Chưa ghi nhận” và “Không áp dụng”; không điền số thay thế để làm giao diện có vẻ đầy đủ.

## Dữ liệu và nguồn dữ liệu

- Oracle’s Elixir là ứng viên nguồn lịch sử chuyên nghiệp chính.
- Riot Data Dragon chỉ dùng để chuẩn hóa mã, tên và tài nguyên hiển thị của tướng; không dùng thay cho nguồn xác định lịch sử thi đấu hoặc phiên bản thi đấu chính thức.
- LoL Esports và Games of Legends dùng để xác định bối cảnh hoặc kiểm tra chéo trường hợp bất thường; không được tự ý biến thành phụ thuộc bắt buộc của pipeline hàng loạt.
- Khi thiếu `playerid` hoặc `teamid`, có thể đối chiếu tên, đội, vị trí và thời gian. Trường hợp còn mơ hồ không được tự gộp.
- Không dùng tên hiển thị làm định danh duy nhất cho đội hoặc tuyển thủ.
- Ánh xạ đổi tên, chuyển đội và quan hệ thành viên phải có khoảng thời gian hiệu lực.
- Khi ghép nguồn không có mã ván chung, phải đối chiếu đồng thời giải, thời gian, hai đội, số thứ tự ván, phiên bản và lực lượng hoặc đội hình khi có.
- Phải lưu nguồn, ngày tải, phạm vi dữ liệu và phiên bản dữ liệu đủ để truy vết.
- Nhãn `complete` hoặc nhãn đầy đủ chung của nguồn không thay thế việc kiểm tra từng trường cần cho PRE và POST.
- Thiếu lượt cấm không chặn phạm vi cốt lõi. Thiếu trường chọn ở dòng đội cũng không chặn nếu các dòng tuyển thủ vẫn xác định chắc chắn đủ đội hình cuối cùng.

## Nhánh mở rộng OP.GG

- OP.GG và dữ liệu Ranked Solo/Duo chỉ là phần mở rộng nghiên cứu, không thuộc điều kiện tối thiểu của luồng PRE/POST cốt lõi.
- Thiếu OP.GG, lỗi truy cập OP.GG hoặc không xác minh được tài khoản tuyệt đối không được làm gián đoạn PRE, POST hoặc phép so sánh cốt lõi.
- Chỉ dùng dữ liệu Solo/Duo với `queueId = 420`.
- Chỉ dùng khi đã xác minh đúng tuyển thủ, tài khoản, khu vực, mùa, hàng đợi và thời điểm snapshot.
- Snapshot phải tồn tại tại hoặc trước mốc đánh giá; không được dùng snapshot thu thập sau để suy ngược cho ván quá khứ.
- Thống kê OP.GG phải được gắn nhãn và trình bày tách biệt với lịch sử thi đấu chuyên nghiệp.
- Không được triển khai nhánh OP.GG trước khi luồng cốt lõi hoạt động độc lập và người dùng yêu cầu hoặc xác nhận phạm vi mở rộng.

## Phạm vi không được triển khai

Không triển khai trong phạm vi cốt lõi:

- dự đoán hoặc cập nhật xác suất trong trận;
- dữ liệu vàng, mạng, sát thương, trụ, rồng, Baron hoặc mục tiêu của chính ván;
- cá cược hoặc dữ liệu thị trường cá cược;
- khuyến nghị tướng hoặc tối ưu lựa chọn tướng;
- phân tích sâu ý đồ hoặc chuỗi cấm/chọn;
- tự động lấy draft trực tiếp;
- mô hình chuyên biệt cho Fearless Draft;
- hướng dẫn chiến thuật mang tính khẳng định;
- chức năng quản trị cho khán giả đối với pipeline chuẩn bị dữ liệu, huấn luyện hoặc đánh giá mô hình.

Không được lách giới hạn bằng cách đổi tên một chức năng ngoài phạm vi thành “thống kê”, “giải thích” hoặc “tiện ích”.

## Công nghệ bắt buộc và giới hạn công nghệ

- Ngôn ngữ lập trình: Python.
- Xử lý dữ liệu: pandas.
- Học máy: scikit-learn.
- Cơ sở dữ liệu: PostgreSQL.
- Giao diện: Streamlit.
- Việc lưu pipeline hoặc mô hình có thể dùng joblib như đặc tả thiết kế.

PostgreSQL thay thế SQLite trong toàn bộ quá trình thi công. Khi chuyển kiểu dữ liệu hoặc ràng buộc từ thiết kế logic sang PostgreSQL, phải giữ nguyên ý nghĩa nghiệp vụ, tính toàn vẹn và khả năng truy vết; không được dựa vào hành vi riêng của SQLite.

Không tự ý thêm React, FastAPI, Django, Redis, Celery, Docker, MLflow hoặc công nghệ lớn khác khi chưa được yêu cầu.

Trước khi thêm bất kỳ thư viện phụ thuộc, dịch vụ ngoài, công nghệ, nguồn dữ liệu hoặc chức năng nào chưa có trong tài liệu, phải:

1. Nêu rõ nhu cầu cần giải quyết.
2. Giải thích vì sao các thành phần hiện có chưa đáp ứng.
3. Nêu ảnh hưởng đến kiến trúc, dữ liệu, triển khai và kiểm thử.
4. Yêu cầu người dùng xác nhận trước khi thay đổi.

Không được cài thư viện chỉ để thuận tiện nếu có thể thực hiện bằng các thành phần đã được chấp thuận.

## Kiến trúc và phân lớp

Phải tách riêng tối thiểu các lớp trách nhiệm sau:

- truy cập cơ sở dữ liệu;
- xử lý và kiểm tra dữ liệu;
- tạo đặc trưng;
- huấn luyện, lưu, nạp và suy luận mô hình;
- dịch vụ nghiệp vụ;
- giao diện Streamlit.

Quy tắc bắt buộc khi phân lớp:

- Trang Streamlit chỉ nhận thao tác người dùng, gọi dịch vụ nghiệp vụ và trình bày kết quả.
- Không đặt truy vấn SQL trực tiếp, tạo đặc trưng, lọc lịch sử, kiểm tra rò rỉ thời gian, suy luận mô hình hoặc quy tắc PRE/POST trực tiếp trong trang Streamlit.
- Lớp giao diện không được tự tính xác suất, chênh lệch, tỷ lệ lịch sử hoặc tự sửa dữ liệu thiếu.
- Lớp dịch vụ nghiệp vụ điều phối luồng chọn ván, PRE, đội hình, POST và so sánh; không được chứa chi tiết hiển thị phụ thuộc Streamlit.
- Lớp tạo đặc trưng phải nhận mốc thời gian rõ ràng và trả về cả giá trị, cờ thiếu, số mẫu và dấu vết phạm vi lịch sử cần thiết.
- Lớp mô hình chỉ nhận đầu vào đã được kiểm tra; không tự truy vấn dữ liệu tương lai hoặc sửa hồ sơ ván.
- Lớp truy cập dữ liệu phải bảo toàn ràng buộc PostgreSQL, giao dịch và tính bất biến của bản đánh giá.
- Pipeline chuẩn bị dữ liệu và pipeline chạy ứng dụng phải dùng cùng định nghĩa nghiệp vụ cho thời gian, định danh và đặc trưng.

Không bắt buộc một cấu trúc thư mục cụ thể khi tài liệu chưa quy định, nhưng ranh giới giữa các lớp phải rõ, có thể kiểm thử độc lập và không tạo phụ thuộc vòng.

## Cơ sở dữ liệu và tính toàn vẹn

Lược đồ cốt lõi phải bao quát các đối tượng nghiệp vụ trong thiết kế: giải đấu, giai đoạn, loạt trận, phiên bản, ván, đội, tuyển thủ, quan hệ thành viên theo thời gian, đội tham gia ván, tuyển thủ tham gia ván, tướng, bản đánh giá và cảnh báo.

Các ràng buộc bắt buộc:

- Một ván hợp lệ có đúng hai đội, một bên xanh và một bên đỏ.
- Mỗi đội trong ván có đúng một tuyển thủ cho từng vị trí TOP, JUNGLE, MID, BOT và SUPPORT khi đủ điều kiện tạo PRE.
- POST hợp lệ có đúng mười dòng tham gia đã gắn tướng, tuyển thủ và vị trí.
- Đội thắng chỉ được ghi sau trận và phải là một trong hai đội tham gia ván.
- Xác suất bên xanh phải nằm trong khoảng từ 0 đến 1.
- Bản PRE không tham chiếu PRE nền; bản POST bắt buộc tham chiếu đúng một PRE nền của cùng ván.
- Không được tạo POST khi PRE nền không còn hợp lệ hoặc nền thông tin không tương thích.
- Với cùng ván, cùng loại đánh giá và cùng phiên bản phương pháp, tại một thời điểm chỉ có tối đa một bản đang hiệu lực.
- Khi tạo lại đánh giá, phải giữ nguyên bản cũ và chỉ chuyển trạng thái hiệu lực; không cập nhật đè nội dung lịch sử.
- Cảnh báo gắn với một bản đánh giá đã tạo. Thông báo làm chặn việc tạo kết quả thuộc trạng thái xử lý của hồ sơ ván, không được giả lập bằng một bản đánh giá rỗng.

Mức thay đổi PRE–POST được tính từ hai bản đánh giá tương thích khi cần trình bày; không tạo bản ghi so sánh riêng nếu chưa có yêu cầu đã được xác nhận.

## Bất biến, truy vết và tái hiện đánh giá

- Mọi bản đánh giá đã tạo phải được giữ để truy vết và không được sửa lại sau khi biết kết quả thật.
- Mỗi đánh giá phải lưu hoặc tham chiếu bản chụp đầu vào bất biến đã dùng tại thời điểm tạo.
- Dấu vết tối thiểu gồm ván, mốc đánh giá, giải, giai đoạn, phiên bản, hai đội, bên thi đấu, lực lượng, phạm vi lịch sử, số mẫu, thống kê đầu vào, phiên bản dữ liệu, phiên bản mô hình và thời điểm tạo.
- Dấu vết POST phải có thêm đội hình cuối cùng, mười cặp tuyển thủ–tướng–vị trí và lịch sử tương ứng.
- POST phải lưu khóa tham chiếu đến đúng PRE nền, không chỉ lưu một chuỗi mô tả.
- Cùng dấu vết đầu vào và cùng phiên bản phương pháp phải cho phép tái hiện kết quả trong giới hạn số học được xác định trước.
- Không cập nhật lại bản chụp đầu vào cũ khi dữ liệu nguồn, ánh xạ định danh hoặc pipeline được sửa về sau.

## Thiết kế đặc trưng

Đặc trưng PRE phải được hình thành từ dữ liệu có trước mốc PRE và có thể gồm:

- phong độ gần đây;
- thành tích theo bên;
- lịch sử đối đầu;
- mức độ liên tục lực lượng.

Đặc trưng POST phải giữ nguyên toàn bộ đặc trưng PRE tương ứng và chỉ bổ sung:

- đội hình cuối cùng theo bên, tuyển thủ và vị trí;
- mã hóa tướng theo vị trí;
- số ván và số thắng của cặp tuyển thủ–tướng trong lịch sử chuyên nghiệp;
- tỷ lệ thắng đã xử lý đúng với mẫu nhỏ;
- thời gian từ lần sử dụng gần nhất;
- các giá trị tổng hợp cấp đội cùng số cặp thiếu hoặc có mẫu nhỏ.

Các nguyên tắc bắt buộc:

- Hai đội phải được biểu diễn đối xứng.
- Mọi tỷ lệ phải đi cùng số mẫu hoặc cờ thiếu.
- Không tính công thức tỷ lệ hoặc làm trơn khi số ván bằng 0.
- Giá trị thay thế dữ liệu thiếu phải được học chỉ từ train.
- Biến phân loại chưa xuất hiện ở validation hoặc test phải được xử lý bằng cơ chế không làm hỏng pipeline và không học lại từ các tập đó.
- Bối cảnh như giải, giai đoạn và phiên bản chỉ được biến thành đặc trưng mô hình khi tài liệu cho phép hoặc người dùng đã xác nhận; mặc định chúng phục vụ xác định phạm vi, lọc lịch sử, kiểm tra và trình bày.
- Không thêm đặc trưng mới ngoài tài liệu mà chưa hỏi lại.

## Huấn luyện, lựa chọn và đánh giá mô hình

Phạm vi bắt buộc gồm:

- baseline dùng tỷ lệ bên xanh thắng trong train;
- Logistic Regression;
- Random Forest là mô hình cây tổ hợp duy nhất trong phạm vi bắt buộc.

Không tự ý thêm họ mô hình boosting, mạng nơ-ron hoặc mô hình phức tạp khác.

Dữ liệu phải được chia theo thời gian ở cấp ván:

- 70% ván sớm nhất cho train;
- 15% tiếp theo cho validation;
- 15% cuối cùng cho test;
- các ván có cùng mốc thời gian phải nằm trong cùng một phần;
- hai mốc cắt phải được ghi vào cấu hình thực nghiệm và cố định trước khi mở test cuối.

Mọi lựa chọn đặc trưng, cửa sổ lịch sử, cách xử lý thiếu, bộ mã hóa, chuẩn hóa, tham số, ngưỡng cảnh báo, ngưỡng diễn giải và quyết định hiệu chỉnh xác suất chỉ được dựa trên train và validation. Test cuối chỉ dùng để báo cáo kết quả cuối cùng.

Các chỉ số chính thức gồm ROC-AUC, Brier Score, Log Loss và đường hiệu chỉnh xác suất. Vì giao diện hiển thị xác suất, phải ưu tiên Brier Score và Log Loss, sau đó mới xem ROC-AUC và độ ổn định.

Nếu cần hiệu chỉnh xác suất, dùng sigmoid hoặc Platt scaling theo quy trình fit trên train hoặc tách chéo nội bộ của train; validation chỉ quyết định có giữ hiệu chỉnh hay không; test không được dùng để fit hoặc lựa chọn.

Cấu hình A và B phải được đánh giá trên cùng tập ván đủ điều kiện, cùng nhãn, cùng phép chia và cùng nhóm chỉ số. Khi đo giá trị bổ sung sau cấm/chọn, không được để khác biệt thuật toán làm sai lệch phép so sánh.

Nếu Random Forest chỉ cải thiện rất nhỏ so với Logistic Regression, ưu tiên Logistic Regression để giảm độ phức tạp. Kết quả POST không cải thiện PRE vẫn phải được ghi nhận trung thực.

## Giao diện Streamlit

Giao diện phải theo đúng trình tự thời gian:

1. Chọn và xác nhận ván.
2. Xem kết quả PRE.
3. Xác nhận đội hình cuối cùng.
4. Xem POST và so sánh PRE–POST.

Quy tắc trình bày:

- Hai đội được trình bày đối xứng.
- Xác suất hệ thống phải tách biệt rõ với tỷ lệ lịch sử.
- Không hiển thị tên biến kỹ thuật, mã hóa one-hot hoặc tham số mô hình trên màn hình chính.
- Không hiển thị thông tin đội hình của chính ván tại màn hình PRE.
- Màn hình đội hình phải thể hiện đủ năm vị trí mỗi đội và phân biệt rõ trạng thái chưa ghi nhận lịch sử với giá trị 0.
- Chỉ cho phép tạo POST khi đủ mười cặp tuyển thủ–tướng–vị trí.
- Màn hình so sánh phải đặt PRE, POST và mức thay đổi trong cùng ngữ cảnh, đồng thời hiển thị thông tin mới sau cấm/chọn, số mẫu và cảnh báo.
- Xác suất PRE và POST hiển thị theo phần trăm với một chữ số thập phân và gắn nhãn “Xác suất ước lượng của hệ thống”.
- Mức thay đổi hiển thị bằng điểm phần trăm, có dấu dương hoặc âm khi phù hợp.
- Tỷ lệ lịch sử hiển thị cùng số thắng và số ván.
- Không dùng xác suất minh họa, số giả hoặc giá trị điền sẵn thay cho kết quả mô hình chưa có.
- Không dùng ngôn ngữ nhân quả trong tiêu đề, chú thích, thẻ kết quả, cảnh báo hoặc nội dung giải thích.

## Thông báo, cảnh báo và trạng thái

Phải phân biệt ít nhất ba nhóm cảnh báo:

- chất lượng hoặc xung đột dữ liệu;
- độ bao phủ lịch sử;
- giới hạn hoặc phạm vi áp dụng của phương pháp.

Các mã cốt lõi trong thiết kế phải được giữ ổn định về ý nghĩa:

- `E_PRE_INPUT_INCOMPLETE`: thiếu bối cảnh hoặc lực lượng bắt buộc, chặn PRE;
- `E_LINEUP_INCOMPLETE`: chưa đủ mười cặp tuyển thủ–tướng, chặn POST và so sánh;
- `E_SOURCE_CONFLICT`: thông tin bắt buộc xung đột, chặn kết quả tương ứng cho đến khi được giải quyết;
- `E_SCOPE_UNSUPPORTED`: ngoài phạm vi đã kiểm chứng, không tạo kết quả chính thức;
- `E_EVAL_INCOMPATIBLE`: PRE và POST không cùng nền thông tin, chặn phép so sánh;
- `W_TEAM_HISTORY_SMALL`: lịch sử cấp đội hạn chế, không chặn nếu vẫn đạt điều kiện tối thiểu;
- `W_H2H_MISSING`: chưa có dữ liệu đối đầu, không chặn PRE;
- `W_PAIR_HISTORY_MISSING`: một hoặc nhiều cặp chưa có lịch sử chuyên nghiệp, không chặn POST nếu phương pháp hỗ trợ;
- `I_OPGG_UNAVAILABLE`: không có OP.GG hợp lệ, chỉ là thông tin và không ảnh hưởng cấu hình cốt lõi.

Không tạo một bản đánh giá giả để mang lỗi. Khi điều kiện bắt buộc không đạt, phải trả trạng thái bị chặn cùng lý do rõ ràng.

## Kiểm thử bắt buộc

Các quy tắc nghiệp vụ quan trọng phải có khả năng kiểm thử tự động, đặc biệt:

- lịch sử được lọc theo thời điểm kết thúc trước mốc đánh giá;
- ván mục tiêu và các ván tương lai không tham gia đặc trưng;
- PRE không chứa dữ liệu draft hoặc diễn biến của ván mục tiêu;
- POST giữ nguyên nền PRE và chỉ bổ sung nhóm thông tin cho phép;
- dữ liệu thiếu, số 0 và không áp dụng được xử lý khác nhau;
- chưa có đối đầu không bị đổi thành 50%;
- chưa có lịch sử tuyển thủ–tướng không bị đổi thành 0% hoặc 50%;
- POST bị chặn khi thiếu bất kỳ cặp tuyển thủ–tướng–vị trí nào;
- không có OP.GG vẫn hoàn thành toàn bộ luồng cốt lõi;
- PRE và POST không tương thích không được tính chênh lệch;
- POST tham chiếu đúng PRE nền;
- bản đánh giá cũ không thay đổi sau khi có kết quả thật;
- tạo lại đánh giá giữ bản cũ để truy vết và chỉ thay đổi trạng thái hiệu lực;
- đội thắng chỉ được ghi sau trận và thuộc một trong hai đội tham gia;
- cùng mốc thời gian không bị tách sang các phần dữ liệu khác nhau;
- bộ tiền xử lý và hiệu chỉnh không được fit bằng validation hoặc test;
- giao diện không tự tính hoặc sửa các giá trị nghiệp vụ.

Ngoài kiểm thử đơn vị, phải có kiểm thử tích hợp cho luồng đầy đủ từ chọn ván đến PRE, xác nhận đủ đội hình, POST và so sánh. Phải có kiểm thử cho các luồng bị chặn do thiếu dữ liệu, xung đột nguồn và ngoài phạm vi.

Không được coi kiểm thử chỉ là bước cuối. Quy tắc thời gian, chống rò rỉ và tính bất biến phải được thiết kế để kiểm thử ngay từ khi tạo lớp dữ liệu và lớp dịch vụ.

## Kỷ luật thay đổi và thi công

Trước mỗi thay đổi, phải xác định:

1. Yêu cầu hoặc quy tắc nguồn nào làm căn cứ.
2. Lớp nào chịu trách nhiệm.
3. Ràng buộc dữ liệu và rủi ro rò rỉ thời gian liên quan.
4. Kiểm thử nào chứng minh thay đổi đúng.
5. Thay đổi có bổ sung phụ thuộc, công nghệ hoặc chức năng ngoài tài liệu hay không.

Khi thay đổi quy tắc nghiệp vụ, phải cập nhật đồng bộ lớp dịch vụ, ràng buộc dữ liệu, dấu vết đầu vào, cảnh báo và kiểm thử liên quan. Không được chỉ sửa giao diện để che một sai lệch ở dữ liệu hoặc mô hình.

Không được tự suy đoán số liệu thực nghiệm, tham số tối ưu, ngưỡng cảnh báo, chất lượng mô hình hoặc kết quả kiểm thử. Những giá trị này chỉ được ghi sau khi pipeline tương ứng đã chạy và có bằng chứng tái hiện được.

Nếu yêu cầu mới có thể làm thay đổi phạm vi cốt lõi, thứ tự luồng PRE/POST, định nghĩa đầu ra, nguồn dữ liệu, kiến trúc, công nghệ hoặc giao thức thực nghiệm, phải dừng phần bị ảnh hưởng và yêu cầu người dùng xác nhận trước khi thi công.
