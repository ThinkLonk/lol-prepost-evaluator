# Match Insight

Bộ khung Python cho hệ thống hỗ trợ khán giả đánh giá một ván Liên Minh Huyền Thoại chuyên nghiệp trước và sau cấm/chọn.

## Khởi tạo môi trường cục bộ

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

## Chạy kiểm tra

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pytest
```

## Khởi động Streamlit

```powershell
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

Ứng dụng hiện chỉ hiển thị thông báo khởi tạo. Chưa có cơ sở dữ liệu, pipeline dữ liệu, đặc trưng, mô hình hoặc nghiệp vụ PRE/POST.
