<div align="center">

# MATCH INSIGHT

### *Read the Rift before the game begins.*

**Phân tích xác suất thắng của một ván Liên Minh Huyền Thoại chuyên nghiệp
trước và sau giai đoạn cấm/chọn.**

<br>

[![React](https://img.shields.io/badge/React-19-0b1f2a?style=for-the-badge\&logo=react\&logoColor=61DAFB)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-0b1f2a?style=for-the-badge\&logo=typescript\&logoColor=3178C6)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-0b1f2a?style=for-the-badge\&logo=fastapi\&logoColor=009688)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-0b1f2a?style=for-the-badge\&logo=postgresql\&logoColor=4169E1)](https://www.postgresql.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-0b1f2a?style=for-the-badge\&logo=scikitlearn\&logoColor=F7931E)](https://scikit-learn.org/)

<br>

**React + TypeScript · FastAPI · PostgreSQL · Machine Learning**

<br>

[Khởi động](#quick-start) ·
[Luồng phân tích](#the-match-flow) ·
[Kiến trúc](#system-architecture) ·
[Cài đặt](#installation) ·
[Demo](#demo-flow) ·
[Xử lý lỗi](#troubleshooting)

</div>

---

> **PRE tells the story before the draft.
> POST tells you what changed after ten champions entered the Rift.**

Match Insight hỗ trợ khán giả phân tích một ván **League of Legends chuyên nghiệp** tại hai thời điểm:

* **PRE** — trước khi đội hình tướng cuối cùng được xác định.
* **POST** — sau khi cấm/chọn hoàn tất.

Hệ thống không chỉ trả về một con số dự đoán. Nó giữ nguyên bối cảnh PRE, bổ sung thông tin đội hình ở POST và cho phép quan sát mức thay đổi xác suất giữa hai trạng thái trên cùng một ván đấu.

Ba pipeline được hiển thị song song:

| Model                   | Vai trò                                                 |
| ----------------------- | ------------------------------------------------------- |
| **Baseline**            | Mốc tham chiếu từ tỷ lệ thắng BLUE trong tập huấn luyện |
| **Logistic Regression** | Mô hình canonical của hệ thống                          |
| **Random Forest**       | Pipeline bổ sung để đối chiếu hành vi dự báo            |

> [!NOTE]
> Chênh lệch PRE → POST được biểu diễn bằng **điểm phần trăm**.
> Đây là thay đổi trong dự báo của mô hình, không phải bằng chứng cho quan hệ nhân quả giữa draft và kết quả trận đấu.

---

# The Match Flow

```text
                 ┌──────────────────────────────┐
                 │       MATCH CONTEXT          │
                 │ teams · side · patch · roster│
                 └──────────────┬───────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │       PRE        │
                      │ win probability  │
                      └────────┬─────────┘
                               │
                               │ Final draft
                               ▼
             ┌──────────────────────────────────┐
             │        10 FINAL CHAMPIONS        │
             │ player × champion history       │
             └────────────────┬─────────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │       POST       │
                      │ win probability  │
                      └────────┬─────────┘
                               │
                               ▼
                    PRE  ───────►  POST
                        Δ probability
```

### PRE

PRE sử dụng thông tin có trước khi đội hình tướng cuối cùng được xác định:

* hai đội thi đấu;
* BLUE / RED side;
* patch;
* roster;
* phong độ lịch sử;
* thành tích theo side;
* đối đầu;
* mức độ liên tục của đội hình.

### POST

POST giữ nguyên toàn bộ bối cảnh PRE và bổ sung:

* mười tướng cuối cùng;
* lịch sử tuyển thủ–tướng;
* số mẫu lịch sử liên quan;
* cảnh báo khi chưa ghi nhận cặp tuyển thủ–tướng.

### Comparison

Mỗi mô hình hiển thị:

```text
PRE probability
POST probability
Δ percentage points
```

Có thể chuyển đội đang xem để quan sát xác suất theo phía còn lại.

---

# Stack

<table>
<tr>
<td width="33%" valign="top">

### Frontend

**React**
**TypeScript**

Giao diện nhập dữ liệu, tìm kiếm đội / tuyển thủ / tướng và bảng so sánh PRE–POST.

</td>

<td width="33%" valign="top">

### Backend

**FastAPI**
**Python**

Quản lý phiên phân tích, điều phối PRE / POST, phục vụ assets và API.

</td>

<td width="33%" valign="top">

### Data & ML

**PostgreSQL**
**pandas**
**scikit-learn**

Lịch sử thi đấu, feature engineering, model inference và lưu evaluation.

</td>
</tr>
</table>

Nguồn dữ liệu lịch sử:

* **Oracle's Elixir** — dữ liệu thi đấu chuyên nghiệp.
* **Riot Data Dragon** — metadata và hình ảnh tướng.

Bản **Streamlit + Plotly** trước đây vẫn được giữ lại để đối chiếu.

---

# Quick Start

> Chạy các lệnh tại **thư mục gốc của project**, nơi chứa `README.md`.

```powershell
cd "C:\Đồ án ngành"

.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-web.txt

.\scripts\start_web.ps1
```

Nếu dự án nằm ở nơi khác, thay đường dẫn của lệnh `cd`.

Sau khi server khởi động:

```text
Application   http://127.0.0.1:8000
API Docs      http://127.0.0.1:8000/docs
```

Script sẽ:

```text
install frontend dependencies when required
        ↓
build React
        ↓
start FastAPI
        ↓
serve the application
```

> [!IMPORTANT]
> Cơ sở dữ liệu PostgreSQL và các model artifact phải được cấu hình trước khi tạo PRE hoặc POST.

Xem thêm:

**[`docs/React_FastAPI.md`](docs/React_FastAPI.md)**

---

# Running an Existing Build

Nếu `.venv`, `.env`, PostgreSQL, model và React build đã sẵn sàng:

```powershell
.\scripts\start_web.ps1 -SkipBuild
```

Giữ terminal mở trong quá trình sử dụng.

Dừng server bằng:

```text
Ctrl + C
```

Sau khi chỉnh sửa frontend:

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\scripts\start_web.ps1 -SkipBuild
```

Nếu server vẫn đang chạy và chỉ frontend thay đổi:

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
```

Sau đó dùng:

```text
Ctrl + F5
```

Nếu Python backend thay đổi, cần restart server.

---

# Legacy Interface

Bản Streamlit cũ vẫn có thể chạy riêng:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m streamlit run streamlit_app.py
```

Mặc định:

```text
http://localhost:8501
```

---

# System Architecture

```mermaid
flowchart LR
    User["User"]

    subgraph Client["CLIENT"]
        React["React + TypeScript"]
    end

    subgraph Backend["APPLICATION"]
        API["FastAPI"]
        Session["Analysis Session"]
        PRE["PRE Service"]
        POST["POST Service"]
        Compare["Comparison Service"]
    end

    subgraph Intelligence["MODEL LAYER"]
        Feature["Feature Engineering"]
        LR["Logistic Regression"]
        RF["Random Forest"]
        Base["Baseline"]
    end

    subgraph Storage["DATA"]
        DB[("PostgreSQL")]
        Models["Model Artifacts"]
        Assets["Teams / Players / Champions"]
    end

    User --> React
    React --> API

    API --> Session
    Session --> PRE
    Session --> POST
    Session --> Compare

    PRE --> Feature
    POST --> Feature

    Feature --> LR
    Feature --> RF
    Feature --> Base

    LR --> Models
    RF --> Models
    Base --> Models

    PRE --> DB
    POST --> DB
    Compare --> DB

    Assets --> React
```

---

# Installation

## 1 — Runtime Requirements

Máy chạy hệ thống cần:

```text
Python      3.12.14
Node.js     24.x tested
npm
PostgreSQL
```

Nếu phục hồi database bằng CLI:

```text
psql
pg_restore
```

cần có trong `PATH`.

Có thể dùng **pgAdmin** thay thế cho thao tác phục hồi bằng command line.

---

## 2 — Model Environment

Model loader kiểm tra chính xác môi trường đã dùng cho model bundle.

| Component    |   Version |
| ------------ | --------: |
| Python       | `3.12.14` |
| NumPy        |   `2.5.2` |
| pandas       |   `3.0.5` |
| scikit-learn |   `1.9.0` |
| SciPy        |  `1.18.1` |
| joblib       |   `1.5.3` |

Nguồn metadata:

```text
artifacts/models/retrospective_pre_post.json
```

Kiểm tra Python:

```powershell
python --version
```

Kết quả cần là:

```text
Python 3.12.14
```

> [!WARNING]
> Không chỉnh metadata hoặc thay version chỉ để vượt qua compatibility check.

---

## 3 — Virtual Environment

Trên máy mới:

```powershell
python -m venv .venv
```

Không nên sao chép `.venv` từ máy khác.

Kiểm tra:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip --version
```

Python cần trả về:

```text
3.12.14
```

Cài dependency:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m pip check
```

Kết quả mong đợi:

```text
No broken requirements found.
```

### Dependency files

```text
requirements.txt
    core runtime + ML dependencies

requirements-web.txt
    FastAPI + Uvicorn + requirements.txt

requirements-dev.txt
    development and testing
```

`requirements.txt` đã cố định các phiên bản ML tương thích với model bundle.

`psycopg[binary]` được sử dụng để kết nối PostgreSQL trên Windows.

Không cần:

```powershell
Activate.ps1
```

vì các lệnh sử dụng trực tiếp Python trong `.venv`.

---

# Database Configuration

Tạo `.env` từ file mẫu:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

Ví dụ:

```dotenv
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/lol_prepost
```

Thay:

```text
user
password
host
port
database
```

bằng thông tin thực tế.

Nếu mật khẩu chứa các ký tự như:

```text
@
#
%
```

cần URL-encode trước.

Không đặt dấu nháy quanh `DATABASE_URL`.

Nếu PowerShell đã tồn tại:

```text
DATABASE_URL
```

thì biến môi trường đó được ưu tiên hơn `.env`.

Các biến:

```text
WIKI_USERNAME_MATCH_INSIGHT
WIKI_PASSWORD_MATCH_INSIGHT
```

chỉ dùng cho script lấy dữ liệu Leaguepedia và có thể để trống khi chạy giao diện bằng dataset đã chuẩn bị.

> [!CAUTION]
> Không commit `.env` chứa mật khẩu lên repository.

---

# Required Data

Chỉ source code và database rỗng **không đủ** để chạy phân tích.

Hệ thống cần đồng thời database lịch sử và model bundle tương ứng.

| Component              | Path / Requirement                                                |
| ---------------------- | ----------------------------------------------------------------- |
| PostgreSQL history     | Lịch sử game, team, player, champion, temporal data và evaluation |
| Canonical model        | `artifacts/models/retrospective_pre_post.joblib`                  |
| Model metadata         | `artifacts/models/retrospective_pre_post.json`                    |
| Comparison manifest    | `artifacts/models_comparison/manifest.json`                       |
| Baseline pipeline      | `artifacts/models_comparison/baseline_pre_post.joblib`            |
| Random Forest pipeline | `artifacts/models_comparison/random_forest_pre_post.joblib`       |
| Temporal reference     | `data/reference/retrospective_pre_inputs.json`                    |
| Champion assets        | `assets/champions/`                                               |
| Team assets            | `assets/teams/`                                                   |
| Player assets          | `assets/players/`                                                 |

Thiếu ảnh không ngăn hệ thống tạo đánh giá.

Một số dữ liệu và `.joblib` không được lưu trên Git do `.gitignore`.

---

# Restoring PostgreSQL

Với backup dạng:

```text
pg_dump -Fc
```

có thể tạo database mới:

```powershell
psql -h localhost -U postgres -c "CREATE DATABASE lol_prepost;"
```

Sau đó restore:

```powershell
pg_restore `
  -h localhost `
  -U postgres `
  -d lol_prepost `
  --no-owner `
  --no-privileges `
  --exit-on-error `
  "C:\duong-dan\lol_prepost.backup"
```

Với `.sql`, sử dụng `psql` hoặc pgAdmin.

Kiểm tra schema:

```powershell
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic heads
```

Revision hiện tại:

```text
c6e31a9d4b72
```

Database bàn giao cần tương ứng revision này.

Model loader còn kiểm tra hash của:

```text
historical data
temporal reference
model artifact
```

Các thành phần này cần được cập nhật đồng bộ.

---

# Database Check

Kiểm tra kết nối mà không ghi dữ liệu:

```powershell
.\.venv\Scripts\python.exe -c "from match_insight.database.engine import check_database_connection; print(check_database_connection())"
```

Kết quả mong đợi:

```text
1
```

Sau đó:

```powershell
.\scripts\start_web.ps1
```

---

# Demo Flow

## Phase I — Match Setup

Chọn:

```text
BLUE team
RED team
Patch
5 BLUE players
5 RED players
```

Có thể tìm kiếm bằng tên rút gọn.

Ví dụ:

```text
HLE       → Hanwha Life Esports
Gen       → Gen.G
geng      → Gen.G
```

Tìm kiếm:

```text
case-insensitive
punctuation-insensitive
whitespace-insensitive
```

Roster cần đủ năm vị trí:

| Position | Role       |
| -------- | ---------- |
| TOP      | Đường trên |
| JUNGLE   | Đi rừng    |
| MID      | Đường giữa |
| BOT      | Xạ thủ     |
| SUPPORT  | Hỗ trợ     |

Tổng cộng phải có **10 tuyển thủ khác nhau**.

Ô:

```text
Giải / bối cảnh ván
```

là tùy chọn.

Sau khi xác nhận:

```text
teams
sides
patch
players
positions
```

có thể tạo PRE.

---

## Phase II — PRE

Nhấn:

```text
Tạo đánh giá PRE
```

Hệ thống:

```text
collect historical context
        ↓
filter eligible history
        ↓
build PRE features
        ↓
run three models
        ↓
persist evaluation
        ↓
return probabilities + warnings
```

Mốc lịch sử của phiên tương tác được xác định tại thời điểm tạo PRE:

```text
00:00 UTC
of the day immediately before
the PRE creation date in UTC
```

POST tiếp tục sử dụng cùng cutoff.

Đây là quy tắc lọc lịch sử của hệ thống, không phải thời điểm draft thật được lấy trực tiếp từ giải đấu.

---

## Phase III — Final Draft

Sau khi draft hoàn tất, chọn đủ:

```text
5 BLUE champions
5 RED champions
```

Không được trùng tướng.

Nhấn:

```text
Tạo POST & so sánh
```

POST bổ sung lịch sử:

```text
player × champion
```

Nếu một cặp chưa từng xuất hiện trong lịch sử:

```text
Chưa ghi nhận
```

Hệ thống không tự quy đổi thành:

```text
0%
50%
```

Cảnh báo sẽ chỉ rõ:

```text
player
champion
team
position
```

---

## Phase IV — PRE vs POST

Kết quả có dạng:

| Model               |         PRE |        POST |  Δ |
| ------------------- | ----------: | ----------: | -: |
| Baseline            | probability | probability | pp |
| Logistic Regression | probability | probability | pp |
| Random Forest       | probability | probability | pp |

Ví dụ:

```text
54% → 58%
```

được đọc là:

```text
+4 percentage points
```

không phải:

```text
+4%
```

Baseline học tỷ lệ thắng BLUE từ train set nên PRE và POST có thể bằng nhau.

Một xác suất lớn hơn ở một ván **không có nghĩa mô hình tốt hơn**.

Đánh giá chất lượng model cần dựa vào các metric như:

```text
Brier Score
Log Loss
ROC-AUC
Calibration
```

trên cùng evaluation set.

---

# Editing an Analysis

| Changed input    | Required action                                 |
| ---------------- | ----------------------------------------------- |
| Chỉ đổi champion | **Chỉnh sửa đội hình** → giữ PRE → tạo POST mới |
| Team             | Tạo PRE mới                                     |
| Side             | Tạo PRE mới                                     |
| Player           | Tạo PRE mới                                     |
| Position         | Tạo PRE mới                                     |
| Patch            | Tạo PRE mới                                     |
| Context          | Tạo PRE mới                                     |

Nếu lưu kết quả thất bại:

```text
fix database connection / write permission
        ↓
retry operation
```

Kết quả chỉ được công bố sau khi persistence thành công.

---

# Saved Evaluations

Mở:

```text
Bản đã lưu
```

Nhập:

```text
Mã đánh giá
```

sau đó:

```text
Tra cứu bản lưu
```

Màn hình cho phép xem lại:

```text
input
prediction
warnings
persistence status
```

Việc tra cứu không phục hồi phiên phân tích để tiếp tục tạo POST.

Model comparison provenance được lưu trong:

```text
input_snapshot.provenance.model_comparison
```

cùng transaction của evaluation canonical.

Các bản ghi cũ chỉ chứa một model vẫn được đọc nguyên trạng và không bị tính lại.

---

# Project Structure

```text
Match Insight
│
├── frontend/
│   └── src/
│       React + TypeScript application
│
├── match_insight/
│   │
│   ├── api/
│   │   FastAPI, analysis sessions, assets
│   │
│   ├── services/
│   │   PRE, POST, comparison, persistence
│   │
│   ├── features/
│   │   historical aggregation and filtering
│   │
│   ├── ml/
│   │   datasets, training, model loading, inference
│   │
│   ├── database/
│   │   schema models, queries, persistence
│   │
│   └── data_processing/
│       source validation and normalization
│
├── artifacts/
│   trained model artifacts
│
├── assets/
│   champions · teams · players
│
├── data/
│   input and reference data
│
├── reports/
│   data-quality and experiment reports
│
├── alembic/
│   PostgreSQL schema migrations
│
├── scripts/
│   ETL, training and maintenance scripts
│
├── docs/
│   project documentation
│
├── streamlit_app.py
│   legacy Streamlit entrypoint
│
└── README.md
```

Một số script như:

```text
apply_oracle_etl.py
apply_oracle_riot_v5_backfill.py
train_real_models.py
```

thuộc pipeline chuẩn bị dữ liệu / experiment và **không phải** bước cần chạy để mở web application.

---

# Troubleshooting

<details>
<summary><b>No module named pip</b></summary>

<br>

Chạy:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
```

Sau đó:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-web.txt
```

</details>

<details>
<summary><b>No module named ensurepip</b></summary>

<br>

Python đang sử dụng thiếu thành phần cần thiết.

Cài bản Python đầy đủ đúng version:

```text
Python 3.12.14
```

sau đó tạo lại `.venv`.

</details>

<details>
<summary><b>No matching distribution found</b></summary>

<br>

Kiểm tra:

```text
Python version
package index
platform compatibility
locked dependencies
```

Không tự thay các version model dependency chỉ để hoàn thành cài đặt.

</details>

<details>
<summary><b>Không tìm thấy DATABASE_URL</b></summary>

<br>

Kiểm tra:

```text
.env
PostgreSQL username
password
host
port
database
```

Đồng thời kiểm tra PowerShell có biến `DATABASE_URL` cũ đang override `.env` hay không.

</details>

<details>
<summary><b>Model không load được</b></summary>

<br>

Kiểm tra:

```text
artifacts/models/retrospective_pre_post.joblib
artifacts/models/retrospective_pre_post.json

artifacts/models_comparison/manifest.json
artifacts/models_comparison/baseline_pre_post.joblib
artifacts/models_comparison/random_forest_pre_post.joblib

data/reference/retrospective_pre_inputs.json
```

Không thay model bằng prediction giả hoặc bỏ qua hash validation.

</details>

<details>
<summary><b>Model environment mismatch</b></summary>

<br>

Đối chiếu đủ:

```text
Python
NumPy
pandas
scikit-learn
SciPy
joblib
```

bao gồm patch version của Python.

</details>

<details>
<summary><b>Nút PRE bị khóa</b></summary>

<br>

Kiểm tra:

```text
2 teams khác nhau
10 players khác nhau
đúng positions
patch
input confirmation
```

</details>

<details>
<summary><b>Nút POST bị khóa</b></summary>

<br>

Cần:

```text
PRE đã tạo thành công
10 champions hợp lệ
không trùng champion
```

</details>

<details>
<summary><b>Không hiển thị ảnh</b></summary>

<br>

Kiểm tra:

```text
assets/champions/
assets/teams/
assets/players/
```

Thiếu ảnh không ngăn inference.

</details>

<details>
<summary><b>Không tìm thấy start_web.ps1</b></summary>

<br>

Đảm bảo terminal đang đứng ở project root:

```powershell
cd "C:\Đồ án ngành"
```

Việc terminal hiển thị:

```text
(.venv)
```

không có nghĩa bạn đang đứng đúng thư mục.

</details>

<details>
<summary><b>Unexpected token hoặc lỗi encoding Windows PowerShell</b></summary>

<br>

Sử dụng bản script hiện tại.

Thông báo nội bộ trong script đã được chuyển sang ASCII để tương thích Windows PowerShell 5.1.

</details>

<details>
<summary><b>Frontend chưa cập nhật</b></summary>

<br>

Build lại:

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
```

sau đó:

```text
Ctrl + F5
```

`-SkipBuild` chỉ dùng build đã tồn tại.

</details>

<details>
<summary><b>Port 8000 already in use</b></summary>

<br>

Nếu application đã chạy, mở instance hiện có.

Hoặc:

```powershell
.\scripts\start_web.ps1 -SkipBuild -Port 8001
```

sau đó truy cập:

```text
http://127.0.0.1:8001
```

</details>

---

# Development & Verification

Cài development dependencies:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-dev.txt
```

Chạy test:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest
```

Build frontend:

```powershell
npm.cmd --prefix frontend run build
```

Biên bản kiểm chứng React + FastAPI:

[`reports/authored/react_fastapi_acceptance.md`](reports/authored/react_fastapi_acceptance.md)

---

# Scope

Match Insight tập trung vào **pre-game analysis**.

Hệ thống hiện không:

```text
fetch live draft automatically
update probability during the match
recommend champion picks
optimize draft strategy
provide betting recommendations
```

Người dùng chủ động cung cấp:

```text
match context
roster
patch
final champions
```

Hệ thống sau đó sử dụng dữ liệu lịch sử đã được chuẩn hóa để tạo PRE và POST.

---

<div align="center">

<br>

### MATCH INSIGHT

*Two states of the same match.*

**Before the draft. After the draft.
See what changed.**

<br>

`PRE` ───────── `DRAFT` ───────── `POST`

<br>

Built around professional League of Legends match data.

</div>
