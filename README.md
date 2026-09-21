<div align="center">

<br>

# MATCH INSIGHT

### `PRE` ───────── `DRAFT` ───────── `POST`

**Read the Rift before the game begins.**

Hệ thống phân tích xác suất thắng của một ván **League of Legends chuyên nghiệp**
trước và sau giai đoạn cấm/chọn.

<br>

![React](https://img.shields.io/badge/REACT-0B1F2A?style=for-the-badge\&logo=react\&logoColor=0AC8B9)
![TypeScript](https://img.shields.io/badge/TYPESCRIPT-0B1F2A?style=for-the-badge\&logo=typescript\&logoColor=C89B3C)
![FastAPI](https://img.shields.io/badge/FASTAPI-0B1F2A?style=for-the-badge\&logo=fastapi\&logoColor=0AC8B9)
![PostgreSQL](https://img.shields.io/badge/POSTGRESQL-0B1F2A?style=for-the-badge\&logo=postgresql\&logoColor=C89B3C)
![scikit-learn](https://img.shields.io/badge/SCIKIT--LEARN-0B1F2A?style=for-the-badge\&logo=scikitlearn\&logoColor=0AC8B9)

<br>

[OVERVIEW](#overview)
  ·  
[MATCH FLOW](#match-flow)
  ·  
[ARCHITECTURE](#architecture)
  ·  
[QUICK START](#quick-start)
  ·  
[SETUP](#full-setup)

<br>

<sub>React + TypeScript · FastAPI · PostgreSQL · Machine Learning</sub>

<br><br>

</div>

---

<div align="center">

`BLUE SIDE`        ◇        `RED SIDE`

</div>

# Overview

**Match Insight** hỗ trợ khán giả phân tích một ván Liên Minh Huyền Thoại chuyên nghiệp tại hai thời điểm khác nhau.

<table>
<tr>
<td width="50%" valign="top">

### PRE / BEFORE THE DRAFT

Ước lượng xác suất thắng từ bối cảnh trước khi đội hình tướng cuối cùng được xác định.

Hệ thống sử dụng:

* đội thi đấu;
* BLUE / RED side;
* patch;
* roster;
* phong độ lịch sử;
* thành tích theo bên;
* đối đầu;
* mức độ liên tục đội hình.

</td>
<td width="50%" valign="top">

### POST / AFTER THE DRAFT

Giữ nguyên toàn bộ bối cảnh của PRE và bổ sung đội hình cuối cùng.

Hệ thống bổ sung:

* 10 tướng được chọn;
* lịch sử tuyển thủ–tướng;
* số mẫu liên quan;
* các cảnh báo thiếu lịch sử.

</td>
</tr>
</table>

Sau POST, hệ thống so sánh trực tiếp:

```text
                 PRE                    POST

            win probability        win probability
                   │                     │
                   └───────── Δ ─────────┘

                    percentage points
```

Ba pipeline được hiển thị song song:

| MODEL                   | ROLE                                                  |
| ----------------------- | ----------------------------------------------------- |
| **Baseline**            | Mốc tham chiếu từ tỷ lệ thắng BLUE của tập huấn luyện |
| **Logistic Regression** | Mô hình canonical của hệ thống                        |
| **Random Forest**       | Pipeline bổ sung để đối chiếu hành vi dự báo          |

> [!NOTE]
> Chênh lệch PRE → POST thể hiện **sự thay đổi dự báo của mô hình theo điểm phần trăm**, không chứng minh tác động nhân quả của đội hình cấm/chọn.

---

<div align="center">

### ◇    THE MATCH, IN TWO STATES    ◇

</div>

# Match Flow

```text
                              MATCH CONTEXT
                teams · side · patch · roster · history

                                    │
                                    │
                                    ▼

                         ╔════════════════════╗
                         ║        PRE         ║
                         ║                    ║
                         ║  Before the draft  ║
                         ╚═════════╤══════════╝
                                   │
                                   │
                              FINAL DRAFT
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    │       10 CHAMPIONS          │
                    │                             │
                    │  player × champion history │
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                         ╔════════════════════╗
                         ║        POST        ║
                         ║                    ║
                         ║   After the draft  ║
                         ╚═════════╤══════════╝
                                   │
                                   ▼

                         PRE  ───────►  POST
                               Δ
                       percentage points
```

### Phase 01 / Match Setup

Người dùng thiết lập:

```text
BLUE team
RED team

TOP
JUNGLE
MID
BOT
SUPPORT

Patch
Optional match context
```

Mỗi bên cần đủ năm tuyển thủ và hai đội phải khác nhau.

---

### Phase 02 / PRE

PRE sử dụng dữ liệu lịch sử hợp lệ trước cutoff để xây dựng đặc trưng.

```text
Historical Match Data
        │
        ├── Team Form
        ├── Side Performance
        ├── Head-to-Head
        └── Roster Continuity
        │
        ▼
   PRE FEATURES
        │
        ▼
   ML PIPELINES
        │
        ▼
 WIN PROBABILITY
```

---

### Phase 03 / Draft

Sau khi cấm/chọn hoàn tất, người dùng nhập đội hình cuối cùng gồm mười tướng khác nhau.

```text
BLUE

TOP       Champion
JUNGLE    Champion
MID       Champion
BOT       Champion
SUPPORT   Champion


RED

TOP       Champion
JUNGLE    Champion
MID       Champion
BOT       Champion
SUPPORT   Champion
```

---

### Phase 04 / POST

POST giữ nguyên context của PRE và bổ sung:

```text
Player × Champion history
Champion lineup
Historical sample count
Missing-history warnings
```

Nếu một cặp tuyển thủ–tướng chưa từng xuất hiện trong lịch sử, hệ thống hiển thị:

```text
Chưa ghi nhận
```

thay vì tự xem dữ liệu đó là `0%` hoặc `50%`.

---

### Phase 05 / Compare

```text
┌──────────────────────┬──────────┬──────────┬──────────┐
│ MODEL                │   PRE    │   POST   │    Δ     │
├──────────────────────┼──────────┼──────────┼──────────┤
│ Baseline             │    —     │    —     │    —     │
│ Logistic Regression  │    —     │    —     │    —     │
│ Random Forest        │    —     │    —     │    —     │
└──────────────────────┴──────────┴──────────┴──────────┘
```

Ví dụ:

```text
54%  →  58%

Δ = +4 percentage points
```

không phải:

```text
+4%
```

Có thể chuyển đội đang xem để quan sát xác suất theo phía đối diện.

---

<div align="center">

`DATA` ───────── ◇ ───────── `MODEL` ───────── ◇ ───────── `INTERFACE`

</div>

# Architecture

```mermaid
flowchart LR
    USER["USER"]

    subgraph CLIENT["CLIENT"]
        REACT["React + TypeScript"]
    end

    subgraph APPLICATION["APPLICATION"]
        API["FastAPI"]
        SESSION["Analysis Session"]
        PRE["PRE Service"]
        POST["POST Service"]
        COMPARE["Comparison"]
    end

    subgraph MODEL["MODEL LAYER"]
        FEATURES["Feature Engineering"]
        BASE["Baseline"]
        LR["Logistic Regression"]
        RF["Random Forest"]
    end

    subgraph DATA["DATA LAYER"]
        DB[("PostgreSQL")]
        ARTIFACTS["Model Artifacts"]
        ASSETS["Teams · Players · Champions"]
    end

    USER --> REACT
    REACT --> API

    API --> SESSION

    SESSION --> PRE
    SESSION --> POST
    SESSION --> COMPARE

    PRE --> FEATURES
    POST --> FEATURES

    FEATURES --> BASE
    FEATURES --> LR
    FEATURES --> RF

    BASE --> ARTIFACTS
    LR --> ARTIFACTS
    RF --> ARTIFACTS

    PRE --> DB
    POST --> DB
    COMPARE --> DB

    ASSETS --> REACT
```

---

# Stack

<table>
<tr>
<td width="33%" valign="top">

### CLIENT

**React**
**TypeScript**

Giao diện thiết lập trận đấu, roster, champion lineup và bảng so sánh.

</td>

<td width="33%" valign="top">

### APPLICATION

**FastAPI**
**Python**

Điều phối phiên phân tích, PRE, POST, comparison và persistence.

</td>

<td width="33%" valign="top">

### DATA / ML

**PostgreSQL**
**pandas**
**scikit-learn**

Dữ liệu lịch sử, feature engineering, inference và lưu evaluation.

</td>
</tr>
</table>

### Data Sources

**Oracle's Elixir**

Dữ liệu lịch sử thi đấu League of Legends chuyên nghiệp.

**Riot Data Dragon**

Danh mục và hình ảnh tướng.

---

# Quick Start

Chạy từ thư mục gốc chứa `README.md`:

```powershell
cd "C:\Đồ án ngành"

.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-web.txt

.\scripts\start_web.ps1
```

Nếu project nằm ở nơi khác, thay đường dẫn của lệnh `cd`.

Sau khi server khởi động:

| SERVICE       | ADDRESS                      |
| ------------- | ---------------------------- |
| Match Insight | `http://127.0.0.1:8000`      |
| FastAPI Docs  | `http://127.0.0.1:8000/docs` |

Startup script sẽ thực hiện:

```text
Frontend dependencies
        │
        ▼
    React build
        │
        ▼
 FastAPI server
        │
        ▼
  Match Insight
```

> [!IMPORTANT]
> PostgreSQL và model artifacts cần được cấu hình đúng trước khi tạo PRE hoặc POST.

Tài liệu chi tiết:

[`docs/React_FastAPI.md`](docs/React_FastAPI.md)

---

# Existing Build

Nếu `.venv`, `.env`, PostgreSQL, model và frontend build đã có sẵn:

```powershell
.\scripts\start_web.ps1 -SkipBuild
```

Giữ terminal mở trong lúc sử dụng.

Dừng server:

```text
Ctrl + C
```

Sau khi cập nhật frontend:

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
```

Nếu server vẫn đang chạy:

```text
Ctrl + F5
```

để tải lại build mới.

Nếu backend Python thay đổi, cần khởi động lại server.

---

# Legacy Interface

Bản Streamlit trước đây vẫn được giữ để đối chiếu:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m streamlit run streamlit_app.py
```

Địa chỉ mặc định:

```text
http://localhost:8501
```

---

<div align="center">

### FULL SYSTEM SETUP

<sub>Environment · Database · Models · Runtime</sub>

</div>

# Full Setup

<details>
<summary><b>01 / Runtime Requirements</b></summary>

<br>

Môi trường đã kiểm thử:

| COMPONENT  |   VERSION |
| ---------- | --------: |
| Python     | `3.12.14` |
| Node.js    |    `24.x` |
| PostgreSQL |  Required |
| npm        |  Required |

Nếu phục hồi database bằng command line, cần:

```text
psql
pg_restore
```

trong `PATH`.

Có thể sử dụng pgAdmin thay thế.

</details>

---

<details>
<summary><b>02 / Model Environment</b></summary>

<br>

Model loader kiểm tra chính xác các phiên bản lấy từ:

```text
artifacts/models/retrospective_pre_post.json
```

| COMPONENT    |   VERSION |
| ------------ | --------: |
| Python       | `3.12.14` |
| NumPy        |   `2.5.2` |
| pandas       |   `3.0.5` |
| scikit-learn |   `1.9.0` |
| SciPy        |  `1.18.1` |
| joblib       |   `1.5.3` |

Kiểm tra Python:

```powershell
python --version
```

Kết quả yêu cầu:

```text
Python 3.12.14
```

Không thay version trong metadata để bỏ qua compatibility check.

</details>

---

<details>
<summary><b>03 / Virtual Environment</b></summary>

<br>

Trên máy mới:

```powershell
python -m venv .venv
```

Không sử dụng `.venv` được sao chép từ máy khác.

Kiểm tra:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip --version
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

Các file dependency:

```text
requirements.txt
    Core runtime + ML dependencies

requirements-web.txt
    FastAPI + Uvicorn + requirements.txt

requirements-dev.txt
    Development + testing
```

Các lệnh sử dụng trực tiếp Python trong `.venv`, do đó không cần chạy:

```text
Activate.ps1
```

</details>

---

<details>
<summary><b>04 / PostgreSQL Connection</b></summary>

<br>

Tạo `.env`:

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

Nếu mật khẩu có các ký tự như:

```text
@
#
%
```

cần URL encode.

Không đặt dấu nháy quanh giá trị trong `.env`.

Nếu PowerShell đang có biến:

```text
DATABASE_URL
```

thì biến đó được ưu tiên hơn `.env`.

Các biến:

```text
WIKI_USERNAME_MATCH_INSIGHT
WIKI_PASSWORD_MATCH_INSIGHT
```

chỉ dành cho script lấy dữ liệu Leaguepedia và có thể để trống khi chạy hệ thống với dữ liệu đã chuẩn bị.

Không commit `.env` chứa mật khẩu lên repository.

</details>

---

<details>
<summary><b>05 / Required Data & Models</b></summary>

<br>

Source code cùng một database rỗng **không đủ** để tạo PRE / POST.

Cần:

| COMPONENT           | PATH / REQUIREMENT                                                |
| ------------------- | ----------------------------------------------------------------- |
| PostgreSQL data     | Lịch sử game, team, player, champion, temporal data và evaluation |
| Canonical model     | `artifacts/models/retrospective_pre_post.joblib`                  |
| Model metadata      | `artifacts/models/retrospective_pre_post.json`                    |
| Comparison manifest | `artifacts/models_comparison/manifest.json`                       |
| Baseline model      | `artifacts/models_comparison/baseline_pre_post.joblib`            |
| Random Forest model | `artifacts/models_comparison/random_forest_pre_post.joblib`       |
| Temporal reference  | `data/reference/retrospective_pre_inputs.json`                    |
| Champion assets     | `assets/champions/`                                               |
| Team assets         | `assets/teams/`                                                   |
| Player assets       | `assets/players/`                                                 |

Thiếu ảnh không ngăn hệ thống phân tích.

Một số `.joblib`, raw data và resources được loại khỏi Git bằng `.gitignore`.

Model loader kiểm tra hash của:

```text
historical data
temporal reference
model artifacts
```

Các thành phần cần thuộc cùng một validated bundle.

</details>

---

<details>
<summary><b>06 / Restore PostgreSQL</b></summary>

<br>

Với backup dạng custom:

```text
pg_dump -Fc
```

tạo database:

```powershell
psql -h localhost -U postgres -c "CREATE DATABASE lol_prepost;"
```

Restore:

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

Với backup `.sql`, dùng `psql` hoặc pgAdmin.

Kiểm tra Alembic:

```powershell
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic heads
```

Revision hiện tại:

```text
c6e31a9d4b72
```

Database cần tương ứng revision này.

</details>

---

<details>
<summary><b>07 / Database Check</b></summary>

<br>

Lệnh này chỉ chạy `SELECT 1`, không ghi dữ liệu:

```powershell
.\.venv\Scripts\python.exe -c "from match_insight.database.engine import check_database_connection; print(check_database_connection())"
```

Kết quả:

```text
1
```

Sau đó:

```powershell
.\scripts\start_web.ps1
```

</details>

---

<div align="center">

`SETUP` ───────── `PRE` ───────── `DRAFT` ───────── `POST`

</div>

# Demo Flow

## 01 / Establish the Match

Chọn hai đội khác nhau.

Có thể tìm nhanh:

```text
HLE       → Hanwha Life Esports
Gen       → Gen.G
geng      → Gen.G
```

Search:

```text
case-insensitive
punctuation-insensitive
whitespace-insensitive
```

Mỗi đội cần:

```text
TOP
JUNGLE
MID
BOT
SUPPORT
```

Tổng cộng phải có **10 tuyển thủ khác nhau**.

Nếu không thấy đội hoặc tuyển thủ cần chọn, bỏ:

```text
Chỉ hiện tên có lịch sử
```

Nhập patch và tùy chọn thêm bối cảnh trận đấu.

Xác nhận lại:

```text
teams
sides
patch
players
positions
```

---

## 02 / Generate PRE

Nhấn:

```text
Tạo đánh giá PRE
```

Hệ thống:

```text
historical context
      │
      ▼
history filtering
      │
      ▼
feature engineering
      │
      ▼
three pipelines
      │
      ▼
PRE probabilities
      │
      ▼
persistence
```

Cutoff của phiên tương tác được xác định tại thời điểm tạo PRE:

```text
00:00 UTC
của ngày liền trước ngày tạo PRE theo UTC
```

POST giữ nguyên cutoff này.

Đây là quy tắc lọc lịch sử, không phải thời điểm draft thật được lấy trực tiếp từ giải đấu.

---

## 03 / Lock the Draft

Sau khi cấm/chọn hoàn tất:

```text
5 BLUE champions
5 RED champions
```

Tổng cộng phải có 10 tướng hợp lệ và không trùng nhau.

Nhấn:

```text
Tạo POST & so sánh
```

Cặp tuyển thủ–tướng chưa có lịch sử được ghi:

```text
Chưa ghi nhận
```

Cảnh báo bao gồm:

```text
player
champion
team
position
```

---

## 04 / Read the Shift

Bảng kết quả:

```text
MODEL                   PRE       POST       Δ

Baseline                 —          —        —
Logistic Regression      —          —        —
Random Forest             —          —        —
```

Baseline sử dụng tỷ lệ thắng BLUE từ tập train, vì vậy PRE và POST có thể bằng nhau.

Một model cho xác suất cao hơn ở một trận cụ thể **không có nghĩa model đó tốt hơn**.

Đánh giá mô hình cần sử dụng:

```text
Brier Score
Log Loss
ROC-AUC
Calibration
```

trên cùng evaluation set.

---

# Editing an Analysis

| CHANGE        | ACTION                             |
| ------------- | ---------------------------------- |
| Champion only | Giữ PRE, hủy POST cũ, tạo POST mới |
| Team          | Tạo PRE mới                        |
| Side          | Tạo PRE mới                        |
| Player        | Tạo PRE mới                        |
| Position      | Tạo PRE mới                        |
| Patch         | Tạo PRE mới                        |
| Match context | Tạo PRE mới                        |

Nếu persistence thất bại:

```text
fix database connection / write permission
                │
                ▼
            retry
```

Kết quả chỉ được công bố sau khi lưu thành công.

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

Sau đó:

```text
Tra cứu bản lưu
```

Hệ thống hiển thị lại:

```text
input
predictions
warnings
persistence state
```

Tra cứu bản lưu không phục hồi analysis session.

Model comparison provenance được lưu tại:

```text
input_snapshot.provenance.model_comparison
```

cùng transaction của evaluation canonical.

Các bản lưu cũ chỉ có một model vẫn được đọc nguyên trạng và không bị tính lại.

---

# Project Structure

```text
MATCH INSIGHT
│
├── frontend/
│   └── src/
│       React + TypeScript application
│
├── match_insight/
│   │
│   ├── api/
│   │   FastAPI · sessions · assets
│   │
│   ├── services/
│   │   PRE · POST · comparison · persistence
│   │
│   ├── features/
│   │   aggregation · historical filtering
│   │
│   ├── ml/
│   │   dataset · training · loading · inference
│   │
│   ├── database/
│   │   schema · queries · persistence
│   │
│   └── data_processing/
│       validation · normalization
│
├── artifacts/
│   trained models
│
├── assets/
│   champions · teams · players
│
├── data/
│   input · reference
│
├── reports/
│   validation · experiments
│
├── alembic/
│   PostgreSQL migrations
│
├── scripts/
│   ETL · training · maintenance
│
├── docs/
│   project documentation
│
├── streamlit_app.py
│   legacy interface
│
└── README.md
```

Các script như:

```text
apply_oracle_etl.py
apply_oracle_riot_v5_backfill.py
train_real_models.py
```

phục vụ ETL hoặc experiment và **không phải** bước cần thiết để khởi động ứng dụng web đã chuẩn bị.

---

# Verification

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

# Troubleshooting

<details>
<summary><b>No module named pip</b></summary>

<br>

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

Cài bản Python đầy đủ có hỗ trợ `venv` và `ensurepip`, sau đó tạo lại `.venv`.

Yêu cầu:

```text
Python 3.12.14
```

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

Không tự chỉnh các dependency version đã cố định.

</details>

<details>
<summary><b>DATABASE_URL missing / PostgreSQL connection failed</b></summary>

<br>

Kiểm tra:

```text
.env
username
password
host
port
database
PostgreSQL service
```

Đồng thời kiểm tra biến `DATABASE_URL` trong PowerShell có đang override `.env` hay không.

</details>

<details>
<summary><b>Model cannot be loaded</b></summary>

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

Không bỏ qua hash validation hoặc thay model bằng prediction giả.

</details>

<details>
<summary><b>Model environment mismatch</b></summary>

<br>

Đối chiếu:

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
<summary><b>PRE button disabled</b></summary>

<br>

Kiểm tra:

```text
two different teams
10 different players
all five positions
patch
input confirmation
```

</details>

<details>
<summary><b>POST button disabled</b></summary>

<br>

Cần:

```text
PRE successfully created
10 valid champions
no duplicated champion
```

</details>

<details>
<summary><b>Missing team / player / champion images</b></summary>

<br>

Kiểm tra:

```text
assets/champions/
assets/teams/
assets/players/
```

Ảnh không ảnh hưởng tới model inference.

</details>

<details>
<summary><b>start_web.ps1 not found</b></summary>

<br>

Chuyển về project root:

```powershell
cd "C:\Đồ án ngành"
```

Việc terminal hiển thị:

```text
(.venv)
```

không có nghĩa terminal đang đứng đúng thư mục.

</details>

<details>
<summary><b>Frontend does not show latest changes</b></summary>

<br>

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
```

Sau đó:

```text
Ctrl + F5
```

`-SkipBuild` chỉ sử dụng build hiện có.

</details>

<details>
<summary><b>Port 8000 is already in use</b></summary>

<br>

Nếu instance cũ vẫn chạy, mở instance đó.

Hoặc:

```powershell
.\scripts\start_web.ps1 -SkipBuild -Port 8001
```

Truy cập:

```text
http://127.0.0.1:8001
```

</details>

---

# Scope

Match Insight tập trung vào **pre-game analysis**.

Hệ thống không:

```text
fetch live draft automatically
update probability during the game
recommend champion picks
optimize draft strategy
provide betting recommendations
```

Người dùng chủ động cung cấp:

```text
match context
roster
patch
final champion lineup
```

Hệ thống sau đó sử dụng dữ liệu lịch sử đã được chuẩn hóa để tạo PRE và POST.

---

<div align="center">

<br>

◇

<br>

## MATCH INSIGHT

### BEFORE THE DRAFT   ───   AFTER THE DRAFT

*Two states of the same match.*

<br>

`PRE`      ◇      `DRAFT`      ◇      `POST`

<br>

<sub>Built around professional League of Legends match data.</sub>

<br><br>

◇

</div>
