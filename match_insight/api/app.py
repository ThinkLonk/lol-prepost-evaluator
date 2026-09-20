"""Local React/FastAPI application; HTTP routes do not implement ML rules."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from match_insight.api.schemas import PostInput, PreInput
from match_insight.api.service import ApiError, WebAnalysisService
from match_insight.features.pre import PreFeatureInputError

PROJECT = Path(__file__).resolve().parents[2]
FRONTEND = PROJECT / "frontend" / "dist"
LOGGER = logging.getLogger(__name__)

ERROR_MESSAGES = {
    "E_PRE_INPUT_INCOMPLETE": "Bối cảnh hoặc danh sách tuyển thủ chưa hợp lệ.",
    "E_LINEUP_INCOMPLETE": "Chọn đủ mười tướng trước khi tạo POST.",
    "E_LINEUP_INVALID": "Đội hình chưa hợp lệ. Kiểm tra tướng và vị trí của hai đội.",
    "E_CHAMPION_UNKNOWN": "Có tướng nằm ngoài danh mục hỗ trợ.",
    "E_EVAL_INCOMPATIBLE": "PRE và POST không còn cùng bối cảnh. Hãy tạo PRE mới.",
    "E_STORAGE_WRITE": "Chưa lưu được kết quả. Hãy thử lại; kết quả chưa được công bố.",
    "E_STORAGE_READ": "Không đọc được bản lưu. Kiểm tra mã đánh giá và kết nối dữ liệu.",
    "E_STORAGE_CONFLICT": "Bản đánh giá đã mất hiệu lực hoặc không còn khớp bối cảnh.",
    "E_MODEL_COMPARISON_MISSING": "Chưa có đủ bộ ba mô hình. Kiểm tra các tệp đã chuẩn bị.",
}


def create_app(service=None):
    service = service or WebAnalysisService()

    @asynccontextmanager
    async def lifespan(app):
        try:
            await run_in_threadpool(service.resources)
        except Exception as error:
            # The frontend can still show a useful retry state if DB/models are unavailable.
            LOGGER.error("Runtime initialization failed (%s)", type(error).__name__)
        yield

    app = FastAPI(title="Match Insight API", version="1.0.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(GZipMiddleware, minimum_size=1500, compresslevel=4)
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def timing(request, call_next):
        started = perf_counter()
        response = await call_next(request)
        response.headers["Server-Timing"] = f"total;dur={(perf_counter() - started) * 1000:.1f}"
        if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/media/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ApiError)
    async def api_error(request, error):
        return JSONResponse({"code": error.code, "message": error.message}, status_code=error.status)

    @app.exception_handler(PreFeatureInputError)
    async def domain_error(request, error):
        if error.code.startswith(("E_MODEL", "E_REAL", "E_DB", "E_EVIDENCE")):
            message = "Chưa nạp được dữ liệu hoặc mô hình hợp lệ. Kiểm tra bộ dữ liệu và thử lại."
            status = 503
        elif error.code == "E_STORAGE_WRITE":
            message, status = ERROR_MESSAGES[error.code], 503
        else:
            message = ERROR_MESSAGES.get(error.code, "Dữ liệu chưa đáp ứng điều kiện phân tích.")
            status = 409
        return JSONResponse({"code": error.code, "message": message}, status_code=status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"code": "E_INPUT_INVALID", "message":
                             "Đầu vào chưa hợp lệ. Kiểm tra hai đội, đủ mười tuyển thủ hoặc tướng, "
                             "patch và ô xác nhận."}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        LOGGER.error("Request failed (%s)", type(error).__name__)
        return JSONResponse({"code": "E_SERVICE_UNAVAILABLE", "message":
                             "Chưa xử lý được yêu cầu. Kiểm tra kết nối dữ liệu rồi thử lại."},
                            status_code=503)

    SessionToken = Annotated[str, Header(alias="X-Analysis-Session", min_length=36, max_length=36)]

    @app.get("/api/health")
    def health():
        return {"status": "ready" if service.runtime is not None else "not_ready",
                "models_loaded": service.models is not None}

    @app.get("/api/bootstrap")
    def bootstrap():
        return service.bootstrap()

    @app.post("/api/sessions", status_code=201)
    def create_session():
        return service.create_session()

    @app.get("/api/session")
    def state(x_session: SessionToken):
        return service.state(x_session)

    @app.post("/api/pre")
    def pre(body: PreInput, x_session: SessionToken):
        return service.create_pre(x_session, body)

    @app.post("/api/post")
    def post(body: PostInput, x_session: SessionToken):
        return service.create_post(x_session, body)

    @app.delete("/api/pre")
    def invalidate_pre(x_session: SessionToken):
        return service.invalidate(x_session)

    @app.delete("/api/post")
    def invalidate_post(x_session: SessionToken):
        return service.invalidate(x_session, post_only=True)

    @app.get("/api/evaluations/{evaluation_id}")
    def saved(evaluation_id: int):
        if evaluation_id <= 0:
            raise HTTPException(404)
        return service.read_saved(evaluation_id)

    @app.get("/api/media/{kind}/{filename}")
    def media(kind: str, filename: str):
        if kind not in {"teams", "players", "champions"} or Path(filename).name != filename:
            raise HTTPException(404)
        if Path(filename).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise HTTPException(404)
        boundary = PROJECT / "assets" / kind
        path = (boundary / filename).resolve()
        if not path.is_relative_to(boundary) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})

    if (FRONTEND / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str, request: Request):
        if path.startswith("api/"):
            raise HTTPException(404)
        index = FRONTEND / "index.html"
        if not index.is_file():
            return JSONResponse({"message": "Chạy npm run build trong frontend trước khi mở UI."},
                                status_code=503)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
