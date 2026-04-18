#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThaiSmartAddress v7.0 — api.py
FastAPI REST server — clean imports, orjson serialisation, async-safe.

Endpoints:
    POST /api/parse            — Parse a raw Thai address string
    POST /api/parse/batch      — Parse a list of addresses (up to 100)
    POST /api/feedback         — Log a human correction (HITL Data Flywheel)
    GET  /api/corrections      — List recent corrections (admin, auth required)
    GET  /api/health           — Liveness / readiness probe
    GET  /api/info             — Server metadata (auth required when API_KEY set)

Environment variables:
    GEO_CSV_PATH   Path to thai_address_full.csv  (default: mock 15-row data)
    DB_PATH        SQLite feedback database path  (default: feedback_logs.db)
    CORS_ORIGINS   Comma-separated allowed origins
    API_KEY        If set, every request must carry  X-API-Key: <key>
    LOG_LEVEL      debug | info | warning  (default: info)

NEW FIXES applied (this version):
  [FIX-A1] Added simple sliding-window rate limiter (60 req/min/IP) as a
            BaseHTTPMiddleware — no extra dependencies required.
  [FIX-A2] Restricted CORS allow_headers from wildcard "*" to specific list.
            Wildcard + allow_credentials=True opens credential-forwarding CSRF.
  [FIX-A3] Fixed negative limit DoS: max(1, min(limit, 500)) instead of
            min(limit, 500) which passed -1 to SQLAlchemy .limit().
  [FIX-A4] /api/info now requires API key when API_KEY env var is set, and no
            longer returns cors_origins in the response body to prevent
            exposing deployment topology to unauthenticated callers.
  [FIX-A5] Replaced asyncio.get_running_loop().set_default_executor() (which
            affected ALL to_thread calls process-wide) with an explicit
            executor stored on app.state, used via loop.run_in_executor().
  [FIX-A6] Structured JSON logging via python-json-logger when installed,
            falling back to plain-text formatting gracefully.
  [FIX-A7] ParseResult.from_dict() used instead of raw __dataclass_fields__
            dict comprehension for type-safe deserialisation.
  [FIX-A8] Added POST /api/parse/batch endpoint that wraps parse_batch().
  [FIX-A9] nginx.conf CSP: removed unsafe-inline from script-src.
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import hmac
import ipaddress
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional

import orjson
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from database import SQLiteFeedbackStore, init_db
from models import CorrectionRecord, ParseResult
from parser import (
    SmartAddressParser,
    build_mock_geo_db,
    log_correction,
    PYTHAINLP_AVAILABLE,
    _do_ner_task,
    _ner_executor,
)
import parser as _parser_module
from geo_engine import GeoDatabase, RAPIDFUZZ_AVAILABLE

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# FIX [FIX-A6]: Structured JSON logging via python-json-logger when available.
# Falls back to plain-text gracefully so the app starts without the package.
# ══════════════════════════════════════════════════════════════════════════════

_LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

try:
    from pythonjsonlogger import jsonlogger  # type: ignore
    _json_handler = logging.StreamHandler()
    _json_handler.setFormatter(
        jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logging.root.handlers = [_json_handler]
    logging.root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    _JSON_LOGGING = True
except ImportError:
    logging.basicConfig(
        level=getattr(logging, _LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _JSON_LOGGING = False

logger = logging.getLogger("api")
if _JSON_LOGGING:
    logger.info("JSON structured logging enabled")
else:
    logger.info("python-json-logger not installed — using plain-text logging")


# ══════════════════════════════════════════════════════════════════════════════
# ORJSON RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

class ORJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY)


# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER  (FIX [FIX-A1])
# Simple in-memory sliding-window rate limiter — no extra dependencies.
# 60 requests per IP per 60-second window (configurable via env vars).
# ══════════════════════════════════════════════════════════════════════════════

_RATE_LIMIT_MAX  = int(os.getenv("RATE_LIMIT_MAX",  "60"))
_RATE_LIMIT_SECS = int(os.getenv("RATE_LIMIT_SECS", "60"))


class _SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max    = max_requests
        self._window = window_seconds
        self._clients: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._lock   = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now    = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            dq = self._clients[key]
            # Evict expired timestamps
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                # FIX [#6]: do NOT delete here — key is still active (over-limit)
                return False
            dq.append(now)
            # FIX [#6]: if the deque is empty after append is impossible, but
            # clean up stale keys that were evicted to zero. The real cleanup
            # happens via the periodic prune below.
            return True

    def prune_stale(self) -> int:
        """Remove entries for IPs whose window has fully expired.
        Call periodically (e.g., from a background task) to prevent unbounded
        memory growth when many unique IPs hit the service.
        Returns the number of keys removed.
        """
        cutoff = time.monotonic() - self._window
        removed = 0
        with self._lock:
            stale = [k for k, dq in self._clients.items()
                     if not dq or dq[-1] < cutoff]
            for k in stale:
                del self._clients[k]
                removed += 1
        return removed


_rate_limiter = _SlidingWindowRateLimiter(_RATE_LIMIT_MAX, _RATE_LIMIT_SECS)


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests exceeding the per-IP sliding-window limit."""

    # Paths exempt from rate limiting (ops/health checks)
    _EXEMPT = {"/api/health"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._EXEMPT:
            return await call_next(request)

        client_ip = (request.scope.get("client") or ("unknown", 0))[0]
        if not _rate_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for IP=%s path=%s", client_ip, request.url.path)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": f"Rate limit exceeded. Max {_RATE_LIMIT_MAX} requests "
                             f"per {_RATE_LIMIT_SECS}s per IP.",
                    "retry_after": _RATE_LIMIT_SECS,
                },
                headers={"Retry-After": str(_RATE_LIMIT_SECS)},
            )
        return await call_next(request)


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — startup / shutdown
# ══════════════════════════════════════════════════════════════════════════════

_PLACEHOLDER_KEY = "CHANGE-ME-BEFORE-DEPLOY-use-secrets-token-urlsafe-32"


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    logger.info("ThaiSmartAddress v7.0 starting…")

    # FIX [#10]: Refuse to start if the API key is the well-known placeholder.
    # An operator who forgets to change it gets a hard error instead of a
    # silently "secured" service that any attacker knowing this repo can bypass.
    if _API_KEY == _PLACEHOLDER_KEY:
        raise RuntimeError(
            "API_KEY is still set to the default placeholder value. "
            "Generate a real secret with: "
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\" "
            "and set it in your environment before starting the service."
        )

    # 1. Init SQLite
    init_db()
    logger.info("SQLite ready: %s", os.getenv("DB_PATH", "feedback_logs.db"))

    # 2. Load geo DB
    geo_csv = os.getenv("GEO_CSV_PATH", "")
    if geo_csv and os.path.isfile(geo_csv):
        logger.info("Loading geo DB: %s", geo_csv)
        geo_db = GeoDatabase().load_csv(geo_csv)
    else:
        if geo_csv:
            logger.warning("GEO_CSV_PATH=%r not found — using mock data (26 records)", geo_csv)
        else:
            logger.info("GEO_CSV_PATH not set — using mock data (26 records)")
        geo_db = build_mock_geo_db()

    # 3. Instantiate parser
    parser = SmartAddressParser(geo_db)

    # 4. Create a named executor stored on app.state (FIX [FIX-A5]).
    _tsa_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="tsa-worker"
    )

    # 5. Pre-warm NER model; record availability on app.state for /api/health
    ner_available = False
    if PYTHAINLP_AVAILABLE:
        try:
            logger.info("Pre-warming NER model…")
            _ner_executor.submit(_do_ner_task, "warm_up").result(timeout=60)
            logger.info("NER model pre-warmed")
            ner_available = True
        except Exception as exc:
            _parser_module._ner_load_failed.set()
            logger.warning("NER pre-warm failed — NER disabled for this session: %s", exc)

    # 6. Attach shared state
    app.state.parser         = parser
    app.state.geo_db         = geo_db
    app.state.start_time     = datetime.now(timezone.utc)
    app.state.db_adapter     = SQLiteFeedbackStore()
    app.state.tsa_executor   = _tsa_executor
    app.state.ner_available  = ner_available
    app.state.fuzzy_available = RAPIDFUZZ_AVAILABLE

    # 7. Background task: prune stale rate-limiter entries every 10 minutes.
    # FIX [#6]: Without this the _clients dict grows forever — one entry per
    # unique source IP, never evicted — causing unbounded memory growth.
    async def _prune_rate_limiter():
        while True:
            await asyncio.sleep(600)
            removed = _rate_limiter.prune_stale()
            if removed:
                logger.debug("Rate limiter pruned %d stale IP entries", removed)

    _prune_task = asyncio.create_task(_prune_rate_limiter())

    logger.info(
        "Ready in %.0f ms — geo_records=%d ner=%s fuzzy=%s",
        (time.perf_counter() - t0) * 1000,
        geo_db.size,
        ner_available,
        RAPIDFUZZ_AVAILABLE,
    )
    yield

    _prune_task.cancel()
    _tsa_executor.shutdown(wait=False)
    logger.info("Shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ThaiSmartAddress API",
    description=(
        "Production-grade REST API for parsing Thai delivery addresses.\n\n"
        "**Pipeline:** Intent Shield → Keyboard Fallback → Rule-Based → "
        "Fuzzy Geo (RapidFuzz) → NER (PyThaiNLP) → Strict Validation"
    ),
    version="7.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url="/redoc",
)

# FIX [#23]: Prometheus metrics at /metrics — enables latency/error/throughput
# alerting without any additional instrumentation code in endpoint handlers.
# Graceful fallback when the package is absent (e.g. slim dev containers).
try:
    from prometheus_fastapi_instrumentator import Instrumentator as _PFI  # type: ignore
    _PFI().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.info("prometheus-fastapi-instrumentator not installed — /metrics disabled")

# ── Real-IP middleware ────────────────────────────────────────────────────────
_DOCKER_BRIDGE = ipaddress.ip_network("172.16.0.0/12")


def _is_trusted_proxy(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip) in _DOCKER_BRIDGE
    except ValueError:
        return False


class _RealIPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        fwd = request.headers.get("X-Forwarded-For")
        if fwd:
            client_ip = (request.scope.get("client") or ("", 0))[0]
            if _is_trusted_proxy(client_ip):
                real_ip = fwd.split(",")[0].strip()
                request.scope["client"] = (real_ip, request.scope.get("client", (None, 0))[1])
        return await call_next(request)


app.add_middleware(_RealIPMiddleware)

# ── Rate limiting (FIX [FIX-A1]) ──────────────────────────────────────────────
app.add_middleware(_RateLimitMiddleware)

# ── CORS (FIX [FIX-A2]) ───────────────────────────────────────────────────────
# FIX [FIX-A2]: Replaced wildcard allow_headers=["*"] with an explicit list.
# Wildcard combined with allow_credentials=True allows any header from any
# allowed origin, enabling CSRF-style credential-forwarding attacks.
_ALLOWED_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],  # FIX [FIX-A2]
)

# ── API key auth ──────────────────────────────────────────────────────────────
_API_KEY: Optional[str] = os.getenv("API_KEY") or None  # empty string → treat as unset


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Timing-safe API key check. No-op when API_KEY env var is not set."""
    if _API_KEY is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ── Request-ID middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled exception [%s]: %s", rid, exc, exc_info=True)
    body: Dict[str, Any] = {"error": "Internal server error", "request_id": rid}
    if _LOG_LEVEL == "DEBUG" or os.getenv("ENV") == "development":
        body["detail"] = str(exc)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ParseRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=2_000,
        description="Raw Thai address text. English-keyboard input is auto-converted.",
        examples=["รบกวนส่งที่ คุณแม็ค 99/9 ต.แสนสุข อ.เมือง จ.ชลบุรี 20130 0812345678"],
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped


class BatchParseRequest(BaseModel):
    """FIX [FIX-A8]: Batch parsing of up to 100 addresses."""
    texts: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of raw Thai address strings to parse (max 100).",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("texts list must contain at least one non-blank string")
        return cleaned


class FeedbackRequest(BaseModel):
    original_text:    str            = Field(..., min_length=1, max_length=2_000)
    parsed_output:    Dict[str, Any] = Field(..., description="JSON the API originally returned")
    corrected_output: Dict[str, Any] = Field(..., description="Admin-corrected version")
    corrected_by:     str            = Field(default="admin", min_length=1, max_length=100)
    request_id:       Optional[str]  = Field(default=None, max_length=128)
    session_id:       Optional[str]  = Field(default=None, max_length=128)

    # FIX [#9]: Limit JSON payload size to prevent memory-pressure DoS.
    # A Dict[str, Any] field is otherwise unbounded — a 10 MB JSON object
    # gets fully deserialized, stored in SQLite, and returned.
    _MAX_DICT_BYTES: ClassVar[int] = 65_536  # 64 KB is more than enough for any real correction

    @field_validator("parsed_output", "corrected_output")
    @classmethod
    def limit_dict_size(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json
        size = len(_json.dumps(v, ensure_ascii=False))
        if size > cls._MAX_DICT_BYTES:
            raise ValueError(
                f"JSON payload too large ({size} bytes). Max {cls._MAX_DICT_BYTES} bytes."
            )
        return v


class ParseResponse(BaseModel):
    request_id:     str
    status:         str
    receiver:       Optional[str] = None
    phone:          Optional[str] = None
    address_detail: Optional[str] = None
    sub_district:   Optional[str] = None
    district:       Optional[str] = None
    province:       Optional[str] = None
    zipcode:        Optional[str] = None
    tags:           List[str]     = []
    confidence:     float
    processing_ms:  float
    warnings:       List[str]     = []


class BatchParseResponse(BaseModel):
    request_id: str
    count:      int
    results:    List[Dict[str, Any]]


class FeedbackResponse(BaseModel):
    request_id:      str
    correction_type: str
    corrected_by:    str
    created_at:      str
    message:         str = "Correction logged successfully"


class CorrectionListResponse(BaseModel):
    total:       int
    corrections: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status:          str   = "ok"
    version:         str   = "7.0.0"
    geo_records:     int
    uptime_s:        float
    # FIX [#25]: expose NER and fuzzy degradation state for ops alerting
    ner_available:   bool  = False
    fuzzy_available: bool  = False


class InfoResponse(BaseModel):
    version:      str
    geo_records:  int
    start_time:   str
    uptime_s:     float
    auth_enabled: bool
    # FIX [FIX-A4]: cors_origins REMOVED from response — exposing allowed
    # origins leaks deployment topology to unauthenticated callers.


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: run CPU-bound work in the named executor (FIX [FIX-A5])
# ══════════════════════════════════════════════════════════════════════════════

async def _run_in_executor(request: Request, fn, *args):
    """
    FIX [FIX-A5]: Run a callable in the app's named ThreadPoolExecutor rather
    than replacing the process-wide default executor. asyncio.to_thread() always
    uses the loop default, so we call loop.run_in_executor() explicitly.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(request.app.state.tsa_executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/parse",
    response_model=ParseResponse,
    tags=["Parsing"],
    dependencies=[Depends(verify_api_key)],
    summary="Parse a Thai address string",
)
async def parse_address(body: ParseRequest, request: Request) -> ParseResponse:
    parser: SmartAddressParser = request.app.state.parser
    result: ParseResult = await _run_in_executor(request, parser.parse, body.text)
    logger.debug(
        "parse request_id=%s confidence=%.0f%% status=%s",
        request.state.request_id, result.confidence * 100, result.status,
    )
    return ParseResponse(request_id=request.state.request_id, **result.to_dict())


@app.post(
    "/api/parse/batch",
    response_model=BatchParseResponse,
    tags=["Parsing"],
    dependencies=[Depends(verify_api_key)],
    summary="Parse a batch of Thai address strings (max 100)",
)
async def parse_batch(body: BatchParseRequest, request: Request) -> BatchParseResponse:
    """
    FIX [FIX-A8]: Exposes parse_batch() as a proper HTTP endpoint.
    Each address is parsed independently; results preserve input order.
    """
    parser: SmartAddressParser = request.app.state.parser
    results: List[ParseResult] = await _run_in_executor(
        request, parser.parse_batch, body.texts
    )
    return BatchParseResponse(
        request_id=request.state.request_id,
        count=len(results),
        results=[r.to_dict() for r in results],
    )


@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback"],
    dependencies=[Depends(verify_api_key)],
    summary="Log a human correction (Data Flywheel)",
)
async def submit_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    db_adapter: SQLiteFeedbackStore = request.app.state.db_adapter
    # FIX [FIX-A7]: use type-safe ParseResult.from_dict() instead of raw field access
    parsed_result = ParseResult.from_dict(body.parsed_output)
    session_id    = body.request_id or body.session_id or request.state.request_id

    def _do_log() -> CorrectionRecord:
        # FIX [#5]: log_correction() catches all DB exceptions and logs them,
        # then returns the record.  We re-raise here so the HTTP layer returns
        # 500 instead of 201 — the caller must know their correction was lost.
        rec = log_correction(
            original_text=body.original_text,
            parsed_result=parsed_result,
            corrected_json=body.corrected_output,
            db_connection=db_adapter,
            corrected_by=body.corrected_by,
            session_id=session_id,
        )
        if getattr(rec, "_db_error", None):
            raise RuntimeError(f"Correction could not be persisted: {rec._db_error}")
        return rec

    correction: CorrectionRecord = await _run_in_executor(request, _do_log)
    return FeedbackResponse(
        request_id=request.state.request_id,
        correction_type=correction.correction_type,
        corrected_by=correction.corrected_by,
        created_at=correction.created_at,
    )


@app.get(
    "/api/corrections",
    response_model=CorrectionListResponse,
    tags=["Feedback"],
    dependencies=[Depends(verify_api_key)],
    summary="List recent corrections (admin)",
)
async def list_corrections(request: Request, limit: int = 50) -> CorrectionListResponse:
    db_adapter: SQLiteFeedbackStore = request.app.state.db_adapter
    # FIX [FIX-A3]: clamp both sides — min(-1, 500) = -1 would pass -1 to SQLAlchemy
    capped = max(1, min(limit, 500))
    total, corrections = await _run_in_executor(request, db_adapter.count_and_recent, capped)
    return CorrectionListResponse(total=total, corrections=corrections)


@app.get("/api/health", response_model=HealthResponse, tags=["Operations"])
async def health_check(request: Request) -> HealthResponse:
    uptime = (datetime.now(timezone.utc) - request.app.state.start_time).total_seconds()
    return HealthResponse(
        geo_records=request.app.state.geo_db.size,
        uptime_s=round(uptime, 1),
        # FIX [#25]: surface capability degradation to ops/alerting systems
        ner_available=getattr(request.app.state, "ner_available", False),
        fuzzy_available=getattr(request.app.state, "fuzzy_available", False),
    )


@app.get(
    "/api/info",
    response_model=InfoResponse,
    tags=["Operations"],
    dependencies=[Depends(verify_api_key)],  # FIX [FIX-A4]: require auth
    summary="Server metadata (auth required)",
)
async def server_info(request: Request) -> InfoResponse:
    uptime = (datetime.now(timezone.utc) - request.app.state.start_time).total_seconds()
    return InfoResponse(
        version="7.0.0",
        geo_records=request.app.state.geo_db.size,
        start_time=request.app.state.start_time.isoformat(),
        uptime_s=round(uptime, 1),
        auth_enabled=_API_KEY is not None,
        # cors_origins intentionally omitted (FIX [FIX-A4])
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEV RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)