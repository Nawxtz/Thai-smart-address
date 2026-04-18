#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThaiSmartAddress v7.0 — database.py
Persistent SQLite feedback store (Data Flywheel — Pillar 1).

NEW FIXES applied (this version):
  [FIX-D1] save_correction() now honours CorrectionRecord.created_at instead
            of silently computing a second datetime.now() call. Previously the
            record field was wasted — two timestamps could differ by
            milliseconds, and the record's creation time was not persisted.
            The field is now parsed from ISO-8601 string → aware datetime and
            written to the DB row. Falls back to utcnow() only if parsing fails.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import DateTime, Integer, String, Text, create_engine, func, text as sa_text, event as sa_event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logger = logging.getLogger("ThaiSmartAddress.database")

# ══════════════════════════════════════════════════════════════════════════════
# DB PATH VALIDATION — blocks path-traversal attacks on DB_PATH env var
# ══════════════════════════════════════════════════════════════════════════════

def _validate_db_path(raw: str) -> str:
    """FIX [#7]: Use os.path.realpath() to canonicalise the path, then check
    it falls within an allowed prefix. The old token-check ('if .. in raw')
    only caught ASCII dot-dot sequences and gave false confidence against
    symlink-based traversals and alternative encodings.
    Allowed prefixes: current working directory or /data (Docker volume).
    """
    import os as _os
    try:
        resolved = _os.path.realpath(raw)
    except (ValueError, OSError):
        logger.warning("DB_PATH=%r could not be resolved — using default", raw)
        return "feedback_logs.db"

    allowed = (
        _os.path.realpath("."),
        "/data",
        "/tmp",
    )
    for prefix in allowed:
        if resolved.startswith(prefix):
            return raw

    logger.warning(
        "DB_PATH=%r resolved to %r which is outside allowed prefixes %r — using default",
        raw, resolved, allowed,
    )
    return "feedback_logs.db"


_DB_PATH = _validate_db_path(os.getenv("DB_PATH", "/tmp/feedback_logs.db"))
_DB_URL  = f"sqlite:///{_DB_PATH}"

_engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False, "timeout": 15},
    echo=False,
    pool_pre_ping=False,
)


@sa_event.listens_for(_engine, "connect")
def _enable_wal(dbapi_conn, _record):
    """Switch SQLite to WAL mode — allows concurrent reads during writes."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.close()


_SessionLocal = sessionmaker(
    bind=_engine, autocommit=False, autoflush=False, expire_on_commit=False
)
logger.info("SQLite engine initialised → %s", _DB_URL)


# ══════════════════════════════════════════════════════════════════════════════
# ORM MODEL
# ══════════════════════════════════════════════════════════════════════════════

class _Base(DeclarativeBase):
    pass


class CorrectionLog(_Base):
    """
    ORM model for the correction_logs table.

    Columns:
        id               — Auto-increment PK
        request_id       — UUID of the originating HTTP request (for tracing)
        original_text    — Raw Thai address string the user submitted
        parsed_output    — JSON: what the API originally returned
        corrected_output — JSON: the human-corrected version
        corrected_by     — Admin / operator identifier
        correction_type  — "geo_only" | "name_only" | "full"
        created_at       — UTC timestamp (from CorrectionRecord, not re-computed)
    """
    __tablename__ = "correction_logs"

    id:               Mapped[int]           = mapped_column(Integer,     primary_key=True, autoincrement=True)
    request_id:       Mapped[str]           = mapped_column(String(128), nullable=False, index=True)
    original_text:    Mapped[str]           = mapped_column(Text,        nullable=False)
    parsed_output:    Mapped[str]           = mapped_column(Text,        nullable=False)
    corrected_output: Mapped[str]           = mapped_column(Text,        nullable=False)
    corrected_by:     Mapped[str]           = mapped_column(String(100), nullable=False, default="admin")
    correction_type:  Mapped[Optional[str]] = mapped_column(String(64),  nullable=True)
    created_at:       Mapped[datetime]      = mapped_column(
        DateTime(timezone=True), nullable=False,
        # FIX [#15]: index=True — count_and_recent() uses ORDER BY created_at DESC.
        # Without an index this is a full table scan on every /api/corrections call.
        index=True,
        server_default=func.now()
    )

    def parsed_output_dict(self) -> Dict[str, Any]:
        return json.loads(self.parsed_output)

    def corrected_output_dict(self) -> Dict[str, Any]:
        return json.loads(self.corrected_output)

    def __repr__(self) -> str:
        return (
            f"<CorrectionLog id={self.id} request_id={self.request_id!r} "
            f"type={self.correction_type!r} by={self.corrected_by!r}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    _Base.metadata.create_all(bind=_engine)
    with _engine.connect() as conn:
        try:
            conn.execute(sa_text("SELECT 1 FROM correction_logs LIMIT 1"))
        except Exception:
            pass
    logger.info("✅ DB schema verified — 'correction_logs' ready: %s", _DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK STORE
# ══════════════════════════════════════════════════════════════════════════════

class SQLiteFeedbackStore:
    """
    Persistent SQLite feedback store — unit-of-work pattern.
    Thread-safe: SQLite WAL mode + check_same_thread=False.
    """

    @staticmethod
    def _to_json(value: Any) -> str:
        if isinstance(value, str):
            try:
                json.loads(value)
                return value
            except json.JSONDecodeError:
                return json.dumps(value, ensure_ascii=False)
        return json.dumps(value, ensure_ascii=False, default=str)

    # FIX [FIX-D1]: Parse CorrectionRecord.created_at (ISO-8601 string) back to
    # an aware datetime and write it to the DB row. Previously a fresh
    # datetime.now() was computed here, wasting the field on the record and
    # creating a subtle timestamp mismatch between the domain object and the DB.
    @staticmethod
    def _parse_created_at(raw: Optional[str]) -> datetime:
        """Parse ISO-8601 string from CorrectionRecord into an aware datetime."""
        if not raw:
            return datetime.now(timezone.utc)
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            logger.warning("Could not parse created_at=%r — using utcnow()", raw)
            return datetime.now(timezone.utc)

    def save_correction(self, record: Any) -> None:
        """Persist a CorrectionRecord to SQLite."""
        req_id = (
            str(getattr(record, "request_id", None) or "")
            or str(getattr(record, "session_id", None) or "")
        )
        # FIX [FIX-D1]: use the record's own timestamp, not a new one
        created_at = self._parse_created_at(getattr(record, "created_at", None))

        row = CorrectionLog(
            request_id       = req_id,
            original_text    = str(getattr(record, "original_text",    "")),
            parsed_output    = self._to_json(getattr(record, "parsed_output",    {})),
            corrected_output = self._to_json(getattr(record, "corrected_output", {})),
            corrected_by     = str(getattr(record, "corrected_by", "admin") or "admin"),
            correction_type  = getattr(record, "correction_type", None),
            created_at       = created_at,
        )
        with _SessionLocal() as session:
            session.add(row)
            session.commit()
            logger.info(
                "✅ Correction #%d saved [req=%r type=%r by=%r created_at=%s]",
                row.id, row.request_id, row.correction_type,
                row.corrected_by, created_at.isoformat(),
            )

    def count_and_recent(self, limit: int = 50) -> Tuple[int, List[Dict[str, Any]]]:
        """Return (total_count, recent_rows) in a single session."""
        with _SessionLocal() as session:
            total = session.query(func.count(CorrectionLog.id)).scalar() or 0
            rows  = (
                session.query(CorrectionLog)
                .order_by(CorrectionLog.created_at.desc())
                .limit(limit)
                .all()
            )
            recent = [
                {
                    "id":               row.id,
                    "request_id":       row.request_id,
                    "original_text":    row.original_text,
                    "parsed_output":    row.parsed_output_dict(),
                    "corrected_output": row.corrected_output_dict(),
                    "corrected_by":     row.corrected_by,
                    "correction_type":  row.correction_type,
                    "created_at":       row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        return total, recent

    def count(self) -> int:
        with _SessionLocal() as session:
            return session.query(func.count(CorrectionLog.id)).scalar() or 0