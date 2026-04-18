#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThaiSmartAddress v7.0 — models.py
All data-transfer objects (dataclasses) used across the pipeline.

NEW FIXES applied (this version):
  [FIX-M1] Added ParseResult.from_dict() classmethod with proper type
            validation and coercion, replacing the raw __dataclass_fields__
            access in api.py's _dict_to_parse_result(). This prevents
            wrong-typed fields from silently corrupting ParseResult objects
            when malformed feedback payloads are submitted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GeoRecord:
    """Immutable snapshot of one canonical geographic row."""
    sub_district: str
    district:     str
    province:     str
    zipcode:      str


@dataclass
class ParseResult:
    """Structured output of SmartAddressParser.parse()."""
    status:         str           = "Success"
    receiver:       Optional[str] = None
    phone:          Optional[str] = None
    address_detail: Optional[str] = None
    sub_district:   Optional[str] = None
    district:       Optional[str] = None
    province:       Optional[str] = None
    zipcode:        Optional[str] = None
    tags:           List[str]     = field(default_factory=list)
    confidence:     float         = 0.0
    processing_ms:  float         = 0.0
    warnings:       List[str]     = field(default_factory=list)

    # FIX [FIX-M1]: Type-safe factory that validates each field before
    # constructing the dataclass. Replaces raw **{k:v for k,v in d.items()
    # if k in __dataclass_fields__} which accepted wrong-typed values.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParseResult":
        """
        Type-safe factory from a plain dict (e.g., parsed_output from feedback).

        Coerces known fields to their correct types and ignores unknown keys,
        so a malformed feedback payload cannot corrupt the resulting ParseResult.
        """
        valid_fields = set(cls.__dataclass_fields__)
        kwargs: Dict[str, Any] = {}

        for k, v in d.items():
            if k not in valid_fields:
                continue
            # Apply type coercion per field category
            if k in ("tags", "warnings"):
                if not isinstance(v, list):
                    v = list(v) if v is not None else []
                else:
                    v = [str(item) for item in v]
            elif k == "confidence":
                try:
                    v = float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    v = 0.0
            elif k == "processing_ms":
                try:
                    v = float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    v = 0.0
            elif k == "status":
                v = str(v) if v is not None else "Success"
            else:
                # Optional[str] fields — allow None or stringify
                v = str(v).strip() if v is not None else None
                if v == "":
                    v = None
            kwargs[k] = v

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        # Manual dict is 3–5× faster than dataclasses.asdict() for flat dataclasses.
        return {
            "status":         self.status,
            "receiver":       self.receiver,
            "phone":          self.phone,
            "address_detail": self.address_detail,
            "address":        self.address_detail,   # alias for legacy/test clients
            "sub_district":   self.sub_district,
            "district":       self.district,
            "province":       self.province,
            "zipcode":        self.zipcode,
            "tags":           list(self.tags),
            "confidence":     self.confidence,
            "processing_ms":  self.processing_ms,
            "warnings":       list(self.warnings),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class NERResult:
    """Output of the NER fallback extractor."""
    receiver:      Optional[str]   = None
    location_hint: Optional[str]   = None
    raw_entities:  List[tuple]     = field(default_factory=list)
    used_ner:      bool            = False


@dataclass
class CorrectionRecord:
    """Human-corrected address record for the HITL Data Flywheel."""
    original_text:    str
    parsed_output:    Dict[str, Any]
    corrected_output: Dict[str, Any]
    corrected_by:     str           = "admin"
    correction_type:  str           = "full"
    created_at:       str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id:       Optional[str] = None
    request_id:       Optional[str] = None
    model_version:    str           = "v7.0"


@dataclass
class ChannelMessage:
    """Normalised message extracted from any supported inbound channel."""
    text:        str
    channel:     str
    customer_id: Optional[str]       = None
    message_id:  Optional[str]       = None
    page_id:     Optional[str]       = None
    raw:         Optional[Dict[str, Any]] = None