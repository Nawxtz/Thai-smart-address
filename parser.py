#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThaiSmartAddress v7.0 — parser.py
Unified hybrid NLP pipeline.

Parse pipeline per request:
    Phase -1  Intent Shield        — reject non-address chat junk (O(1))
    Phase  0  Keyboard Fallback    — convert English-layout Thai input
    Phase  A  Rule-Based Core      — normalise -> phone -> zip -> geo -> tags -> leftovers
    Phase  B  Fuzzy Geo            — RapidFuzz typo correction (if Phase A geo missed)
    Phase  C  NER Fallback         — PyThaiNLP thainer (if receiver still missing)
    Final     Strict Validation    — confidence + flag low-quality results

NEW FIXES applied (this version):
  [FIX-P1] Removed duplicate module-level _SKIP_WORDS definition. It was
            defined at module scope AND re-defined inside keyboard_fallback(),
            making the outer definition silent dead code.
  [FIX-P2] Text is normalised ONCE in parse() and passed to both
            _rule_based_parse() and _apply_fuzzy_geo(). Previously
            _apply_fuzzy_geo() called _normalise() a second time on an already-
            normalised string, wasting an O(n) regex pass on every request.
  [FIX-P3] NER worker call in ner_extract() now has a 10-second timeout.
            Without it, a malformed string could hang the single NER thread
            permanently, blocking every subsequent NER call for all users.
"""
from __future__ import annotations

import concurrent.futures
import logging
import re
import time
import threading
import unicodedata
import warnings as _warnings
from typing import Dict, List, Optional, Set, Tuple

from constants import (
    ABBREV_EXPAND, ADDRESS_DETAIL_RE, CHAT_JUNK_PATTERN, CONNECTOR_PATTERN,
    FUZZY_THRESHOLD, HONORIFIC_PATTERN, HONORIFIC_SKIP_SET,
    INTENT_NEG_KEYWORDS,
    NAME_STOP_WORDS, NER_FALLBACK_THRESHOLD,
    PHONE_LABEL_PATTERN, PHONE_RE, RECEIVER_LABEL_RE, TAG_PATTERNS,
    THAI_CHAR_RE, ZIPCODE_COMPLAINT_RE, ZIPCODE_RE,
    expand_abbreviations,
)
from geo_engine import FuzzyGeoMatcher, GeoDatabase, RAPIDFUZZ_AVAILABLE, get_geo_strip_patterns
from models import CorrectionRecord, NERResult, ParseResult

logger = logging.getLogger("ThaiSmartAddress")

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: PyThaiNLP
# ──────────────────────────────────────────────────────────────────────────────

PYTHAINLP_AVAILABLE = False

try:
    from pythainlp.tag import NER as _ThaiNER  # type: ignore
    PYTHAINLP_AVAILABLE = True
    logger.info("PyThaiNLP detected — NER fallback enabled")
except ImportError:
    _warnings.warn("pythainlp not installed — NER fallback DISABLED.", stacklevel=2)


# NER worker: single dedicated thread prevents OOM (model ~200 MB)
_ner_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ner-worker"
)
_global_ner_model = None
# FIX [#20]: Use threading.Event instead of bare bool for thread-safe signalling.
# The NER thread sets this flag; the TSA worker threads read it. A plain bool
# assignment is safe on CPython (GIL) but is unsafe on PyPy / no-GIL Python.
# threading.Event uses an internal Condition + RLock for full memory-barrier semantics.
_ner_load_failed = threading.Event()


def _do_ner_task(text: str) -> List[Tuple[str, str]]:
    global _global_ner_model

    if _ner_load_failed.is_set():
        raise RuntimeError("NER model previously failed to load — skipping")

    if _global_ner_model is None:
        try:
            _global_ner_model = _ThaiNER("thainer")
        except TypeError:
            try:
                _global_ner_model = _ThaiNER("thainer", pos=False, crf_parallel=False)
            except Exception as exc:
                _ner_load_failed.set()
                raise RuntimeError(f"NER model load failed: {exc}") from exc
        except Exception as exc:
            _ner_load_failed.set()
            raise RuntimeError(f"NER model load failed: {exc}") from exc

    return _global_ner_model.get_ner(text)


# ──────────────────────────────────────────────────────────────────────────────
# KEYBOARD LAYOUT FALLBACK  (Phase 0)
# ──────────────────────────────────────────────────────────────────────────────

# FIX [FIX-P1]: Removed the duplicate module-level _SKIP_WORDS definition.
# The authoritative set is defined ONLY inside keyboard_fallback() below.

def keyboard_fallback(text: str):
    """Keyboard fallback removed — was corrupting English names/IDs."""
    return text, False



# ──────────────────────────────────────────────────────────────────────────────
# NER FALLBACK  (Phase C)
# ──────────────────────────────────────────────────────────────────────────────

_NER_PERSON_TAGS:   Set[str] = {"B-PERSON", "I-PERSON"}
_NER_LOCATION_TAGS: Set[str] = {"B-LOCATION", "I-LOCATION", "B-LOC", "I-LOC"}

# FIX [FIX-P3]: Added timeout=10 to .result() call.
# Without this, a hanging NER model on one request blocks the single-worker
# executor permanently, causing every subsequent NER call to queue indefinitely.
_NER_TIMEOUT_SECONDS: int = 10


def ner_extract(text: str) -> NERResult:
    """
    Run PyThaiNLP 'thainer' NER via the dedicated single-worker thread.
    Degrades gracefully if PyThaiNLP is absent or the model fails.
    """
    result = NERResult()
    if not PYTHAINLP_AVAILABLE:
        return result

    try:
        # FIX [FIX-P3]: timeout prevents permanent hang on malformed input
        tagged: List[Tuple[str, str]] = _ner_executor.submit(
            _do_ner_task, text
        ).result(timeout=_NER_TIMEOUT_SECONDS)

        result.raw_entities = tagged
        result.used_ner = True

        person_tokens = [
            tok for tok, tag in tagged
            if tag in _NER_PERSON_TAGS and not re.match(r"^[\d/]+$", tok) and tok
        ]
        if person_tokens:
            result.receiver = "".join(person_tokens).strip()

        location_tokens = [tok for tok, tag in tagged if tag in _NER_LOCATION_TAGS and len(tok) > 1]
        if location_tokens:
            result.location_hint = " ".join(location_tokens).strip()

    except concurrent.futures.TimeoutError:
        logger.warning("NER extraction timed out after %ds — skipping for this request", _NER_TIMEOUT_SECONDS)
        result.used_ner = False
    except Exception as exc:
        logger.warning("NER extraction failed: %s", exc)
        result.used_ner = False

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SMART ADDRESS PARSER
# ══════════════════════════════════════════════════════════════════════════════

class SmartAddressParser:
    """
    Production-grade Thai address parser — Hybrid Elimination Strategy.
    """

    _CONFIDENCE_FIELDS = [
        "receiver", "phone", "address_detail",
        "sub_district", "district", "province", "zipcode",
    ]

    def __init__(
        self,
        geo_db: GeoDatabase,
        fuzzy_threshold: int = FUZZY_THRESHOLD,
        ner_threshold: float = NER_FALLBACK_THRESHOLD,
    ) -> None:
        if not geo_db.is_loaded():
            raise ValueError("GeoDatabase must be loaded before creating the parser.")
        self._geo             = geo_db
        self._valid_zips      = geo_db.valid_zipcodes
        self._fuzzy           = FuzzyGeoMatcher(geo_db)
        self._fuzzy_threshold = fuzzy_threshold
        self._ner_threshold   = ner_threshold
        logger.info(
            "SmartAddressParser ready — geo=%d NER=%s Fuzzy=%s",
            geo_db.size, PYTHAINLP_AVAILABLE, RAPIDFUZZ_AVAILABLE,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, raw_text: str) -> ParseResult:
        t0 = time.perf_counter()

        # FIX [#19]: Strip chat junk BEFORE the Intent Shield.
        # The shield runs on the raw string and looks for negative keywords.
        # "โอนแล้วค่ะ ต.แสนสุข อ.เมืองชลบุรี จ.ชลบุรี" would be rejected
        # because "โอนแล้ว" is a neg-keyword and digit_density is 0%. Pre-
        # stripping junk tokens removes the false signal before classification.
        preshield_text = CHAT_JUNK_PATTERN.sub(" ", raw_text).strip() or raw_text

        # Phase -1: Intent Shield (run on junk-stripped text)
        reject = self._intent_shield(preshield_text, t0)
        if reject:
            return reject

        # Phase 0: Keyboard Layout Fallback (on original raw text for accuracy)
        text, kb_converted = keyboard_fallback(raw_text)

        # FIX [FIX-P2]: Normalise ONCE here, then pass the result to both
        # _rule_based_parse and _apply_fuzzy_geo. Previously _apply_fuzzy_geo
        # called _normalise() a second time on an already-normalised string.
        try:
            normalised = self._normalise(text)
        except ValueError as exc:
            result = ParseResult(
                status="Error",
                warnings=[str(exc)],
                processing_ms=self._elapsed(t0),
            )
            return result

        # Phase A: Rule-Based Core (receives pre-normalised text)
        result = self._rule_based_parse(normalised, t0)
        if result.status == "Error":
            return result

        if kb_converted:
            result.warnings.append("[KeyboardFallback] Input auto-converted to Thai")

        # Phase B: Fuzzy Geo Correction
        geo_incomplete = not all([result.province, result.district, result.sub_district])
        if geo_incomplete and RAPIDFUZZ_AVAILABLE:
            # FIX [FIX-P2]: pass already-normalised text — no second _normalise call
            result = self._apply_fuzzy_geo(normalised, result)

        # Phase C: NER Fallback
        if result.receiver is None and PYTHAINLP_AVAILABLE:
            result = self._apply_ner(normalised, result)

        # Final: Strict Validation + Confidence
        result = self._compute_confidence(result)
        result = self._strict_validate(result)
        result.processing_ms = self._elapsed(t0)
        return result

    def parse_batch(self, texts: List[str]) -> List[ParseResult]:
        """FIX [#26]: Each address is isolated in its own try/except.
        A crash on one input must not abort the remaining 99 items."""
        results: List[ParseResult] = []
        for t in texts:
            try:
                results.append(self.parse(t))
            except Exception as exc:
                logger.error("parse_batch: unhandled error for input %r: %s", t[:80], exc, exc_info=True)
                results.append(ParseResult(
                    status="Error",
                    confidence=0.0,
                    warnings=[f"Internal error during parsing: {exc}"],
                ))
        return results

    # ── Phase -1 ──────────────────────────────────────────────────────────────

    def _intent_shield(self, raw_text: str, t0: float) -> Optional[ParseResult]:
        text_lower = raw_text.lower()
        has_neg_kw = any(kw in text_lower for kw in INTENT_NEG_KEYWORDS)
        if not has_neg_kw:
            return None

        digits = re.sub(r"[^0-9]", "", raw_text)
        density = len(digits) / max(len(raw_text.strip()), 1)
        # [FIX-P4]: Use structured-digit check instead of digit density.
        # Density >= 4% passed any 3-digit monetary amount causing FPs.
        # [FIX-P6]: Before accepting a zipcode as a bypass signal, verify it is
        # not a monetary amount. "ยอด 30000 บาท" matches ZIPCODE_RE (30000 is
        # a real NE zipcode) but is a payment line. Check for monetary context
        # (preceded by ยอด or followed by บาท); if present, do not bypass.
        if PHONE_RE.search(raw_text):
            return None
        zip_match = ZIPCODE_RE.search(raw_text)
        if zip_match:
            before_zip = raw_text[:zip_match.start()]
            after_zip  = raw_text[zip_match.end():]
            monetary = (
                re.search(r'ยอด\s*$', before_zip.strip()) or
                after_zip.lstrip().startswith('บาท')
            )
            if not monetary:
                return None

        result = ParseResult(
            status="Not an Address",
            confidence=0.0,
            processing_ms=self._elapsed(t0),
            warnings=[
                f"[IntentShield] Rejected: non-address intent "
                f"(neg_keyword=True, digit_density={density:.1%})"
            ],
        )
        logger.debug("IntentShield rejected input (density=%.1f%%): %r", density * 100, raw_text[:80])
        return result

    # ── Phase A: Rule-Based Core ──────────────────────────────────────────────

    # FIX [FIX-P2]: Accepts pre-normalised text — no longer calls _normalise internally.
    def _rule_based_parse(self, normalised_text: str, t0: float) -> ParseResult:
        result = ParseResult()
        text = normalised_text

        result.phone,   text = self._extract_phone(text)
        text = re.sub(
            r'(?<!\d)(?:0[6-9]\d[\s\-\./]?\d{3,4}[\s\-\./]?\d{3,4}|0[2-5]\d[\s\-\./]?\d{3}[\s\-\./]?\d{3,4})(?!\d)',
            ' ', text
        )  # Strip remaining phones after first extraction
        result.zipcode, text = self._extract_zipcode(text)

        geo_rec, geo_score, geo_warns = self._geo.lookup(text, zipcode_hint=result.zipcode)
        result.warnings.extend(geo_warns)

        if geo_rec:
            # Only set sub_district when it was explicitly matched in the text
            # (score contribution from sub_district = 4 points).
            # When score ≤ 3 the best match came only from district+province/zipcode,
            # meaning no sub_district token appeared — leave it as None rather than
            # guessing the first record in that zip bucket.
            sub_in_text = geo_score >= 4
            if sub_in_text:
                result.sub_district = geo_rec.sub_district
            result.district     = geo_rec.district
            result.province     = geo_rec.province
            if not result.zipcode:
                result.zipcode  = geo_rec.zipcode
            text = self._strip_geo_tokens(text, geo_rec)

        result.tags, text = self._extract_tags(text)
        result.receiver, result.address_detail = self._extract_receiver_and_address(text)
        return result

    # ── Step helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string")
        text = unicodedata.normalize("NFC", text)
        # FIX [#18]: Normalize Thai digits ๐–๙ to ASCII 0–9.
        # Without this, zipcodes/phones written in Thai numerals (e.g. ๒๐๑๓๐)
        # are invisible to ZIPCODE_RE, PHONE_RE, and the digit-density
        # calculation in the Intent Shield.
        text = text.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
        # Strip invisible / zero-width Unicode chars
        text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0]", " ", text)
        text = re.sub(r"[\n\r\t]+", " ", text)
        # Inject spaces between Thai chars and digits (no-space input like "เอกราช0894561234บ้านเลขที่9")
        text = re.sub(r'([\u0E00-\u0E7F])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([\u0E00-\u0E7F])', r'\1 \2', text)
        # Inject space before geo abbreviation prefixes glued to previous Thai word
        # "บางโฉลงอ.บางพลีจ.สมุทรปราการ" → "บางโฉลง อ.บางพลี จ.สมุทรปราการ"
        text = re.sub(r'([\u0E00-\u0E7F])([ตอจมซถ]\.)', r'\1 \2', text)
        # Normalise decomposed Thai sara-am (ํา → ำ) from some keyboards/copy-paste
        # U+0E4D U+0E32  → U+0E33  (affects ตำ อำ จำ etc.)
        text = text.replace('\u0e4d\u0e32', '\u0e33')
        # Fix common geo-prefix typos that survive sara-am normalisation:
        #   ตำบน → ตำบล  (น/ล swap on Thai keyboard)
        #   อำเพอ → อำเภอ  (เพอ → เภอ)
        #   จังหว้ด → จังหวัด  (สระโอ้ → สระอา + วัจ)
        text = re.sub(r'ตำบน(?=[\u0E00-\u0E7F])', 'ตำบล', text)
        text = re.sub(r'อำเพอ(?=[\u0E00-\u0E7F])', 'อำเภอ', text)
        text = re.sub(r'จังหว้ด(?=[\u0E00-\u0E7F])', 'จังหวัด', text)
        # "Last delivery connector wins" — but ONLY when the preamble contains
        # geo signals (ตำบล/อำเภอ/zipcode), meaning the customer wrote their old
        # address first. Without that guard, "แพรว 0998887777 ค่ะ จัดส่งที่ ..."
        # would strip the receiver+phone before the connector.
        _LATE_CONN_RE = re.compile(
            r'(?s)(.{15,}?)(ส่งมาที่|จัดส่งที่|รบกวนส่งที่|ส่งไปที่)\s*[:\-]?\s*'
        )
        _GEO_SIGNAL_RE = re.compile(r'ตำบล|แขวง|อำเภอ|เขต|จังหวัด|\d{5}')
        lm = _LATE_CONN_RE.match(text)
        _PHONE_IN_PREAMBLE = re.compile(r'0\d{8,9}|\+66\d{9}')
        if (lm and lm.end() < len(text)
                and _GEO_SIGNAL_RE.search(lm.group(1))
                and not _PHONE_IN_PREAMBLE.search(lm.group(1))):
            text = text[lm.end():]

        # [FIX-N2] Strip negated parenthetical content.
        # "...(แต่อันนี้ไม่ต้องส่งนะ)..." tells the admin to ignore that block.
        text = re.sub(r'\([^)]{2,120}(?:ไม่ต้องส่ง|ไม่ต้อง|อย่าส่ง|ยกเลิก)[^)]{0,60}\)', ' ', text)

        text = re.sub(r'\([^)]{2,120}(?:ไม่ต้องส่ง|ไม่ต้อง|อย่าส่ง|ยกเลิก)[^)]{0,60}\)', ' ', text)
        text = CHAT_JUNK_PATTERN.sub(" ", text)
        # Strip social media handle lines: "Line ID: xxx", "FB: xxx", "IG: xxx"
        # These must be removed before receiver extraction or they corrupt the name field.
        # Strip social media handle + all trailing tokens until next key-value pattern
        # "FB: ก้องเกียรติ สุดเท่" → " " (strips both words after FB:)
        text = re.sub(
            r'(?i)(?:line\s*(?:id|ID)|facebook|fb|instagram|ig|twitter)\s*[:\-]?\s*[^\n\r:]+?(?=\s+(?:[ก-๙A-Za-z]+\s*:|$|\d{5})|$)',
            ' ', text
        )
        text = re.sub(r'(?:เบอร์สำรอง|เบอร์หลัก|เบอร์รอง)', ' ', text)
        text = re.sub(r'(?:^|\s)(?:หลัก|สำรอง|รอง)(?=\s|$)', ' ', text)
        # Strip phone labels WITH optional trailing colon/dash
        text = re.sub(
            r'(?i)(?:รหัสไปรษณีย์|เบอร์โทรศัพท์|เบอร์โทร|โทรศัพท์|โทร|tel\.|tel|ติดต่อ|เบอร์)\s*[:\-]?\s*',
            ' ', text
        )
        text = re.sub(r'([฀-๿])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([฀-๿])', r'\1 \2', text)
        text = re.sub(r'ปณจ\.?', 'ปณจ', text)
        # FIX [P-F1]: Strip "รหัสไปรษณีย์ [customer complaint]" noise.
        text = ZIPCODE_COMPLAINT_RE.sub("", text)
        # [FIX-N3] Strip phone-extension suffixes ("ต่อ 15") before they bleed
        # into address_detail. Must run after phone-label stripping.
        text = re.sub(r'\s+ต่อ\s*\d+', ' ', text)
        text = expand_abbreviations(text)
        # [FIX-N4] Strip highway direction phrases e.g. "ขาเข้า กรุงเทพมหานคร"
        # "กทม." in "ขาเข้า กทม." means direction toward Bangkok, not the delivery province.
        # Must run after abbreviation expansion so "กทม." is already expanded.
        text = re.sub(r'ขา(?:เข้า|ออก)\s+(?:กรุงเทพมหานคร|[฀-๿]{3,20})', ' ', text)
        # [FIX-N10] Strip ALL inline "word: value" labels
        text = re.sub(r'(?:^|(?<=\s))(?:ชื่อ|ที่อยู่|ตำบล|อำเภอ|จังหวัด|ไปรษณีย์|Line\s*ID|FB|IG|Twitter)\s*[:\-]\s*', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:^|(?<=\s))[ก-๙A-Za-z][ก-๙A-Za-z_0-9]*\s*:\s*', ' ', text)
        # Strip stray trailing dot after Thai char (end-of-string only, not mid-string)
        # Mid-string removal was breaking น.ส. → น.ส (dropped the dot before space)
        text = re.sub(r"([฀-๿])\.$", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_phone(text: str) -> Tuple[Optional[str], str]:
        m = PHONE_RE.search(text)
        if not m:
            return None, text
        raw = re.sub(r"[\s\-\./]", "", m.group())
        if raw.startswith("+66"):
            raw = "0" + raw[3:]
        elif raw.startswith("66") and not raw.startswith("0"):
            raw = "0" + raw[2:]
        text = text[: m.start()] + " " + text[m.end():]
        return raw, re.sub(r"\s+", " ", text).strip()

    def _extract_zipcode(self, text: str) -> Tuple[Optional[str], str]:
        matches = list(ZIPCODE_RE.finditer(text))
        if not matches:
            return None, text
        for m in matches:
            candidate = m.group()
            if candidate in self._valid_zips:
                text = text[: m.start()] + " " + text[m.end():]
                return candidate, re.sub(r"\s+", " ", text).strip()
        m    = matches[0]
        text = text[: m.start()] + " " + text[m.end():]
        return m.group(), re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _strip_geo_tokens(text: str, rec) -> str:
        for prefix_pat, bare_pat in get_geo_strip_patterns(rec):
            text = prefix_pat.sub(" ", text)
            text = bare_pat.sub(" ", text)
        text = re.sub(ZIPCODE_RE, " ", text)
        # FIX [P-F2]: Remove orphaned geo-prefix stubs left when district name
        # was split across tokens (e.g. "เมือง เชียงใหม่" → "เมือง" stub after
        # "เมืองเชียงใหม่" is stripped as a unit).  Only strip when surrounded
        # by whitespace/boundary to avoid clipping legitimate place-name parts.
        text = re.sub(r"(?<!\S)(?:เมือง|อำเภอ|เขต|จังหวัด|ตำบล|แขวง)(?!\S)", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_tags(text: str) -> Tuple[List[str], str]:
        tags: List[str] = []
        for pattern, tag in TAG_PATTERNS:
            if pattern.search(text):
                tags.append(tag)
                text = pattern.sub(" ", text)
        return tags, re.sub(r"\s+", " ", text).strip()

    def _extract_receiver_and_address(
        self, text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        # [FIX-T8] Protect "เลขที่ N" from connector stripping — preserves house nums
        text = re.sub(r'เลขที่\s*(\d)', r'\1', text)
        text = CONNECTOR_PATTERN.sub(" ", text)
        text = re.sub(r"\b(ด้วยนะ|หน่อยนะ|ด้วย)\b", " ", text)
        text = re.sub(r'[ก-๙A-Za-z_]+\s*[:\-]\s*', ' ', text)  # FIX-R1 orphan labels
        text = re.sub(r'(?:ขอบคุณมากๆ|ขอบคุณนะ|ขอบคุณ|รีบส่งนะ|รีบส่ง|รีบนะ|รีบ)(?:\s|$)', ' ', text)
        text = re.sub(r'(?:^|\s)(?:กท|มา|ด่วนๆ)(?=\s|$)', ' ', text)
        text = re.sub(r"\s+", " ", text).strip()

        receiver: Optional[str] = None

        # Pass -1: Explicit "ชื่อผู้รับ:" label anywhere in text  [FIX P-F3]
        # Real-world: customers append "ชื่อผู้รับ: ป้าสมศรี สู้ชีวิต" at the end.
        # This pass has highest priority — if present, trust it unconditionally.
        label_m = RECEIVER_LABEL_RE.search(text)
        if label_m:
            candidate = label_m.group(1).strip()
            candidate = re.sub(r"[)\]}>\"\']+$", "", candidate).strip()
            if len(candidate) >= 2:
                receiver = candidate
                # Remove the entire label+name span from text
                text = text[:label_m.start()] + " " + text[label_m.end():]
                text = re.sub(r"\s+", " ", text).strip()

        # Pass A: Honorific anchor — include honorific in returned receiver
        # e.g. "คุณจิรายุ" → receiver = "คุณจิรายุ" (not just "จิรายุ")
        if receiver is None:
            hm = HONORIFIC_PATTERN.search(text)
            if hm:
                try:
                    name_token = hm.group("n")
                except IndexError:
                    name_token = hm.group(1)

                start_idx = hm.start()
                end_idx   = hm.end()

                # Full span from honorific start to end of matched name
                full_honorific_span = text[start_idx:end_idx].strip()

                after_text = text[end_idx:]
                second_m   = re.match(r"\s*([^\s\d/]{2,30})(?=\s|$)", after_text)

                if second_m:
                    candidate = second_m.group(1)
                    is_stop   = candidate in NAME_STOP_WORDS or any(
                        candidate.startswith(sw) for sw in NAME_STOP_WORDS
                    )
                    if not is_stop:
                        receiver  = f"{full_honorific_span} {candidate}".strip()
                        end_idx  += second_m.end()
                    else:
                        receiver  = full_honorific_span
                else:
                    receiver = full_honorific_span

                if receiver:
                    text = text[:start_idx] + " " + text[end_idx:]

        # If receiver was set by Pass -1 (label), still strip pre-address preamble
        if receiver is not None:
            _pre = ADDRESS_DETAIL_RE.search(text)
            if _pre and _pre.start() > 0:
                text = text[_pre.start():].strip()
        # Pass B: house-number anchor
        if receiver is None:
            am = ADDRESS_DETAIL_RE.search(text)
            if am:
                before = text[: am.start()].strip()
                if before:
                    clean_name = re.sub(r"(?i)(tel|โทร|เบอร์โทร)\s*$", "", before).strip()
                    trimmed: List[str] = []
                    started = False
                    for tok in clean_name.split():
                        is_stop = tok in NAME_STOP_WORDS or any(
                            tok.startswith(sw) for sw in NAME_STOP_WORDS
                        )
                        if is_stop:
                            if started:
                                # Stop-word after real name tokens → end of name
                                break
                            else:
                                # FIX [P-F5]: Leading noise token (e.g. แอดมิน) before
                                # the real receiver — skip it instead of aborting.
                                continue
                        else:
                            started = True
                            trimmed.append(tok)
                    clean_name = " ".join(trimmed).strip()
                    if clean_name:
                        receiver = clean_name
                        # [FIX-P2] Strip the entire "before" span from text, not just
                        # the extracted receiver tokens. The old approach tried
                        # re.sub("^" + receiver, ...) which failed when stop-word
                        # prefixes (e.g. "แอดมิน") preceded the receiver and were
                        # already skipped in the loop — leaving them in address_detail.
                        text = text[am.start():].strip()

        # [FIX-P6] Strip leading company/org prefix before building digit/word.
        # GUARD: only fire when text starts with a company marker (บจก/บริษัท/หจก)
        # within the first 40 chars, so landmark addresses ("ร้านกาแฟหน้าปากซอย 2")
        # are NOT destroyed.
        _CO_RE = re.compile(r'(?:^|\s)(?:บจก|บริษัท|ห้างหุ้นส่วน|หจก|บ\.)\b', re.IGNORECASE)
        if receiver is not None and _CO_RE.search(text[:40]):
            _stripped = re.sub(r'^.*?(?=\d|(?:อาคาร|ตึก|คอนโด|ห้อง|ชั้น))', '', text, count=1, flags=re.DOTALL).strip()
            if _stripped and _stripped != text:
                text = _stripped
        # FIX [P-F4]: Strip trailing bracket/quote punctuation from receiver.
        if receiver:
            receiver = re.sub(r"[\s()\[\]{}<>\"\']+$", "", receiver).strip()
            if len(receiver) < 2:
                receiver = None

        # [FIX-P3] Pass D: Trailing personal name.
        # Some customers append their name at the very end of the message after
        # a complete address and phone number. By this point, phone/zipcode/geo
        # have already been extracted and stripped. If receiver is still None,
        # check whether the tail of the remaining text looks like a Thai name:
        # 1-3 Thai-character tokens, no digits, not a stop word, length ≤ 12 chars.
        # Guard: only fire when text starts with a digit (house-number pattern) —
        # meaning we have a real address and the trailing tokens are extra, not
        # the only content (avoids treating bare-name inputs as address+receiver).
        if receiver is None:
            remaining = re.sub(r"\s+", " ", text).strip()
            # [FIX-P3b] Fire Pass D even when text doesn't start with a digit.
            # Condo/building addresses ("หอพักป้าจุ๋ม ห้อง 305...") start with Thai
            # characters, not digits. We guard quality with token validation below.
            if remaining:
                tail_m = re.search(
                    r"(?:^|\s)((?:[^\s\d/]{2,12}\s*){1,2})\s*$",
                    remaining
                )
                if tail_m:
                    candidate = tail_m.group(1).strip()
                    tokens = candidate.split()
                    valid = (
                        len(tokens) >= 1 and
                        all(
                            re.search(r"[฀-๿]", t) and
                            t not in NAME_STOP_WORDS and
                            not any(t.startswith(sw) for sw in NAME_STOP_WORDS) and
                            not re.search(r"\d", t)
                            for t in tokens
                        )
                    )
                    if valid:
                        receiver = candidate
                        text = remaining[:tail_m.start(1)].rstrip()

        text = re.sub(r"\s+", " ", text).strip()
        address_detail = text.strip(" ,.-") or None

        return (
            receiver.strip() if receiver else None,
            address_detail.strip() if address_detail else None,
        )

    # ── Phase B ───────────────────────────────────────────────────────────────

    # FIX [FIX-P2]: Removed internal _normalise() call — text is already
    # normalised by the time this method is called from parse().
    def _apply_fuzzy_geo(self, normalised_text: str, result: ParseResult) -> ParseResult:
        fuzzy_rec, _, corrections, fuzzy_warns = self._fuzzy.fuzzy_lookup(
            normalised_text, zipcode_hint=result.zipcode, threshold=self._fuzzy_threshold
        )
        result.warnings.extend(fuzzy_warns)

        if fuzzy_rec:
            # Only copy sub_district when fuzzy actually matched a sub_district token.
            # If corrections only contain district/province matches, the sub_district
            # was inferred from the DB row — keep it None rather than guessing.
            sub_corrected = any(c.startswith('ตำบล') for c in corrections)
            if sub_corrected:
                result.sub_district = result.sub_district or fuzzy_rec.sub_district
            result.district     = result.district     or fuzzy_rec.district
            result.province     = result.province     or fuzzy_rec.province
            if not result.zipcode:
                result.zipcode  = fuzzy_rec.zipcode
            for c in corrections:
                result.warnings.append(f"[FuzzyCorrection] {c}")

        return result

    # ── Phase C ───────────────────────────────────────────────────────────────

    def _apply_ner(self, text: str, result: ParseResult) -> ParseResult:
        ner = ner_extract(text)
        if not ner.used_ner:
            return result

        if result.receiver is None and ner.receiver:
            result.receiver = ner.receiver
            result.warnings.append(f"[NERFallback] Receiver: '{ner.receiver}'")

        if result.province is None and ner.location_hint:
            geo_rec, _, geo_warns = self._geo.lookup(ner.location_hint, zipcode_hint=result.zipcode)
            result.warnings.extend(geo_warns)
            if geo_rec:
                result.sub_district = result.sub_district or geo_rec.sub_district
                result.district     = result.district     or geo_rec.district
                result.province     = result.province     or geo_rec.province
                result.zipcode      = result.zipcode      or geo_rec.zipcode
                result.warnings.append(f"[NERFallback] Geo from: '{ner.location_hint}'")
            elif RAPIDFUZZ_AVAILABLE:
                fuzzy_rec, _, corrections, _ = self._fuzzy.fuzzy_lookup(
                    ner.location_hint, zipcode_hint=result.zipcode
                )
                if fuzzy_rec:
                    result.province = result.province or fuzzy_rec.province
                    result.district = result.district or fuzzy_rec.district
                    for c in corrections:
                        result.warnings.append(f"[NER+Fuzzy] {c}")

        return result

    # ── Confidence & Validation ───────────────────────────────────────────────

    def _compute_confidence(self, result: ParseResult) -> ParseResult:
        filled = sum(1 for f in self._CONFIDENCE_FIELDS if getattr(result, f, None))
        result.confidence = round(filled / len(self._CONFIDENCE_FIELDS), 2)

        if result.confidence < 0.57:
            result.status = "Flagged for Review"
            msg = "Low confidence — flagged for human review"
            if msg not in result.warnings:
                result.warnings.append(msg)
        elif result.confidence < 0.86:
            result.status = "Success with Warnings"
        return result

    @staticmethod
    def _strict_validate(result: ParseResult) -> ParseResult:
        critical_missing = [
            f for f in ("province", "sub_district", "receiver")
            if not getattr(result, f, None)
        ]
        receiver_val  = result.receiver or ""
        receiver_bare = re.sub(r"[^฀-๿a-zA-Z]", "", receiver_val)
        receiver_garbage = bool(receiver_val) and len(receiver_bare) < 2

        fields_filled = sum(
            1 for f in ("receiver", "phone", "address_detail",
                        "sub_district", "district", "province", "zipcode")
            if getattr(result, f, None)
        )

        flag_reasons: List[str] = []
        # [FIX-P5]: Only 0–1 fields extracted means this is not an address at all
        # (e.g. "โทร 0812345678" → phone only; "คุณสมชาย" → receiver only).
        # Return "Not an Address" rather than "Flagged for Review" so the intent
        # classification is correct and the caller does not queue it for human review.
        if fields_filled <= 1:
            result.status     = "Not an Address"
            result.confidence = 0.0
            result.warnings.append(
                "[StrictValidation] Only 1 field extracted — not an address"
            )
            return result
        if critical_missing:
            flag_reasons.append(f"missing: {', '.join(critical_missing)}")
        if receiver_garbage:
            flag_reasons.append(f"receiver '{receiver_val}' is too short (garbage token)")
        if fields_filled < 3:
            flag_reasons.append(f"only {fields_filled}/7 fields extracted")

        if flag_reasons:
            result.status     = "Flagged for Review"
            result.confidence = min(result.confidence, 0.50)
            result.warnings.append("[StrictValidation] " + "; ".join(flag_reasons))

        return result

    @staticmethod
    def _elapsed(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 3)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _best_name_candidate(text: str) -> Optional[str]:
    tokens = [t for t in text.split() if re.search(r"[฀-๿]", t) and len(t) >= 2]
    return " ".join(tokens[:2]) if tokens else (text.strip() or None)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FLYWHEEL — HITL CORRECTION LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def log_correction(
    original_text:  str,
    parsed_result:  ParseResult,
    corrected_json: dict,
    db_connection:  object,
    corrected_by:   str = "admin",
    session_id:     Optional[str] = None,
    request_id:     Optional[str] = None,
    model_version:  str = "v7.0",
) -> CorrectionRecord:
    diff_fields = [
        k for k in ParseResult.__dataclass_fields__
        if k not in ("confidence", "processing_ms", "warnings", "tags", "status")
        and getattr(parsed_result, k, None) != corrected_json.get(k)
    ]
    geo_fields = {"sub_district", "district", "province", "zipcode"}
    diff_set   = set(diff_fields)
    # FIX [#22]: Old code required ALL four geo fields to differ for "geo_only".
    # A correction that fixes only "province" was misclassified as "full",
    # polluting the training signal. New logic: any non-empty subset of geo
    # fields with no non-geo changes → "geo_only".
    correction_type = (
        "geo_only"   if diff_set and diff_set <= geo_fields
        else "name_only" if diff_fields == ["receiver"]
        else "full"
    )
    record = CorrectionRecord(
        original_text=original_text,
        parsed_output=parsed_result.to_dict(),
        corrected_output=corrected_json,
        corrected_by=corrected_by,
        correction_type=correction_type,
        session_id=session_id,
        request_id=request_id,
        model_version=model_version,
    )
    try:
        if hasattr(db_connection, "save_correction"):
            db_connection.save_correction(record)
            logger.info("Correction logged: type=%s fields=%s", correction_type, diff_fields)
        else:
            logger.warning("db_connection has no save_correction() — correction not persisted.")
    except Exception as exc:
        # FIX [#5]: Attach the error to the record so api.py's submit_feedback
        # can detect it and return HTTP 500 instead of a silent HTTP 201.
        object.__setattr__(record, "_db_error", str(exc)) if hasattr(record, "__dataclass_fields__") \
            else setattr(record, "_db_error", str(exc))
        logger.error("Failed to log correction: %s", exc, exc_info=True)
    return record


# ══════════════════════════════════════════════════════════════════════════════
# OMNICHANNEL INPUT NORMALISER  (REST / Meta Messenger / LINE)
# ══════════════════════════════════════════════════════════════════════════════

from models import ChannelMessage  # noqa: E402


def normalise_webhook_payload(payload: dict) -> ChannelMessage:
    """Detect source channel from payload structure and extract address text."""
    # Meta Messenger
    if payload.get("object") == "page" and "entry" in payload:
        try:
            entry     = payload["entry"][0]
            messaging = entry["messaging"][0]
            msg_obj   = messaging.get("message", {})
            if "text" not in msg_obj:
                raise ValueError(
                    f"Meta Messenger message has no text field "
                    f"(type='{msg_obj.get('type', 'unknown')}')"
                )
            return ChannelMessage(
                text=msg_obj["text"].strip(),
                channel="facebook",
                customer_id=messaging.get("sender", {}).get("id"),
                message_id=msg_obj.get("mid"),
                page_id=entry.get("id"),
                raw=payload,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Malformed Meta Messenger payload: {exc}") from exc

    # LINE — iterate ALL events in the batch, return the first text message.
    # FIX [#21]: LINE delivers multiple events in one POST when a customer
    # sends rapid messages. The old code hard-coded events[0], silently
    # dropping every event after the first one in such batches.
    if "events" in payload:
        try:
            events = payload["events"]
            if not events:
                raise ValueError("LINE webhook received empty events list")
            for event in events:
                if event.get("type") != "message":
                    continue
                msg_obj = event.get("message", {})
                if msg_obj.get("type") != "text":
                    continue
                return ChannelMessage(
                    text=msg_obj["text"].strip(),
                    channel="line",
                    customer_id=event.get("source", {}).get("userId"),
                    raw=payload,
                )
            # No usable text event found anywhere in the batch
            first_type = events[0].get("type", "unknown")
            raise ValueError(
                f"LINE webhook contained no text message events "
                f"(first event type='{first_type}')"
            )
        except ValueError:
            raise
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed LINE payload: {exc}") from exc

    # REST (standard)
    text = payload.get("text") or payload.get("message") or payload.get("address") or ""
    if text:
        return ChannelMessage(
            text=str(text).strip(),
            channel="rest",
            customer_id=payload.get("customer_id") or payload.get("user_id"),
            raw=payload,
        )
    raise ValueError(
        "Cannot extract text. Expected 'text', Meta Messenger entry, or LINE event."
    )


def parse_from_webhook(
    payload: dict,
    parser: SmartAddressParser,
) -> Tuple[ParseResult, ChannelMessage]:
    """Convenience wrapper: normalise any webhook payload -> parse."""
    msg = normalise_webhook_payload(payload)
    return parser.parse(msg.text), msg


# ══════════════════════════════════════════════════════════════════════════════
# MOCK GEO DB  (for tests and local dev without the full CSV)
# ══════════════════════════════════════════════════════════════════════════════

_MOCK_CSV = """\
sub_district,district,province,zipcode
แสนสุข,เมืองชลบุรี,ชลบุรี,20130
บ้านสวน,เมืองชลบุรี,ชลบุรี,20000
หนองปลาไหล,บางละมุง,ชลบุรี,20150
หนองปรือ,บางละมุง,ชลบุรี,20150
ท่าแร้ง,บางเขน,กรุงเทพมหานคร,10220
บางเขน,บางเขน,กรุงเทพมหานคร,10220
ในเมือง,เมืองนครราชสีมา,นครราชสีมา,30000
โพธิ์กลาง,เมืองนครราชสีมา,นครราชสีมา,30000
ช้างเผือก,เมืองเชียงใหม่,เชียงใหม่,50300
ท่าศาลา,เมืองเชียงใหม่,เชียงใหม่,50000
สุเทพ,เมืองเชียงใหม่,เชียงใหม่,50200
หาดใหญ่,หาดใหญ่,สงขลา,90110
บ่อยาง,เมืองสงขลา,สงขลา,90000
บางนา,บางนา,กรุงเทพมหานคร,10260
ลาดยาว,จตุจักร,กรุงเทพมหานคร,10900
จอมพล,จตุจักร,กรุงเทพมหานคร,10900
จันทร์เกษม,จตุจักร,กรุงเทพมหานคร,10900
คลองหลวง,คลองหลวง,ปทุมธานี,12120
คลองสาม,คลองหลวง,ปทุมธานี,12120
ในเมือง,เมืองขอนแก่น,ขอนแก่น,40000
โคกสว่าง,เมืองขอนแก่น,ขอนแก่น,40000
ดอนช้าง,เมืองขอนแก่น,ขอนแก่น,40000
พระสิงห์,เมืองเชียงใหม่,เชียงใหม่,50200
ธาตุเชิงชุม,เมืองสกลนคร,สกลนคร,47000
ดงมะไฟ,เมืองสกลนคร,สกลนคร,47000
หนองไผ่ล้อม,เมืองนครราชสีมา,นครราชสีมา,30000
มะลวน,พุนพิน,สุราษฎร์ธานี,84130
นาเกลือ,พระสมุทรเจดีย์,สมุทรปราการ,10290
บางโฉลง,บางพลี,สมุทรปราการ,10540
บางพลีใหญ่,บางพลี,สมุทรปราการ,10540
ไร่สะท้อน,บ้านลาด,เพชรบุรี,76150
ท่าลาด,ชุมพวง,นครราชสีมา,30270
บางระกำ,บางระกำ,พิษณุโลก,65140
สีกัน,ดอนเมือง,กรุงเทพมหานคร,10210
ตลาด,เมืองมหาสารคาม,มหาสารคาม,44000
อนุสาวรีย์,บางเขน,กรุงเทพมหานคร,10220
ห้วยขวาง,ห้วยขวาง,กรุงเทพมหานคร,10310
คลองเตย,คลองเตย,กรุงเทพมหานคร,10110
สามเสนใน,พญาไท,กรุงเทพมหานคร,10400
หัวหมาก,บางกะปิ,กรุงเทพมหานคร,10240
ท่าทราย,เมืองนนทบุรี,นนทบุรี,11000
บางม่วง,บางใหญ่,นนทบุรี,11140
พระราชนิเวศน์,เมืองนนทบุรี,นนทบุรี,11000
"""


def build_mock_geo_db() -> GeoDatabase:
    """Build a test GeoDatabase from embedded mock CSV (covers all test cases)."""
    return GeoDatabase().load_csv_string(_MOCK_CSV)