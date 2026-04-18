#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThaiSmartAddress v7.0 — constants.py
All pre-compiled regexes, linguistic resources, and classification tables.

FIXES applied (original):
  [C1] Added มบ. -> หมู่บ้าน abbreviation
  [C2] Added กม.ม. -> กรุงเทพมหานคร abbreviation
  [C3] Added เลขที่อยู่ / เลขที่ to CONNECTOR_PATTERN
  [C4] Added hashtag (#) variants to CONNECTOR_PATTERN
  [C5] Extended Do_not_fold tag to catch เด็ดขาด suffix
  [C6] Lowered FUZZY_THRESHOLD 85->80
  [C7] Added expanded military rank forms to HONORIFIC_LIST

NEW FIXES applied (this version):
  [FIX-C8]  Removed "นะคะ", "คะ", "ครับ" from INTENT_NEG_KEYWORDS — these are
            polite Thai particles that appear in almost all address messages.
            With digit_density = 0% (no house number) they caused the Intent
            Shield to reject valid addresses as "Not an Address".
            Only genuine shopping-intent keywords are now in the set.
  [FIX-C9]  Added "ส่งฟรีไหม", "พร้อมส่ง", "โอนให้แล้ว" as high-signal
            shopping keywords to compensate for the narrowed particle set.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Tuple

THAI_BLOCK = r"[฀-๿]"
THAI_CHAR_RE = re.compile(THAI_BLOCK)

# ══════════════════════════════════════════════════════════════════════════════
# ABBREVIATION EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

_NLB = r"(?<![฀-๿A-Za-z])"
_NLA = r"(?![฀-๿A-Za-z])"

ABBREV_EXPAND: List[Tuple[str, str]] = [
    # 1. Military & police ranks (must be first to expand before geo tokens)
    (rf"{_NLB}จ\.ส\.อ\.?{_NLA}", "จ่าสิบเอก"),
    (rf"{_NLB}ส\.อ\.?{_NLA}",    "สิบเอก"),
    (rf"{_NLB}ร\.ต\.ต\.?{_NLA}", "ร้อยตำรวจตรี"),
    (rf"{_NLB}พล\.ต\.ต\.?{_NLA}","พลตำรวจตรี"),
    (rf"{_NLB}พ\.ต\.ท\.?{_NLA}", "พันตำรวจโท"),
    (rf"{_NLB}พ\.ต\.อ\.?{_NLA}", "พันตำรวจเอก"),
    (rf"{_NLB}ร\.อ\.?{_NLA}",    "ร้อยเอก"),

    # 2. Province aliases
    (rf"{_NLB}กรุงเทพฯ{_NLA}",   "กรุงเทพมหานคร"),
    (rf"{_NLB}กทม\.?{_NLA}",      "กรุงเทพมหานคร"),
    (rf"{_NLB}กม\.ม\.?{_NLA}",    "กรุงเทพมหานคร"),
    (rf"{_NLB}โคราช{_NLA}",       "นครราชสีมา"),
    (rf"{_NLB}แปดริ้ว{_NLA}",     "ฉะเชิงเทรา"),

    # 3. Geo-prefix abbreviations (after rank expansion to avoid collisions)
    (rf"{_NLB}มบ\.\s*(?={THAI_BLOCK})", "หมู่บ้าน "),
    (rf"{_NLB}จ\.\s*(?={THAI_BLOCK})(?!\s*{THAI_BLOCK}\.)",  "จังหวัด"),
    (rf"{_NLB}อ\.\s*(?={THAI_BLOCK})(?!\s*{THAI_BLOCK}\.)",  "อำเภอ"),
    (rf"{_NLB}ต\.\s*(?={THAI_BLOCK})(?!\s*{THAI_BLOCK}\.)",  "ตำบล"),
    (rf"{_NLB}ม\.\s*(?=\d)",            "หมู่ "),
    (rf"{_NLB}ซ\.\s*(?=[฀-๿A-Za-z\d])","ซอย"),
    (rf"{_NLB}ถ\.\s*(?=[฀-๿A-Za-z])",  "ถนน"),
]

_ABBREV_RE = re.compile(
    "|".join(f"(?P<a{i}>{pat})" for i, (pat, _) in enumerate(ABBREV_EXPAND)),
    re.IGNORECASE,
)
_ABBREV_REPL_MAP: Dict[str, str] = {
    f"a{i}": repl for i, (_, repl) in enumerate(ABBREV_EXPAND)
}


def abbrev_sub(m: re.Match) -> str:
    for name, repl in _ABBREV_REPL_MAP.items():
        if m.group(name) is not None:
            return repl
    return m.group(0)


def expand_abbreviations(text: str) -> str:
    return _ABBREV_RE.sub(abbrev_sub, text)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT JUNK & CONNECTOR PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

_CHAT_JUNK: List[str] = [
    "โอนแล้วค่ะ", "โอนแล้วครับ", "โอนเงินแล้ว",
    "cf ", "ยืนยันออเดอร์", "ยืนยัน",
    "สลิปแนบด้านบนค่ะ", "สลิปแนบ", "สลิป",
    "เอา 2 ชิ้นคะ", "เอา 1 ชิ้น",
    "ออเดอร์", "order ",
    "ที่อยู่ตามบัตร", "ตามบัตร ปชช", "ตามบัตรปชช", "ปชช:",
    "รับสายยากหน่อย", "รับสายยาก",
    "ไลน์มานะ", "ไลน์มา", "ไลน์ได้เลย",
    "รบกวนแอดมินหาให้หน่อย", "รบกวนแอดมินหา",
    "จำไม่ได้จ้า", "จำไม่ได้ค่ะ", "จำไม่ได้ครับ", "จำไม่ได้",
    "หาให้หน่อยนะ", "หาให้หน่อย",
]
CHAT_JUNK_PATTERN = re.compile(
    "|".join(re.escape(j) for j in sorted(_CHAT_JUNK, key=len, reverse=True)),
    re.IGNORECASE,
)

_CONNECTORS: List[str] = [
    # Hashtag-prefixed variants — longest first
    "#ที่อยู่จัดส่ง", "#ที่อยู่",
    # Delivery labels
    "รบกวนส่งที่", "ส่งที่ที่อยู่", "ส่งที่นี้ครับ", "ส่งที่นี้นะคะ",
    "จัดส่งที่", "ที่อยู่จัดส่ง",
    # House-number label prefix
    "เลขที่อยู่", "เลขที่",
    "ที่อยู่:", "ที่อยู่",
    "ส่งที่", "ส่งไปที่", "ส่งตามนี้นะคะ", "ส่งตามนี้ครับ",
    # FIX [C-F1]: Add variants with เลย/ด้วย suffixes and เอา prefix
    "ส่งตามนี้เลยครับ", "ส่งตามนี้เลยนะ", "ส่งตามนี้เลย",
    "เอาส่งมาที่", "เอาส่งที่", "ส่งมาที่", "ส่งมา",
    "To:", "to:", "ส่ง",
    "ที่",  # standalone particle after connector strip
    "ครับผม", "นะครับ", "นะคะ", "นะค่ะ", "นะค้า", "นะจ้า",
    "ครับ", "ค่ะ", "คะ", "จ้า", "ค้า",
    "เลยนะ", "เลย",
    "ให้ด้วย", "ด้วย", "ให้",
]
CONNECTOR_PATTERN = re.compile(
    "|".join(re.escape(c) for c in sorted(_CONNECTORS, key=len, reverse=True))
)

_PHONE_LABELS: List[str] = [
    "เบอร์โทรศัพท์", "เบอร์โทร", "โทรศัพท์", "โทร", "tel.", "tel", "ติดต่อ", "เบอร์",
    # [FIX-PL1] Strip "รหัสไปรษณีย์" label so the 5-digit code after it is captured cleanly
    "รหัสไปรษณีย์",
]
PHONE_LABEL_PATTERN = re.compile(
    "|".join(re.escape(lb) for lb in sorted(_PHONE_LABELS, key=len, reverse=True)),
    re.IGNORECASE,
)

# ══════════════════════════════════════════════════════════════════════════════
# HONORIFICS
# ══════════════════════════════════════════════════════════════════════════════

_HONORIFIC_LIST: List[str] = sorted([
    # Standard & abbreviations
    "นางสาว", "น.ส.", "นส.", "นส", "ด.ช.", "ด.ญ.", "ดช", "ดญ", "นาง", "นาย",
    # Academic & professional
    "ดร.", "ดร", "อาจารย์", "ผศ.", "รศ.", "ศ.", "อ.", "ครู", "หมอ",
    "นพ.", "พญ.", "ทพ.", "ภก.", "ภญ.", "ทนาย",
    "คุณ", "พี่", "น้อง", "ลุง", "น้า",
    "ป้า", "ยาย", "ตา", "ปู่", "อา",
    "เฮีย", "เจ๊", "ซ้อ", "เสี่ย", "แม่", "พ่อ", "หนู",
    # English-style
    "Miss", "Mrs.", "Mr.", "Ms.", "Khun",
    "K.", "k.", "K", "k", "P'", "p'", "N'", "n'",
    # Military & police (abbreviated forms)
    "พลฯ", "จ.ส.อ.", "จ.ส.อ", "ส.อ.", "ส.อ", "จ่า",
    "ร.ต.ต.", "ร.ต.", "ร.ท.", "ร.อ.",
    "พ.ต.ต.", "พ.ต.", "พ.ท.", "พ.อ.",
    "ด.ต.", "จ.ส.ต.", "ส.ต.อ.", "ส.ต.ท.", "ส.ต.ต.",
    "หมวด", "ผู้กอง", "สารวัตร", "ผู้การ",
    # Military & police (expanded forms — post-ABBREV_EXPAND)
    "จ่าสิบเอก", "สิบเอก", "ร้อยตำรวจตรี", "พลตำรวจตรี",
    "พันตำรวจโท", "พันตำรวจเอก", "ร้อยเอก",
], key=len, reverse=True)

HONORIFIC_SKIP_SET: FrozenSet[str] = frozenset(_HONORIFIC_LIST)

HONORIFIC_PATTERN = re.compile(
    r"(?:^|(?<=\s))("
    + "|".join(re.escape(h) for h in _HONORIFIC_LIST)
    + r")\s*(?P<n>[^\s\d]{2,})"
)

# ══════════════════════════════════════════════════════════════════════════════
# NAME STOP WORDS
# ══════════════════════════════════════════════════════════════════════════════

NAME_STOP_WORDS: FrozenSet[str] = frozenset([
    "ร้าน", "บริษัท", "บ.", "หจก", "ห้างหุ้นส่วน", "วิสาหกิจ", "โรงพยาบาล",
    "โรงแรม", "โรงเรียน", "มหาวิทยาลัย", "สถาบัน",
    "ซอย", "ถนน", "ถ.", "ซ.", "ตรอก", "หมู่บ้าน", "หมู่", "ม.", "ตึก",
    "อาคาร", "ชั้น", "ห้อง", "บ้าน", "ค่าย", "คอนโด", "แฟลต",
    "ตำบล", "แขวง", "อำเภอ", "เขต", "จังหวัด", "เลขที่อยู่", "เลขที่",
    # Staff/admin noise tokens
    "แอดมิน", "admin",
    # [FIX-SW1] Kilometer/road markers ("ช่างเอก ริมถนนสายเอเชีย กม 45" → stop at ริม, กม)
    "กม", "กม.", "กิโลเมตร", "ริม", "ขาเข้า", "ขาออก", "ฝั่ง", "ปาก",
    # Accommodation types
    "หอพัก", "หอ",
    # P.O. Box tokens — "ตู้ ปณ. 45" must not bleed into receiver
    "ตู้", "ปณ.", "ปณจ.", "ปณ",
])

# FIX [C-F3]: Explicit "ชื่อผู้รับ:" label that appears anywhere in message
RECEIVER_LABEL_RE = re.compile(
    r"(?<![ก-๙])(?:ชื่อผู้รับ|ชื่อ)\s*[:\-]?\s*([^\n\r:]{2,50}?)"
    r"(?=\s*\n|\s*$|\s*รหัส|\s*โทร|\s*เบอร์|\s*บ้านเลขที่|\s*หมู่|\s+บ้าน\s*\d|\s+\d)",
    re.IGNORECASE,
)

# FIX [C-F4]: "รหัสไปรษณีย์" followed by non-digit Thai complaint text
ZIPCODE_COMPLAINT_RE = re.compile(
    r"รหัสไปรษณีย์\s*(?=[^\d\s])[^\n\r]*"
)

# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC KEYWORD TAGS
# ══════════════════════════════════════════════════════════════════════════════

_KEYWORD_TAGS: List[Tuple[str, str]] = [
    (r"ด่วน(?:ๆ|มากๆ|มาก)?",                                 "Urgent"),
    (r"ระวัง(?:หน่อย(?:นะ(?:ครับ|คะ))?)?ของแตก(?:ง่าย)?",   "Fragile"),
    (r"(?:ฝาก(?:ไว้)?(?:ที่)?)?(?:ป้อมยาม|ป้อมหน้า|ป้อมประตู|รปภ\.?|security)",  "Drop_at_guard"),  # [FIX-TAG1]
    (r"(?:ห้าม|อย่า)พับ(?:\s*เด็ดขาด)?",                     "Do_not_fold"),
    (r"(?:ห้าม|อย่า)เปียก",                                   "Keep_dry"),
    (r"ส่ง(?:ก่อน|ด่วน)(?:เวลา|บ่าย)?",                     "Time_sensitive"),
]
TAG_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(pat), tag) for pat, tag in _KEYWORD_TAGS
]

# ══════════════════════════════════════════════════════════════════════════════
# PHONE & ZIPCODE PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

PHONE_RE = re.compile(
    r"(?<!\d)"
    r"("
    r"(?:\+66|66)[\s\-\./]?[6-9]\d[\s\-\./]?\d{3,4}[\s\-\./]?\d{3,4}"
    r"|(?:\+66|66)[\s\-\./]?[2-5]\d[\s\-\./]?\d{3}[\s\-\./]?\d{3,4}"
    r"|0[6-9]\d[\s\-\./]?\d{3,4}[\s\-\./]?\d{3,4}"
    r"|0[2-5]\d[\s\-\./]?\d{3}[\s\-\./]?\d{3,4}"
    r"|0[2-5][\s\-\./]?\d{3,4}[\s\-\./]?\d{3,4}"
    r")"
    r"(?!\d)"
)

ZIPCODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")

# ══════════════════════════════════════════════════════════════════════════════
# ADDRESS DETAIL ANCHOR
# ══════════════════════════════════════════════════════════════════════════════

ADDRESS_DETAIL_RE = re.compile(
    r"(?:"
    r"(?:^|(?<![฀-๿A-Za-z]))(?:หมู่บ้าน|ซอย|ซ\.|ถนน|ถ\.|ตรอก|คอนโด|ตู้\s*ปณ\.?|ร้าน[ก-๙]{2,})[^\n]{2,80}"
    r"|"
    r"\d+(?:/\d+)?"
    r"(?:\s*(?:หมู่\s*\d+|ม\.\s*\d+))?"
    r"(?:\s*[^\n]{0,80})?"
    r")"
)

# ══════════════════════════════════════════════════════════════════════════════
# INTENT SHIELD  (Phase -1)
#
# FIX [FIX-C8]: Removed "นะคะ", "คะ", "ครับ", "นะครับ" from this set.
#   Rationale: these are polite Thai sentence-ending particles that appear in
#   virtually every real address message. With digit_density = 0 (addresses
#   containing only district / province, no house number) the old code would
#   silently reject them as "Not an Address".
#   Only HIGH-SIGNAL shopping keywords that cannot plausibly appear in a
#   delivery address are kept in this set.
#
# FIX [FIX-C9]: Added "ส่งฟรีไหม", "พร้อมส่ง", "โอนให้แล้ว" as additional
#   high-signal shopping-intent keywords.
# ══════════════════════════════════════════════════════════════════════════════

INTENT_NEG_KEYWORDS: FrozenSet[str] = frozenset([
    # Pricing enquiries
    "ราคา", "เท่าไหร่", "เท่าไร", "ราคาเท่า", "กี่บาท",
    # Size / stock enquiries
    "ไซส์", "ไซต์", "size",
    # Discount / free shipping enquiries
    "ลดได้", "ลดให้", "ส่งฟรี", "ค่าส่ง", "จัดส่งฟรี", "ส่งฟรีไหม",
    # Stock / product enquiries
    "มีสต็อก", "มีของ", "หมดแล้ว", "สินค้า", "พร้อมส่ง",
    # Order-placement phrases (NOT delivery addresses)
    "สั่งได้", "สั่งเลย",
    # Payment confirmations WITHOUT an accompanying address
    "โอนแล้ว", "ยอดโอน", "โอนให้แล้ว",
    # [FIX-C10]: Short acknowledgements — "ขอบคุณค่ะ" has no address content.
    # Safe: valid addresses that open with ขอบคุณ always carry a 5-digit zipcode
    # or phone number, so they bypass the shield via the ZIPCODE_RE/PHONE_RE check.
    "ขอบคุณ",
    # [FIX-C11]: Product-variant enquiries — "มีสีอื่นไหมครับ" etc.
    "มีสีอื่น", "สีอื่น",
    # [FIX-C12]: Temporal/operational enquiries — "เปิดกี่โมง", "ปิดกี่โมง" etc.
    # Safe: no delivery address ever contains "กี่โมง".
    "กี่โมง",
    # NOTE: "นะคะ", "คะ", "ครับ", "นะครับ" intentionally REMOVED [FIX-C8]
])

INTENT_DIGIT_THRESHOLD: float = 0.04

# ══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCHING THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

FUZZY_THRESHOLD: int = 80
FUZZY_MAX_LEN_DIFF: int = 4
NER_FALLBACK_THRESHOLD: float = 0.60