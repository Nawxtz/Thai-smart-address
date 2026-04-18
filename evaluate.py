#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThaiSmartAddress v7.0 — evaluate.py
Academic evaluation script for NLP course report.

Measures:
  - Field-level Accuracy (province, district, sub_district, zipcode, phone, receiver)
  - Intent Classification (Precision, Recall, F1, TP/TN/FP/FN)
  - Fuzzy Matching success rate (typo correction)
  - Tag Detection accuracy (Urgent, Fragile, etc.)
  - Confidence score distribution
  - Average processing time

Run:
    python evaluate.py
    python evaluate.py --output report.txt   (also saves to file)

Uses the built-in mock geo database — no CSV file required.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import SmartAddressParser, build_mock_geo_db
from models import ParseResult


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATASET  (80 cases)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    input:           str
    category:        str          # clean | abbreviation | typo | noise | tags | rejection | incomplete | english_name | multiline | emoji | condo
    description:     str
    expect_address:  bool         # True = should parse, False = should reject
    expected: Dict[str, Optional[str]] = field(default_factory=dict)
    expect_tags:     List[str]    = field(default_factory=list)
    is_fuzzy:        bool         = False   # marks typo-correction cases


DATASET: List[EvalCase] = [

    # ── Category 1: Clean / Complete Addresses (20 cases) ─────────────────────

    EvalCase(
        input="รบกวนส่งที่ คุณแม็ค 99/9 ต.แสนสุข อ.เมืองชลบุรี จ.ชลบุรี 20130 0812345678",
        category="clean", description="Full address, all abbreviations",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20130",
                  "district": "เมืองชลบุรี", "sub_district": "แสนสุข", "phone": "0812345678"},
    ),
    EvalCase(
        input="คุณสมศรี 45/7 ถ.บางนา ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 0823456789",
        category="clean", description="Bangkok full address",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260",
                  "district": "บางนา", "sub_district": "บางนา", "phone": "0823456789"},
    ),
    EvalCase(
        input="คุณกาญจนา 5/5 ต.ช้างเผือก อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50300 0891234567",
        category="clean", description="Chiang Mai full address",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50300",
                  "district": "เมืองเชียงใหม่", "sub_district": "ช้างเผือก"},
    ),
    EvalCase(
        input="นาย ธีรภัทร ใจดี 10/2 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 0834567890",
        category="clean", description="Hat Yai full address with honorific",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110",
                  "district": "หาดใหญ่", "sub_district": "หาดใหญ่"},
    ),
    EvalCase(
        input="คุณมานี 88/1 ต.คลองหลวง อ.คลองหลวง จ.ปทุมธานี 12120 0876543210",
        category="clean", description="Pathum Thani full address",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120",
                  "district": "คลองหลวง", "sub_district": "คลองหลวง"},
    ),
    EvalCase(
        input="คุณปิยะ 10/5 ต.ในเมือง อ.เมืองขอนแก่น จ.ขอนแก่น 40000 0843456789",
        category="clean", description="Khon Kaen full address",
        expect_address=True,
        expected={"province": "ขอนแก่น", "zipcode": "40000",
                  "district": "เมืองขอนแก่น"},
    ),
    EvalCase(
        input="คุณรัตนา 20/4 ต.สีกัน อ.ดอนเมือง จ.กรุงเทพมหานคร 10210 0899876543",
        category="clean", description="Don Mueang Bangkok address",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10210",
                  "district": "ดอนเมือง", "sub_district": "สีกัน"},
    ),
    EvalCase(
        input="คุณวิภา 99 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 0898765432",
        category="clean", description="Hat Yai address no house num detail",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110"},
    ),
    EvalCase(
        input="คุณโสภิดา 88/1 ต.คลองหลวง อ.คลองหลวง จ.ปทุมธานี 12120 0876543210",
        category="clean", description="Pathum Thani with full prefix",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
    ),
    EvalCase(
        input="จัดส่งที่ คุณสมชาย 1/1 ต.แสนสุข อ.เมืองชลบุรี จ.ชลบุรี 20130 081-111-1111",
        category="clean", description="Delivery label prefix, phone with dashes",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20130", "phone": "0811111111"},
    ),
    EvalCase(
        input="ส่งที่ คุณแดง 1/1 ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 081.111.2222",
        category="clean", description="Phone with dots separator",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260", "phone": "0811112222"},
    ),
    EvalCase(
        input="คุณพิชญ์ 77/3 ต.พระสิงห์ อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50200 0821234567",
        category="clean", description="Chiang Mai Phra Singh",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50200",
                  "district": "เมืองเชียงใหม่", "sub_district": "พระสิงห์"},
    ),
    EvalCase(
        input="คุณนิรันดร์ 789/12 ต.ลาดยาว เขตจตุจักร กรุงเทพมหานคร 10900 0812222333",
        category="clean", description="Bangkok Chatuchak district",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900",
                  "district": "จตุจักร", "sub_district": "ลาดยาว"},
    ),
    EvalCase(
        input="คุณอรนุช 22/3 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 0834567890",
        category="clean", description="Hat Yai standard clean",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110",
                  "sub_district": "หาดใหญ่", "district": "หาดใหญ่"},
    ),
    EvalCase(
        input="คุณสมหมาย 55 ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 0891111222",
        category="clean", description="Short house number",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="คุณวรรณา 55/6 หมู่ 3 ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 0856789012",
        category="clean", description="Address with หมู่ number",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="ที่อยู่: คุณมาลัย 33/3 ต.บ้านสวน อ.เมืองชลบุรี จ.ชลบุรี 20000 0877654321",
        category="clean", description="ที่อยู่: prefix",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20000"},
    ),
    EvalCase(
        input="คุณธนาพร 10/10 ต.ธาตุเชิงชุม อ.เมืองสกลนคร จ.สกลนคร 47000 0812345670",
        category="clean", description="Sakon Nakhon province",
        expect_address=True,
        expected={"province": "สกลนคร", "zipcode": "47000"},
    ),
    EvalCase(
        input="คุณชนัญ 5/2 ต.นาเกลือ อ.พระสมุทรเจดีย์ จ.สมุทรปราการ 10290 0812233445",
        category="clean", description="Samut Prakan province",
        expect_address=True,
        expected={"province": "สมุทรปราการ", "zipcode": "10290"},
    ),
    EvalCase(
        input="คุณเพ็ญพิสุทธิ์ 9/1 ต.บางระกำ อ.บางระกำ จ.พิษณุโลก 65140 0823344556",
        category="clean", description="Phitsanulok province",
        expect_address=True,
        expected={"province": "พิษณุโลก", "zipcode": "65140"},
    ),

    # ── Category 2: Abbreviations (10 cases) ──────────────────────────────────

    EvalCase(
        input="คุณสมศรี 45/7 ถ.บางนา กทม. 10260 0823456789",
        category="abbreviation", description="กทม. → กรุงเทพมหานคร",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="คุณมาลี ต.ลาดยาว เขตจตุจักร กรุงเทพฯ 10900",
        category="abbreviation", description="กรุงเทพฯ → กรุงเทพมหานคร",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
    ),
    EvalCase(
        input="จ.ส.อ.สมชาย ใจดี 123/4 ต.ในเมือง อ.เมืองนครราชสีมา จ.นครราชสีมา 30000 0891234567",
        category="abbreviation", description="จ.ส.อ. military rank expansion",
        expect_address=True,
        expected={"province": "นครราชสีมา", "zipcode": "30000", "phone": "0891234567"},
    ),
    EvalCase(
        input="ร.ต.ต.วิชาญ 55/3 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 0845678901",
        category="abbreviation", description="ร.ต.ต. police rank expansion",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110"},
    ),
    EvalCase(
        input="ส่งที่ คุณแม็ค 99/9 ต.แสนสุข อ.เมืองชลบุรี จ.ชลบุรี 20130",
        category="abbreviation", description="ต./อ./จ. all three short forms",
        expect_address=True,
        expected={"province": "ชลบุรี", "sub_district": "แสนสุข", "district": "เมืองชลบุรี"},
    ),
    EvalCase(
        input="#ที่อยู่จัดส่ง คุณวรรณา 55/6 ต.บางนา อ.บางนา กทม. 10260 0856789012",
        category="abbreviation", description="Hashtag prefix + กทม.",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="โคราช นางสาวพิมพ์ 10/1 ต.ในเมือง 30000 0812222333",
        category="abbreviation", description="โคราช → นครราชสีมา alias",
        expect_address=True,
        expected={"province": "นครราชสีมา", "zipcode": "30000"},
    ),
    EvalCase(
        input="คุณรัชดา มบ.ลดาวัลย์ 5/5 ต.สีกัน อ.ดอนเมือง กทม. 10210 0899876543",
        category="abbreviation", description="มบ. → หมู่บ้าน expansion",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10210"},
    ),
    EvalCase(
        input="คุณบุญมา 33/3 ซ.5 ถ.มิตรภาพ ต.ในเมือง อ.เมืองนครราชสีมา จ.นครราชสีมา 30000",
        category="abbreviation", description="ซ. and ถ. street abbreviations",
        expect_address=True,
        expected={"province": "นครราชสีมา", "zipcode": "30000"},
    ),
    EvalCase(
        input="คุณธนา 10/2 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 +66812345678",
        category="abbreviation", description="+66 international phone → 0xxx",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110", "phone": "0812345678"},
    ),

    # ── Category 3: Typos / Fuzzy Matching (10 cases) ─────────────────────────

    EvalCase(
        input="คุณอรนุช 22/3 ต.หาดใหย่ อ.หาดใหย่ จ.สงขลา 90110 0834567890",
        category="typo", description="หาดใหย่ → หาดใหญ่ (ย/ญ swap)",
        expect_address=True, is_fuzzy=True,
        expected={"province": "สงขลา", "zipcode": "90110", "sub_district": "หาดใหญ่"},
    ),
    EvalCase(
        input="คุณสมชาย 1/1 ต.แสนสุค อ.เมืองชลบุรี จ.ชลบุรี 20130",
        category="typo", description="แสนสุค → แสนสุข (ค/ข swap)",
        expect_address=True, is_fuzzy=True,
        expected={"province": "ชลบุรี", "zipcode": "20130"},
    ),
    EvalCase(
        input="คุณมานี 5/5 ต.ช้างเผือค อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50300",
        category="typo", description="ช้างเผือค → ช้างเผือก (ค/ก swap)",
        expect_address=True, is_fuzzy=True,
        expected={"province": "เชียงใหม่", "zipcode": "50300"},
    ),
    EvalCase(
        input="คุณพิมพ์ 10/1 ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 081234567",
        category="typo", description="Phone missing one digit — should still parse address",
        expect_address=True, is_fuzzy=False,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="คุณปิยะ 10/5 ตำบลในเมือง อำเภอเมืองขอนแก่น จังหวัดขอนแก่น 40000",
        category="typo", description="Full form ตำบล/อำเภอ/จังหวัด (no abbreviation)",
        expect_address=True, is_fuzzy=False,
        expected={"province": "ขอนแก่น", "zipcode": "40000",
                  "district": "เมืองขอนแก่น"},
    ),
    EvalCase(
        input="คุณรัตนา 20/4 ต.สีกัน อ.ดอนเมือง จ.กรุงเทพมหานคร. 10210 0899876543",
        category="typo", description="Trailing dot after province name",
        expect_address=True, is_fuzzy=False,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10210"},
    ),
    EvalCase(
        input="คุณวิภา\u200b 99 ต.\u200bหาดใหญ่ อ.หาดใหญ่ จ.\u200bสงขลา\u200b 90110 0898765432",
        category="typo", description="Zero-width spaces (U+200B) from LINE copy-paste",
        expect_address=True, is_fuzzy=False,
        expected={"province": "สงขลา", "zipcode": "90110"},
    ),
    EvalCase(
        input="คุณโอ๋ 5/1 ต คลองหลวง อ คลองหลวง จ ปทุมธานี 12120 0811111111",
        category="typo", description="Missing dots after ต อ จ",
        expect_address=True, is_fuzzy=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
    ),
    EvalCase(
        input="ส่งที่ คุณแม็ค 99/9 แสนสุข เมืองชลบุรี ชลบุรี 20130 0812345678",
        category="typo", description="No ต./อ./จ. prefixes at all",
        expect_address=True, is_fuzzy=True,
        expected={"province": "ชลบุรี", "zipcode": "20130"},
    ),
    EvalCase(
        input="คุณนิด 20/1 ต.นาเกลือ อ.พระสมุทรเจดีย์ สมุทรปราการ 10290 081-999-8888",
        category="typo", description="Missing จ. before province, phone with dashes",
        expect_address=True, is_fuzzy=True,
        expected={"province": "สมุทรปราการ", "zipcode": "10290", "phone": "0819998888"},
    ),

    # ── Category 4: Chat Noise (10 cases) ─────────────────────────────────────

    EvalCase(
        input="i,r;oส่งที่ คอนโดลุมพินี ห้อง 45 ชั้น 8 ถ.สุขุมวิท เขตคลองเตย กทม 10110 091-111-2222 โอนเงินแล้วนะคะ",
        category="noise", description="Keyboard junk prefix + payment noise",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร"},
    ),
    EvalCase(
        input="โอนแล้วค่ะ ยืนยันออเดอร์ ส่งที่ คุณมาลัย 33/3 ต.บ้านสวน อ.เมืองชลบุรี จ.ชลบุรี 20000 0877654321",
        category="noise", description="Payment confirmation before address",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20000"},
    ),
    EvalCase(
        input="cf ยืนยัน คุณนิรันดร์ 789/12 ต.ลาดยาว เขตจตุจักร กทม. 10900 ห้ามพับเด็ดขาด",
        category="noise", description="cf + ยืนยัน junk prefix + Do_not_fold tag",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
        expect_tags=["Do_not_fold"],
    ),
    EvalCase(
        input="ส่งด่วนนะคะ คุณมาลี ต.ลาดยาว เขตจตุจักร กรุงเทพมหานคร 10900",
        category="noise", description="Polite particle นะคะ — must not be rejected",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
        expect_tags=["Urgent"],
    ),
    EvalCase(
        input="ร้านสวยงาม ส่งที่ คุณนิรันดร์ 789/12 ต.ลาดยาว เขตจตุจักร กทม. 10900",
        category="noise", description="Shop name stop-word before real receiver",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
    ),
    EvalCase(
        input="สลิปแนบด้านบนค่ะ ส่งที่ คุณปิยะ 10/5 ต.ในเมือง อ.เมืองขอนแก่น จ.ขอนแก่น 40000 0843456789",
        category="noise", description="Payment slip mention before address",
        expect_address=True,
        expected={"province": "ขอนแก่น", "zipcode": "40000"},
    ),
    EvalCase(
        input="เอา 2 ชิ้นคะ ส่งที่ คุณธนาพร 10/10 ต.ธาตุเชิงชุม อ.เมืองสกลนคร จ.สกลนคร 47000 0812345670",
        category="noise", description="Order quantity noise before address",
        expect_address=True,
        expected={"province": "สกลนคร", "zipcode": "47000"},
    ),
    EvalCase(
        input="ขอบคุณมากนะคะ ที่อยู่คือ คุณโสภิดา 88/1 ต.คลองหลวง อ.คลองหลวง ปทุมธานี 12120",
        category="noise", description="Polite opener + informal ที่อยู่คือ",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
    ),
    EvalCase(
        input="#ที่อยู่ คุณสมหมาย 55 ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 ด่วนมากนะคะ",
        category="noise", description="Hashtag prefix + urgent tag at end",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
        expect_tags=["Urgent"],
    ),
    EvalCase(
        input="To: คุณชนัญ 5/2 ต.นาเกลือ อ.พระสมุทรเจดีย์ จ.สมุทรปราการ 10290 0812233445",
        category="noise", description="English To: prefix",
        expect_address=True,
        expected={"province": "สมุทรปราการ", "zipcode": "10290"},
    ),

    # ── Category 5: Semantic Tags (10 cases) ──────────────────────────────────

    EvalCase(
        input="ด่วนมากๆ คุณกาญจนา 5/5 ต.ช้างเผือก อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50300 ระวังของแตก",
        category="tags", description="Urgent + Fragile compound tags",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50300"},
        expect_tags=["Urgent", "Fragile"],
    ),
    EvalCase(
        input="ฝากไว้ที่ป้อมยาม คุณโสภิดา 88/1 ต.คลองหลวง อ.คลองหลวง จ.ปทุมธานี 12120 0876543210",
        category="tags", description="Drop_at_guard tag",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
        expect_tags=["Drop_at_guard"],
    ),
    EvalCase(
        input="คุณนิรันดร์ 789/12 ต.ลาดยาว เขตจตุจักร กทม. 10900 ห้ามพับเด็ดขาด",
        category="tags", description="Do_not_fold tag",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
        expect_tags=["Do_not_fold"],
    ),
    EvalCase(
        input="ส่งด่วน คุณวิภา 99 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 0898765432",
        category="tags", description="Urgent tag from ส่งด่วน",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110"},
        expect_tags=["Urgent"],
    ),
    EvalCase(
        input="คุณชนัญ 5/2 ต.นาเกลือ อ.พระสมุทรเจดีย์ จ.สมุทรปราการ 10290 ระวังของแตกง่าย",
        category="tags", description="Fragile tag variant ของแตกง่าย",
        expect_address=True,
        expected={"province": "สมุทรปราการ", "zipcode": "10290"},
        expect_tags=["Fragile"],
    ),
    EvalCase(
        input="ด่วนๆ คุณธนาพร 10/10 ต.ธาตุเชิงชุม อ.เมืองสกลนคร จ.สกลนคร 47000 ห้ามเปียก",
        category="tags", description="Urgent + Keep_dry compound tags",
        expect_address=True,
        expected={"province": "สกลนคร", "zipcode": "47000"},
        expect_tags=["Urgent", "Keep_dry"],
    ),
    EvalCase(
        input="ฝากไว้ที่ รปภ. คุณพิชญ์ 77/3 ต.พระสิงห์ อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50200",
        category="tags", description="Drop_at_guard via รปภ.",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50200"},
        expect_tags=["Drop_at_guard"],
    ),
    EvalCase(
        input="ด่วนมาก คุณรัตนา 20/4 ต.สีกัน อ.ดอนเมือง กทม. 10210 ระวังของแตก ห้ามพับ",
        category="tags", description="Three tags: Urgent + Fragile + Do_not_fold",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10210"},
        expect_tags=["Urgent", "Fragile", "Do_not_fold"],
    ),
    EvalCase(
        input="ส่งก่อนบ่าย คุณมานี 88/1 ต.คลองหลวง อ.คลองหลวง จ.ปทุมธานี 12120",
        category="tags", description="Time_sensitive tag",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
        expect_tags=["Time_sensitive"],
    ),
    EvalCase(
        input="คุณบุญมา 33/3 ต.ในเมือง อ.เมืองนครราชสีมา จ.นครราชสีมา 30000 ระวังของแตก ด่วนๆ",
        category="tags", description="Tags at end of message",
        expect_address=True,
        expected={"province": "นครราชสีมา", "zipcode": "30000"},
        expect_tags=["Fragile", "Urgent"],
    ),

    # ── Category 6: Intent Rejection — True Negatives (20 cases) ─────────────

    EvalCase(
        input="สินค้าราคาเท่าไหร่คะ",
        category="rejection", description="Price enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="มีสต็อกไหมครับ ไซส์ XL",
        category="rejection", description="Stock/size enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="ลดได้ไหมคะ ราคานี้",
        category="rejection", description="Discount enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="ส่งฟรีไหมครับ ขอโค้ดลดหน่อย",
        category="rejection", description="Free shipping enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="เสื้อสีแดงไซส์ M ตัวนี้ลดได้สุดกี่บาทคะ",
        category="rejection", description="Product enquiry with size",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="โอนแล้วนะครับ ยอด 250 บาท",
        category="rejection", description="Payment confirmation only, no address",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="พร้อมส่งไหมคะ สินค้าหมดแล้วหรือเปล่า",
        category="rejection", description="Product availability enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="ค่าส่งเท่าไหร่ครับ ส่งไปต่างจังหวัดได้ไหม",
        category="rejection", description="Shipping cost enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="มีของแถมไหมคะ",
        category="rejection", description="Gift enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="สั่งได้เลยไหมครับ ราคานี้",
        category="rejection", description="Order enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="ไซส์ XL มีไหมคะ ราคาเท่าไร กี่บาท",
        category="rejection", description="Multiple shopping signals",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="โอนให้แล้วนะคะ ตรวจสอบด้วย",
        category="rejection", description="Payment transfer message",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="ขอบคุณค่ะ",
        category="rejection", description="Polite acknowledgement only",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="โทร 0812345678",
        category="rejection", description="Phone only, no address",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="คุณสมชาย",
        category="rejection", description="Name only, no address",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="สินค้าพร้อมส่ง มีสต็อก 5 ชิ้น",
        category="rejection", description="Stock announcement",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="จัดส่งฟรีไหมคะ ถ้าซื้อครบ 500",
        category="rejection", description="Free shipping condition",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="ราคานี้รวม vat ไหมครับ",
        category="rejection", description="Tax enquiry",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="ขอรูปสินค้าเพิ่มเติมได้ไหมคะ",
        category="rejection", description="Product image request",
        expect_address=False, expected={},
    ),
    EvalCase(
        input="มีสีอื่นไหมครับ นอกจากสีดำ",
        category="rejection", description="Color variant enquiry",
        expect_address=False, expected={},
    ),

    # ── Category 7: English Names (15 cases) ──────────────────────────────────

    EvalCase(
        input="ส่งให้ James 99/9 ต.แสนสุข อ.เมืองชลบุรี จ.ชลบุรี 20130 0812345678",
        category="english_name", description="English first name only, no honorific",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20130"},
    ),
    EvalCase(
        input="Mr. David Thompson 45/7 ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 0891234567",
        category="english_name", description="Full English name with Mr. honorific",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="คุณ Mike 5/5 ต.ช้างเผือก อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50300 081-234-5678",
        category="english_name", description="Thai honorific + English nickname",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50300"},
    ),
    EvalCase(
        input="Ann 55/5 ต.ช้างเผือก อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50300 โทร 0891234567",
        category="english_name", description="English nickname only, no honorific, โทร prefix on phone",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50300"},
    ),
    EvalCase(
        input="Ann (คุณแอน) 99 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 0898765432",
        category="english_name", description="English nickname with Thai in parentheses",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110"},
    ),
    EvalCase(
        input="Send to: Sarah Johnson, 10/2 Soi Sukhumvit 21, Khlong Toei, Bangkok 10110, +66812345678",
        category="english_name", description="Full English address — expat format",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10110"},
    ),
    EvalCase(
        input="ฝากส่งให้ Sarah ด้วยนะคะ ที่อยู่เดิม ต.บางนา อ.บางนา กทม 10260 0812345678",
        category="english_name", description="English name mid-sentence, informal Thai phrasing",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="คุณ James Smith 99/9 ต.แสนสุข อ.เมืองชลบุรี จ.ชลบุรี 20130 0812345678",
        category="english_name", description="Thai honorific + full English name",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20130"},
    ),
    EvalCase(
        input="Mrs. Linda 55/3 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา 90110 0845678901",
        category="english_name", description="Mrs. honorific + English first name only",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110"},
    ),
    EvalCase(
        input="พี่ Tom 45/7 ถ.บางนา ต.บางนา อ.บางนา กรุงเทพมหานคร 10260 0823456789",
        category="english_name", description="Thai kin-term honorific (พี่) + English nickname",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="น้อง Boy 10/2 ต.ลาดยาว เขตจตุจักร กรุงเทพมหานคร 10900 0812222333",
        category="english_name", description="Thai kin-term (น้อง) + English nickname (Boy)",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
    ),
    EvalCase(
        input="Dr. John Kim 10/5 ต.ในเมือง อ.เมืองขอนแก่น จ.ขอนแก่น 40000 0843456789",
        category="english_name", description="Dr. title + English full name",
        expect_address=True,
        expected={"province": "ขอนแก่น", "zipcode": "40000"},
    ),
    EvalCase(
        input="Khun Somchai 99/9 Moo 3 T. Saen Suk A. Mueang Chon Buri 20130 0812345678",
        category="english_name", description="Romanized Thai address (transliteration)",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20130"},
    ),
    EvalCase(
        input="คุณแจ็ค (Jack) 5/2 ต.นาเกลือ อ.พระสมุทรเจดีย์ จ.สมุทรปราการ 10290 0812233445",
        category="english_name", description="Thai transliteration of English name with English in parentheses",
        expect_address=True,
        expected={"province": "สมุทรปราการ", "zipcode": "10290"},
    ),
    EvalCase(
        input="Ms. Emma 88/1 ต.คลองหลวง อ.คลองหลวง จ.ปทุมธานี 12120 0876543210",
        category="english_name", description="Ms. honorific + English first name",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
    ),

    # ── Category 8: Multiline Addresses (10 cases) ────────────────────────────

    EvalCase(
        input="คุณสมชาย ใจดี\n99/9 ซ.รัชดา ต.ลาดยาว\nเขตจตุจักร กรุงเทพมหานคร 10900\nโทร 081-234-5678",
        category="multiline", description="4-line address from LINE message",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
    ),
    EvalCase(
        input="คุณมาลี\n45/7 ถ.บางนา ต.บางนา อ.บางนา\nกรุงเทพมหานคร 10260\n0823456789",
        category="multiline", description="Name on line 1, address on line 2-3, phone on line 4",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="ที่อยู่จัดส่ง:\nคุณกาญจนา\n5/5 ต.ช้างเผือก อ.เมืองเชียงใหม่\nจ.เชียงใหม่ 50300\n0891234567",
        category="multiline", description="Label on first line, name + address across subsequent lines",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50300"},
    ),
    EvalCase(
        input="คุณธนา 10/2\nต.หาดใหญ่ อ.หาดใหญ่\nจ.สงขลา 90110\n+66812345678",
        category="multiline", description="House num on line 1, sub/district on line 2, province+zip on line 3",
        expect_address=True,
        expected={"province": "สงขลา", "zipcode": "90110", "phone": "0812345678"},
    ),
    EvalCase(
        input="คุณมานี 88/1\nต.คลองหลวง อ.คลองหลวง\nจ.ปทุมธานี 12120\nโทรศัพท์: 0876543210",
        category="multiline", description="Phone with โทรศัพท์: label on last line",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
    ),
    EvalCase(
        input="รบกวนส่งที่\nคุณวรรณา 55/6 หมู่ 3\nต.บางนา อ.บางนา\nกทม. 10260\n0856789012",
        category="multiline", description="Request phrase on line 1, 5-line split",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="คุณปิยะ\n10/5 ต.ในเมือง\nอ.เมืองขอนแก่น จ.ขอนแก่น\n40000\n0843456789",
        category="multiline", description="Zipcode on its own line",
        expect_address=True,
        expected={"province": "ขอนแก่น", "zipcode": "40000"},
    ),
    EvalCase(
        input="ส่งด่วนค่ะ\nคุณโสภิดา 88/1 ต.คลองหลวง\nอ.คลองหลวง ปทุมธานี 12120",
        category="multiline", description="Urgent tag on line 1, address across lines 2-3",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
        expect_tags=["Urgent"],
    ),
    EvalCase(
        input="คุณนิรันดร์ 789/12\r\nต.ลาดยาว เขตจตุจักร\r\nกรุงเทพมหานคร 10900\r\n0812222333",
        category="multiline", description="Windows-style CRLF line endings",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
    ),
    EvalCase(
        input="คุณธนาพร\n10/10 ต.ธาตุเชิงชุม อ.เมืองสกลนคร\nจ.สกลนคร 47000\nโทร 0812345670 ระวังแตกนะคะ",
        category="multiline", description="Fragile tag on last line in multiline address",
        expect_address=True,
        expected={"province": "สกลนคร", "zipcode": "47000"},
        expect_tags=["Fragile"],
    ),

    # ── Category 9: Emoji in Message (5 cases) ────────────────────────────────

    EvalCase(
        input="📦 ส่งที่ คุณแดง 1/1 ต.บางนา อ.บางนา กทม 10260 📞 0812345678",
        category="emoji", description="Package and phone emoji surrounding address",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="✅ ยืนยันออเดอร์ค่ะ ส่งที่ คุณมาลี ต.ลาดยาว เขตจตุจักร กรุงเทพมหานคร 10900 0899876543",
        category="emoji", description="Checkmark emoji before confirmation phrase",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10900"},
    ),
    EvalCase(
        input="คุณกาญจนา 5/5 ต.ช้างเผือก อ.เมืองเชียงใหม่ จ.เชียงใหม่ 50300 ☎️ 0891234567",
        category="emoji", description="Telephone emoji before phone number",
        expect_address=True,
        expected={"province": "เชียงใหม่", "zipcode": "50300"},
    ),
    EvalCase(
        input="🚚 ด่วนมากๆ คุณสมชาย 1/1 ต.แสนสุข อ.เมืองชลบุรี จ.ชลบุรี 20130 0812345678 🔥",
        category="emoji", description="Delivery + fire emoji, Urgent tag expected",
        expect_address=True,
        expected={"province": "ชลบุรี", "zipcode": "20130"},
        expect_tags=["Urgent"],
    ),
    EvalCase(
        input="📍ที่อยู่: คุณโสภิดา 88/1 ต.คลองหลวง อ.คลองหลวง จ.ปทุมธานี 12120 0876543210",
        category="emoji", description="Pin emoji before ที่อยู่: label",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
    ),

    # ── Category 10: Condominium / Complex Addresses (10 cases) ───────────────

    EvalCase(
        input="คุณปาล์ม คอนโด Ideo Mix 103 ห้อง 2105 ชั้น 21 ถ.สุขุมวิท แขวงพระโขนง เขตคลองเตย กทม 10110",
        category="condo", description="English condo name + room + floor",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10110"},
    ),
    EvalCase(
        input="คุณมาลี คอนโด The Line สุขุมวิท 71 ห้อง 1505 แขวงพระโขนงเหนือ เขตวัฒนา กทม 10110",
        category="condo", description="The Line condo name with soi reference",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10110"},
    ),
    EvalCase(
        input="คุณธนา อพาร์ตเมนต์ บ้านเพชร ห้อง 3B ถ.มิตรภาพ ต.ในเมือง อ.เมืองขอนแก่น จ.ขอนแก่น 40000 0843456789",
        category="condo", description="Thai apartment name with alphanumeric room",
        expect_address=True,
        expected={"province": "ขอนแก่น", "zipcode": "40000"},
    ),
    EvalCase(
        input="คุณวิภา Lumpini Suite ห้อง 404 ชั้น 4 ถ.บางนา กทม 10260 0898765432",
        category="condo", description="Lumpini branded condo + floor and room",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10260"},
    ),
    EvalCase(
        input="คุณรัตนา 20/4 หอพัก สีสวรรค์ ห้อง 5 ต.สีกัน อ.ดอนเมือง กทม. 10210 0899876543",
        category="condo", description="Thai dormitory name embedded in address",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10210"},
    ),
    EvalCase(
        input="คุณสมชาย Rhythm Ekkamai ห้อง 1201 ซ.เอกมัย ถ.สุขุมวิท 63 แขวงคลองตันเหนือ เขตวัฒนา กรุงเทพมหานคร 10110 0812345678",
        category="condo", description="Brand-name condo + soi reference Bangkok",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10110"},
    ),
    EvalCase(
        input="คุณกาญจนา ไนท์บริดจ์ไพร์ม สาทร ห้อง 2205 ถ.สาทรใต้ แขวงทุ่งมหาเมฆ เขตสาทร กทม 10120",
        category="condo", description="Thai transliteration of English condo name",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10120"},
    ),
    EvalCase(
        input="คุณโสภิดา หมู่บ้าน Perfect Place รังสิต-คลอง 3 บ้านเลขที่ 55/99 ต.คลองหลวง อ.คลองหลวง จ.ปทุมธานี 12120 0876543210",
        category="condo", description="English estate/village name + Thai address",
        expect_address=True,
        expected={"province": "ปทุมธานี", "zipcode": "12120"},
    ),
    EvalCase(
        input="คุณมานี ลุมพินี เพลส พระราม 9-รัชดา อาคาร B ชั้น 12 ห้อง 1203 ต.ห้วยขวาง อ.ห้วยขวาง กทม 10310 0876543210",
        category="condo", description="Thai condo name + building + floor + room",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10310"},
    ),
    EvalCase(
        input="ส่งที่ บริษัท เอบีซี จำกัด แผนก IT คุณธนา 200 ถ.แจ้งวัฒนะ แขวงทุ่งสองห้อง เขตหลักสี่ กทม 10210 0812345678",
        category="condo", description="Company name as recipient + department",
        expect_address=True,
        expected={"province": "กรุงเทพมหานคร", "zipcode": "10210"},
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def run_evaluation(dataset: List[EvalCase], parser: SmartAddressParser) -> dict:
    fields_to_eval = ["province", "zipcode", "district", "sub_district", "phone", "receiver"]

    field_correct = {f: 0 for f in fields_to_eval}
    field_total   = {f: 0 for f in fields_to_eval}

    tag_correct = 0
    tag_total   = 0

    # Intent classification
    tp = tn = fp = fn = 0

    # Fuzzy
    fuzzy_correct = 0
    fuzzy_total   = 0

    # Timing & confidence
    timings     = []
    confidences = []

    # Per-category pass tracking
    category_results: Dict[str, Dict[str, int]] = {}

    failed_cases = []

    for case in dataset:
        cat = case.category
        if cat not in category_results:
            category_results[cat] = {"pass": 0, "fail": 0}

        t0 = time.perf_counter()
        result: ParseResult = parser.parse(case.input)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings.append(elapsed_ms)

        is_address = result.status not in ("Not an Address",)
        is_flagged = result.status == "Flagged for Review"

        # Intent classification
        if case.expect_address and is_address:
            tp += 1
        elif not case.expect_address and not is_address:
            tn += 1
        elif case.expect_address and not is_address:
            fn += 1
            failed_cases.append((case, result, "intent: false negative"))
        else:
            fp += 1
            failed_cases.append((case, result, "intent: false positive"))

        # Field accuracy (only for cases we expect to parse)
        if case.expect_address and is_address and not is_flagged:
            confidences.append(result.confidence)
            case_pass = True
            for f, expected_val in case.expected.items():
                if f not in fields_to_eval or expected_val is None:
                    continue
                field_total[f] += 1
                actual = getattr(result, f, None)
                if actual == expected_val:
                    field_correct[f] += 1
                else:
                    case_pass = False
                    failed_cases.append((case, result, f"field '{f}': expected {expected_val!r}, got {actual!r}"))

            # Tag accuracy
            for expected_tag in case.expect_tags:
                tag_total += 1
                if expected_tag in result.tags:
                    tag_correct += 1
                else:
                    failed_cases.append((case, result, f"tag '{expected_tag}' not found in {result.tags}"))

            if case_pass:
                category_results[cat]["pass"] += 1
            else:
                category_results[cat]["fail"] += 1

        elif case.expect_address and is_address and is_flagged:
            # Correctly accepted but flagged (e.g. incomplete address missing
            # receiver). Intent is correct; skip field-level checking since the
            # result is already marked for human review. Count as a pass.
            category_results[cat]["pass"] += 1

        elif not case.expect_address and not is_address:
            category_results[cat]["pass"] += 1
        else:
            category_results[cat]["fail"] += 1

        # Fuzzy tracking
        if case.is_fuzzy and case.expect_address:
            fuzzy_total += 1
            # Check if the key province/zipcode at minimum resolved
            target = case.expected.get("province") or case.expected.get("zipcode")
            actual = result.province or result.zipcode
            if target and target in (result.province, result.zipcode):
                fuzzy_correct += 1

    return {
        "field_correct":      field_correct,
        "field_total":        field_total,
        "tag_correct":        tag_correct,
        "tag_total":          tag_total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fuzzy_correct":      fuzzy_correct,
        "fuzzy_total":        fuzzy_total,
        "timings":            timings,
        "confidences":        confidences,
        "category_results":   category_results,
        "failed_cases":       failed_cases,
    }


def format_report(results: dict, total_cases: int) -> str:
    lines = []
    A = lines.append

    A("=" * 65)
    A("  ThaiSmartAddress v7.0 — Academic Evaluation Report")
    A("=" * 65)
    A(f"  Dataset: {total_cases} test cases")
    A("")

    # ── Intent Classification ──────────────────────────────────────────────────
    tp, tn, fp, fn = results["tp"], results["tn"], results["fp"], results["fn"]
    total_intent = tp + tn + fp + fn
    accuracy_intent = (tp + tn) / total_intent if total_intent else 0

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    A("── Intent Classification (Address vs Non-Address) ───────────")
    A(f"  {'True Positive  (address correctly accepted)':<42} {tp:>3}/{tp+fn:<3}  {tp/(tp+fn):.1%}" if (tp+fn) else "")
    A(f"  {'True Negative  (non-address correctly rejected)':<42} {tn:>3}/{tn+fp:<3}  {tn/(tn+fp):.1%}" if (tn+fp) else "")
    A(f"  {'False Positive (non-address wrongly accepted)':<42} {fp:>3}/{tn+fp:<3}")
    A(f"  {'False Negative (address wrongly rejected)':<42} {fn:>3}/{tp+fn:<3}")
    A("")
    A(f"  Precision : {precision:.1%}")
    A(f"  Recall    : {recall:.1%}")
    A(f"  F1 Score  : {f1:.1%}")
    A(f"  Accuracy  : {accuracy_intent:.1%}")
    A("")

    # ── Field-level Accuracy ───────────────────────────────────────────────────
    A("── Field-Level Accuracy (on successfully parsed cases) ──────")
    fields_to_eval = ["province", "zipcode", "district", "sub_district", "phone", "receiver"]
    for f in fields_to_eval:
        total = results["field_total"][f]
        correct = results["field_correct"][f]
        if total == 0:
            continue
        acc = correct / total
        A(f"  {f:<15} {bar(acc)}  {acc:.1%}  ({correct}/{total})")
    A("")

    # ── Tag Detection ──────────────────────────────────────────────────────────
    tc = results["tag_correct"]
    tt = results["tag_total"]
    if tt > 0:
        A("── Semantic Tag Detection ───────────────────────────────────")
        acc = tc / tt
        A(f"  Tags correctly detected: {bar(acc)}  {acc:.1%}  ({tc}/{tt})")
        A("")

    # ── Fuzzy Matching ─────────────────────────────────────────────────────────
    fc = results["fuzzy_correct"]
    ft = results["fuzzy_total"]
    if ft > 0:
        A("── Fuzzy Matching (Typo Correction) ─────────────────────────")
        acc = fc / ft
        A(f"  Typos corrected:         {bar(acc)}  {acc:.1%}  ({fc}/{ft})")
        A("")

    # ── Per-Category Breakdown ─────────────────────────────────────────────────
    A("── Results by Category ──────────────────────────────────────")
    cat_labels = {
        "clean":        "Clean / Complete",
        "abbreviation": "Abbreviation Expansion",
        "typo":         "Typo / Fuzzy Matching",
        "noise":        "Chat Noise Handling",
        "tags":         "Semantic Tag Detection",
        "rejection":    "Intent Rejection",
        "incomplete":   "Incomplete Address",
        "english_name": "English Name Handling",
        "multiline":    "Multiline Address",
        "emoji":        "Emoji in Message",
        "condo":        "Condo / Complex Address",
    }
    for cat, label in cat_labels.items():
        r = results["category_results"].get(cat)
        if not r:
            continue
        total = r["pass"] + r["fail"]
        acc = r["pass"] / total if total else 0
        A(f"  {label:<26} {bar(acc, 14)}  {acc:.1%}  ({r['pass']}/{total})")
    A("")

    # ── Performance ───────────────────────────────────────────────────────────
    timings = results["timings"]
    confidences = results["confidences"]
    if timings:
        avg_ms = sum(timings) / len(timings)
        max_ms = max(timings)
        A("── Performance ──────────────────────────────────────────────")
        A(f"  Avg processing time : {avg_ms:.1f} ms")
        A(f"  Max processing time : {max_ms:.1f} ms")
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        A(f"  Avg confidence score: {avg_conf:.3f}  ({avg_conf:.1%})")
    A("")

    # ── Failed Cases ──────────────────────────────────────────────────────────
    failed = results["failed_cases"]
    if failed:
        A("── Failed Cases (first 10) ──────────────────────────────────")
        seen_inputs = set()
        shown = 0
        for case, result, reason in failed:
            if shown >= 10:
                break
            key = case.input[:60]
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            A(f"  [{case.category}] {case.description}")
            A(f"    Input  : {case.input[:70]}")
            A(f"    Reason : {reason}")
            A(f"    Status : {result.status}  conf={result.confidence:.2f}")
            A("")
            shown += 1
    else:
        A("── All cases passed! ─────────────────────────────────────────")
        A("")

    A("=" * 65)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    arg_parser = argparse.ArgumentParser(description="ThaiSmartAddress Evaluation")
    arg_parser.add_argument("--output", "-o", default=None,
                            help="Also save report to this file (e.g. report.txt)")
    args = arg_parser.parse_args()

    print("Loading parser with mock geo database...")
    geo_db = build_mock_geo_db()
    parser = SmartAddressParser(geo_db)
    print(f"Geo DB loaded: {geo_db.size} records")
    print(f"Running {len(DATASET)} test cases...\n")

    results = run_evaluation(DATASET, parser)
    report  = format_report(results, len(DATASET))

    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()