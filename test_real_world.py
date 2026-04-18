import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = ""  

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["X-API-Key"] = API_KEY

# Define all the test cases
TEST_CASES = [
    {
        "name": "7. The Missing Province (Relies on Zipcode/District)",
        "input": "จัดส่ง คุณจิรายุ 088-777-6655 45/6 ซอยอารีย์สัมพันธ์ แขวงสามเสนใน เขตพญาไท 10400",
        # Parser Challenge: "กทม" or "กรุงเทพ" is completely missing. 
        # The parser must infer the province from "พญาไท" or "10400".
        "expected": {
            "receiver": "คุณจิรายุ",
            "phone": "088-777-6655",
            "address": "45/6 ซอยอารีย์สัมพันธ์",
            "sub_district": "สามเสนใน",
            "district": "พญาไท",
            "province": "กรุงเทพมหานคร", 
            "zipcode": "10400"
        }
    },
    {
        "name": "8. The Typo Disaster",
        "input": "น้องเมย์ 0900001111 88 ม.2 ตําบนบางพลีใหญ่ อําเพอบางพลี จังหว้ดสมุทรปราการ 10540",
        # Parser Challenge: Misspelled keywords like "ตําบน" (Tambon), "อําเพอ" (Amphoe), and "จังหว้ด" (Changwat).
        # A good parser uses fuzzy matching or ignores the prefixes entirely based on geography.
        "expected": {
            "receiver": "น้องเมย์",
            "phone": "0900001111",
            "address": "88 ม.2",
            "sub_district": "บางพลีใหญ่",
            "district": "บางพลี",
            "province": "สมุทรปราการ",
            "zipcode": "10540"
        }
    },
    {
        "name": "9. Multiple Phone Numbers & Chat Garbage",
        "input": "ฝากไว้ที่ป้อมยามนะคะ น.ส. สุดสวย งามตา 101/1 ถ.รามคำแหง แขวงหัวหมาก เขตบางกะปิ กทม. 10240 เบอร์หลัก 081-111-1111 เบอร์สำรอง 082-222-2222 โทรมาด่วนๆ",
        # Parser Challenge: Extracting the first/primary phone number while ignoring the second one, 
        # and stripping out conversational text ("ฝากไว้ที่ป้อมยามนะคะ", "โทรมาด่วนๆ").
        "expected": {
            "receiver": "น.ส สุดสวย งามตา",
            "phone": "081-111-1111", # Or handle both if your DB supports it
            "address": "101/1 ถ.รามคำแหง",
            "sub_district": "หัวหมาก",
            "district": "บางกะปิ",
            "province": "กรุงเทพมหานคร",
            "zipcode": "10240"
        }
    },
    {
        "name": "10. The P.O. Box (ตู้ ปณ.)",
        "input": "คุณนพดล 0834445555 ตู้ ปณ. 45 ปณจ. จตุจักร กทม 10900",
        # Parser Challenge: Recognizing P.O. Boxes instead of standard street/house numbers.
        "expected": {
            "receiver": "คุณนพดล",
            "phone": "0834445555",
            "address": "ตู้ ปณ. 45 ปณจ.",
            "sub_district": "จตุจักร", # Usually the post office location matches the district
            "district": "จตุจักร",
            "province": "กรุงเทพมหานคร",
            "zipcode": "10900"
        }
    },
    {
        "name": "11. The 'Zero Spaces' Nightmare",
        "input": "เอกราช0894561234บ้านเลขที่9หมู่1ต.บางโฉลงอ.บางพลีจ.สมุทรปราการ10540",
        # Parser Challenge: Absolutely no delimiters. The parser has to rely purely on Regex boundaries
        # (like detecting the 10-digit number) and geographic dictionary lookups.
        "expected": {
            "receiver": "เอกราช",
            "phone": "0894561234",
            "address": "บ้านเลขที่9หมู่1",
            "sub_district": "บางโฉลง",
            "district": "บางพลี",
            "province": "สมุทรปราการ",
            "zipcode": "10540"
        }
    },
    {
        "name": "12. The E-commerce Format (Key-Value pairs)",
        "input": "ชื่อ: วีระชัย\nที่อยู่: 77/8 ม.9 ซอยวัดพระเงิน\nตำบล: บางม่วง\nอำเภอ: บางใหญ่\nจังหวัด: นนทบุรี\nไปรษณีย์: 11140\nเบอร์โทร: 099-555-4433",
        # Parser Challenge: Buyers sometimes copy-paste a form format. 
        # The parser shouldn't get confused by the labels (e.g., shouldn't include "ชื่อ:" in the receiver name).
        "expected": {
            "receiver": "วีระชัย",
            "phone": "099-555-4433",
            "address": "77/8 ม.9 ซอยวัดพระเงิน",
            "sub_district": "บางม่วง",
            "district": "บางใหญ่",
            "province": "นนทบุรี",
            "zipcode": "11140"
        }
    },
    {
        "name": "1. The Standard but Heavily Abbreviated Buyer",
        "input": "ส่งตามนี้เลยครับ แอดมิน\nสมชาย ใจดี\n123/45 ม.6 ซ.สุขสบาย ถ.เส้นหลัก\nต.หนองปลาไหล อ.บางละมุง จ.ชลบุรี 20150\nโทร 081-234-5678",
        "expected": {
            "receiver": "สมชาย ใจดี",
            "phone": "081-234-5678",
            "address": "123/45 ม.6 ซ.สุขสบาย ถ.เส้นหลัก",
            "sub_district": "หนองปลาไหล",
            "district": "บางละมุง",
            "province": "ชลบุรี",
            "zipcode": "20150"
        }
    },
    {
        "name": "2. The Bangkok Condo Minimalist",
        "input": "แพรว 0998887777 ค่ะ จัดส่งที่ คอนโด แฮปปี้ไลฟ์ ชั้น 5 ห้อง 501 เลขที่ 88 รัชดาภิเษก ห้วยขวาง กทม 10310",
        "expected": {
            "receiver": "แพรว",
            "phone": "0998887777",
            "address": "คอนโด แฮปปี้ไลฟ์ ชั้น 5 ห้อง 501 เลขที่ 88 รัชดาภิเษก",
            "sub_district": "ห้วยขวาง",
            "district": "ห้วยขวาง",
            "province": "กรุงเทพมหานคร",
            "zipcode": "10310"
        }
    },
    {
        "name": "3. The Everything in One Long String",
        "input": "นาย สมศักดิ์ รักเรียน 44/4 ถ.เชียงใหม่-ลำปาง ช้างเผือก เมือง เชียงใหม่ 50300 0823334455",
        "expected": {
            "receiver": "นาย สมศักดิ์ รักเรียน",
            "phone": "0823334455",
            "address": "44/4 ถ.เชียงใหม่-ลำปาง",
            "sub_district": "ช้างเผือก",
            "district": "เมืองเชียงใหม่", # Expecting parser to normalize "เมือง" to "เมืองเชียงใหม่"
            "province": "เชียงใหม่",
            "zipcode": "50300"
        }
    },
    {
        "name": "4. The Address First, Name Last Buyer",
        "input": "เอาส่งมาที่ 99 ม.1 หมู่บ้านโคกสว่าง ต.โคกสว่าง อ.เมือง จ.ขอนแก่น นะคะ\nรหัสไปรษณีย์ 40000\nเบอร์ 085-111-2233\nชื่อผู้รับ: ป้าสมศรี สู้ชีวิต",
        "expected": {
            "receiver": "ป้าสมศรี สู้ชีวิต",
            "phone": "085-111-2233",
            "address": "99 ม.1 หมู่บ้านโคกสว่าง",
            "sub_district": "โคกสว่าง",
            "district": "เมืองขอนแก่น", # Expecting normalization
            "province": "ขอนแก่น",
            "zipcode": "40000"
        }
    },
    {
        "name": "5. The Office Delivery (Complex Routing)",
        "input": "บจก. เทสติ้ง ดาต้า (ฝากส่งให้คุณสมหญิง แผนกบัญชี)\n555 อาคารรวมโชค ชั้น 10 ถ.สุขุมวิท คลองเตย กรุงเทพมหานคร 10110\nเบอร์ติดต่อ 02-111-2222 ต่อ 15",
        "expected": {
            "receiver": "คุณสมหญิง แผนกบัญชี", # Or however you choose to handle company/dept routing
            "phone": "02-111-2222 ต่อ 15",
            "address": "555 อาคารรวมโชค ชั้น 10 ถ.สุขุมวิท",
            "sub_district": "คลองเตย",
            "district": "คลองเตย",
            "province": "กรุงเทพมหานคร",
            "zipcode": "10110"
        }
    },
    {
        "name": "6. The Missing Postal Code Buyer",
        "input": "ส่ง ยายบุญมา ใจเย็น\n7/2 บ้านหนองหอย หมู่ 3 ต.ดงมะไฟ อ.เมือง สกลนคร\nโทร 0884445555\nรหัสไปรษณีย์ยายจำไม่ได้จ้า รบกวนแอดมินหาให้หน่อยนะ",
        "expected": {
            "receiver": "ยายบุญมา ใจเย็น",
            "phone": "0884445555",
            "address": "7/2 บ้านหนองหอย หมู่ 3",
            "sub_district": "ดงมะไฟ",
            "district": "เมืองสกลนคร",
            "province": "สกลนคร",
            "zipcode": "47000" # Ideally, the parser autofills this based on the geography
        }
    },
    {
        "name": "13. The Storyteller (High Conversational Noise)",
        "input": "แอดมินคะ รบกวนส่งให้คุณแม่หน่อยนะคะ พอดีท่านอยู่บ้านคนเดียว ชื่อ ยายสมใจ รักดี เบอร์ 0891112222 บ้านเลขที่ 45/9 หมู่ 8 ต.คลองสาม อ.คลองหลวง ปทุมธานี 12120 ค่ะ ขอบคุณมากๆ ค่ะ รีบส่งนะคะ",
        # Parser Challenge: Filtering out all the polite particles (คะ, ค่ะ), the backstory, 
        # and instructions ("รบกวนส่ง...", "รีบส่งนะคะ") to isolate just the entities.
        "expected": {
            "receiver": "ยายสมใจ รักดี",
            "phone": "0891112222",
            "address": "บ้านเลขที่ 45/9 หมู่ 8",
            "sub_district": "คลองสาม",
            "district": "คลองหลวง",
            "province": "ปทุมธานี",
            "zipcode": "12120"
        }
    },
    {
        "name": "14. The Romanized / English Address",
        "input": "Mr. David Smith 0819998888 12/34 Sukhumvit 71 Rd., Phra Khanong Nuea, Watthana, Bangkok 10110",
        # Parser Challenge: Does your Thai parser handle English? It needs to recognize 
        # "Bangkok" as "กรุงเทพมหานคร" and understand English formatting.
        "expected": {
            "receiver": "Mr. David Smith",
            "phone": "0819998888",
            "address": "12/34 Sukhumvit 71 Rd.",
            "sub_district": "Phra Khanong Nuea", # Or the Thai translation if your API translates
            "district": "Watthana",
            "province": "Bangkok",
            "zipcode": "10110"
        }
    },
    {
        "name": "15. Missing District (Amphoe)",
        "input": "สมชาย 0855554444 ส่งที่ 99/9 ต.หนองปรือ จ.ชลบุรี 20150",
        # Parser Challenge: The user skipped the District entirely. 
        # The parser must infer "บางละมุง" based on the Tambon (หนองปรือ) and Zipcode (20150).
        "expected": {
            "receiver": "สมชาย",
            "phone": "0855554444",
            "address": "99/9",
            "sub_district": "หนองปรือ",
            "district": "บางละมุง", # Auto-filled by geographic dictionary
            "province": "ชลบุรี",
            "zipcode": "20150"
        }
    },
    {
        "name": "16. Missing Sub-district (Tambon)",
        "input": "พิมดาว 0944443333 55 หมู่บ้านสุขสันต์ อ.เมือง จ.เชียงใหม่ 50000",
        # Parser Challenge: The user skipped the Sub-district. Because Amphoe Mueang Chiang Mai 
        # has many sub-districts sharing 50000, it might be impossible to guess. 
        # The parser should safely leave it blank or None, rather than guessing wrong.
        "expected": {
            "receiver": "พิมดาว",
            "phone": "0944443333",
            "address": "55 หมู่บ้านสุขสันต์",
            "sub_district": None, # Or "" depending on your API's null behavior
            "district": "เมืองเชียงใหม่",
            "province": "เชียงใหม่",
            "zipcode": "50000"
        }
    },
    {
        "name": "17. Social Media Handles & Extra IDs",
        "input": "ชื่อ: ก้องเกียรติ\nLine ID: kongkiat_123\nโทร: 0867778888\nFB: ก้องเกียรติ สุดเท่\nที่อยู่: 111 ถ.ลาดพร้าว แขวงจอมพล เขตจตุจักร กทม 10900",
        # Parser Challenge: The parser must NOT confuse the Line ID or FB handle 
        # with the Receiver's name or the Address line.
        "expected": {
            "receiver": "ก้องเกียรติ",
            "phone": "0867778888",
            "address": "111 ถ.ลาดพร้าว",
            "sub_district": "จอมพล",
            "district": "จตุจักร",
            "province": "กรุงเทพมหานคร",
            "zipcode": "10900"
        }
    },
    {
        "name": "18. Landmark Only (No House Number)",
        "input": "พี่หนุ่ม 0812345678 ร้านกาแฟหน้าปากซอย 2 ท่าทราย เมือง นนทบุรี 11000",
        # Parser Challenge: There is no formal "บ้านเลขที่" (house number). 
        # The address is just a descriptive landmark ("ร้านกาแฟหน้าปากซอย 2").
        "expected": {
            "receiver": "พี่หนุ่ม",
            "phone": "0812345678",
            "address": "ร้านกาแฟหน้าปากซอย 2",
            "sub_district": "ท่าทราย",
            "district": "เมืองนนทบุรี",
            "province": "นนทบุรี",
            "zipcode": "11000"
        }
    }
]

def print_what_it_should_be(expected):
    """Prints the clean layout format when a test fails."""
    print("\n--- What it should be ---")
    print("ผู้รับ (RECEIVER)")
    print(f"- {expected.get('receiver')}")
    print("เบอร์โทร (PHONE)")
    print(f"- {expected.get('phone')}")
    print("รายละเอียด (ADDRESS)")
    print(f"- {expected.get('address')}")
    print("ตำบล / แขวง (SUB_DISTRICT)")
    print(f"- {expected.get('sub_district')}")
    print("อำเภอ / เขต (DISTRICT)")
    print(f"- {expected.get('district')}")
    print("จังหวัด (PROVINCE)")
    print(f"- {expected.get('province')}")
    print("รหัสไปรษณีย์ (ZIPCODE)")
    print(f"- {expected.get('zipcode')}\n")

def run_tests():
    url = f"{BASE_URL}/api/parse"
    total_tests = len(TEST_CASES)
    passed_tests = 0

    print(f"Starting API Parser Tests on {url}...\n" + "="*50)

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}/{total_tests}: {test['name']}")
        print(f"Input Data: '{test['input']}'")
        
        payload = {"text": test['input']}
        
        try:
            response = requests.post(url, json=payload, headers=HEADERS, timeout=5)
        except requests.exceptions.ConnectionError:
            print("❌ ERROR: Could not connect to the API. Is your server running?")
            sys.exit(1)

        if response.status_code == 429:
            print("❌ ERROR: Rate limit exceeded. Please wait and try again.")
            sys.exit(1)
        elif response.status_code == 401:
            print("❌ ERROR: Unauthorized. Check your API key.")
            sys.exit(1)
        elif response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            continue

        response_data = response.json()
        expected = test['expected']
        
        print("--- Checking Output Structure ---")
        all_fields_passed = True
        
        # We explicitly check these fields to ensure consistent reporting
        keys_to_check = ["receiver", "phone", "address", "sub_district", "district", "province", "zipcode"]
        
        for field in keys_to_check:
            expected_val = expected.get(field)
            actual_val = response_data.get(field)

            # Smart comparison rules:
            # phone: normalise dashes/dots before comparing
            # address: check actual contains the expected key fragment
            # others: exact match (None == None counts as pass)
            passed = False
            if expected_val is None and actual_val is None:
                passed = True
            elif expected_val is None or actual_val is None:
                passed = False
            elif field == "phone":
                import re as _re
                norm_exp = _re.sub(r"[\s\-\./]", "", str(expected_val))
                norm_act = _re.sub(r"[\s\-\./]", "", str(actual_val))
                # Allow "02-111-2222 ต่อ 15" to match "021112222" (primary number)
                norm_exp_base = _re.sub(r"\s*ต่อ.*$", "", norm_exp)
                passed = (norm_exp == norm_act) or (norm_exp_base == norm_act)
            elif field == "address":
                # Check that actual address contains the expected key fragment
                passed = (expected_val in str(actual_val)) if actual_val else False
            else:
                passed = (actual_val == expected_val)

            if not passed:
                print(f"❌ FAILED on '{field}':")
                print(f"   Expected : {expected_val}")
                print(f"   Actual   : {actual_val}")
                all_fields_passed = False
            else:
                print(f"✅ PASSED '{field}': {actual_val}")

        if all_fields_passed:
            passed_tests += 1
            print("🎉 RESULT: FULLY PASSED")
        else:
            print("⚠️ RESULT: FAILED")
            print_what_it_should_be(expected)
            
        print("-" * 50)

    # Final Summary
    print(f"\n=== TEST RUN COMPLETE ===")
    print(f"Passed: {passed_tests}/{total_tests}")
    if passed_tests == total_tests:
        print("✅ SUCCESS: Your parser handles all edge cases perfectly!")
    else:
        print("⚠️ IMPROVEMENT NEEDED: Check the failed test cases to adjust your regex or matching logic.")

if __name__ == "__main__":
    try:
        health = requests.get(f"{BASE_URL}/api/health", timeout=2)
        if health.status_code == 200:
            print(f"Server is healthy. Uptime: {health.json().get('uptime_s', 'Unknown')}s")
            run_tests()
        else:
            print(f"Server health check failed: {health.status_code}")
    except requests.exceptions.RequestException:
        print("❌ ERROR: Server is not running on port 8000.")