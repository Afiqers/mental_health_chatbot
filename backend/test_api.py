# backend/test_api.py
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Run this AFTER starting app.py manually in another terminal:
#   python app.py
# Then in a second terminal run:
#   python test_api.py

import requests
import sys
import time

BASE_URL = "http://127.0.0.1:5000"

# ── Poll until Flask is ready ────────────────────────────────
print("Waiting for Flask server to be ready...")
for attempt in range(30):  # wait up to 5 minutes (30 x 10s)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"[OK] Server is ready! (attempt {attempt + 1})\n")
            break
    except requests.exceptions.ConnectionError:
        pass
    print(f"  [{attempt + 1}/30] Not ready yet, retrying in 10s...")
    time.sleep(10)
else:
    print("[FAIL] Flask server did not start in time. Make sure app.py is running.")
    sys.exit(1)

# ── Test cases ───────────────────────────────────────────────
test_cases = [
    ("I'm feeling really anxious and can't calm down",          "ANXIETY"),
    ("Everything feels hopeless, I don't see a point anymore",  "DEPRESSION / SUICIDAL"),
    ("I want to kill myself, I can't take it anymore",          "SUICIDAL"),
    ("I had a good day today, feeling okay",                    "NORMAL"),
    ("saya rasa sedih dan tidak berdaya",                       "DEPRESSION (Malay)"),
]

print("=" * 60)
print("BACKEND API TEST RESULTS")
print("=" * 60)

passed = 0
failed = 0

for message, expected_label in test_cases:
    try:
        res = requests.post(f"{BASE_URL}/chat", json={"message": message}, timeout=30)
        data = res.json()

        emotion     = data.get("emotion", "N/A")
        confidence  = data.get("confidence", 0)
        is_high_risk = data.get("is_high_risk", False)
        response_text = data.get("response", "")[:80]

        status = "[OK]" if res.status_code == 200 else "[FAIL]"
        print(f"\n{status} Input   : {message[:60]}")
        print(f"   Expected : {expected_label}")
        print(f"   Got      : {emotion} (confidence: {confidence:.0%}, high_risk={is_high_risk})")
        print(f"   Reply    : {response_text}...")
        passed += 1

    except Exception as e:
        print(f"\n[FAIL]: {message[:50]}")
        print(f"   Error: {e}")
        failed += 1

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)
