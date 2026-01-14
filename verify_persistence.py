import requests
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_persistence():
    print("1. Sending a test message...")
    try:
        payload = {"message": "Hello, this is a persistence test."}
        res = requests.post(f"{BASE_URL}/chat", json=payload)
        res.raise_for_status()
        print("   Chat response:", res.json())
    except Exception as e:
        print(f"   FAILED to chat: {e}")
        print("   Make sure server is running: python backend/app.py")
        return

    print("\n2. Checking history endpoint...")
    try:
        res = requests.get(f"{BASE_URL}/history")
        res.raise_for_status()
        history = res.json()
        print(f"   Retrieved {len(history)} messages from history.")
        
        # Check if our message is there
        found = False
        for msg in history:
            if msg.get('content') == "Hello, this is a persistence test." and msg.get('role') == 'user':
                found = True
                break
        
        if found:
            print("   SUCCESS: Test message found in history!")
        else:
            print("   FAILURE: Test message NOT found in history.")
            print("   Full History:", history)
            
    except Exception as e:
        print(f"   FAILED to get history: {e}")

if __name__ == "__main__":
    test_persistence()
