import requests

tests = [
    'hi', 
    'bye', 
    'hello', 
    'goodbye', 
    'thanks', 
    'I want to kill myself, I cannot take this pain anymore'
]

print(f"{'Status':<10} {'Emotion':<15} {'Confidence'}  Input")
print("-" * 60)

for msg in tests:
    try:
        r = requests.post('http://127.0.0.1:5000/chat', json={'message': msg}, timeout=30).json()
        flag = '[CRISIS]' if r.get('is_high_risk') else '[SAFE]  '
        print(f"{flag:<10} [{r.get('emotion', 'N/A'):<13}] {r.get('confidence', 0):>6.0%}  <- \"{msg[:45]}\"")
    except Exception as e:
        print(f"Error testing '{msg}': {e}")
