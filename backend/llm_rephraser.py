import json
import urllib.request
import urllib.error

# We will use the 'llama3.2' model via Ollama (a fast and capable local model)
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

def rephrase_response(text: str, emotion: str, user_message: str = "") -> str:
    """
    Rephrases the input text to be natural and empathetic using a local Ollama model.
    Adapts the language based on the user's message.
    """
    prompt = (
        f"You are a highly empathetic, clinical-lite mental health companion similar to Wysa. "
        f"The user said: '{user_message}' and is feeling {emotion}. "
        f"Your task is to rephrase the following response to sound extremely warmly conversational, safe, and non-judgmental: '{text}'.\n\n"
        f"CRITICAL RULES:\n"
        f"1. Do not act like a human. Use 'I' only to refer to yourself as a supportive AI.\n"
        f"2. IF the text contains a question asking for consent (e.g., 'would you like to try...', 'can we do...'), or an interactive step (e.g., 'name 3 things', 'take a breath', 'breathe in for 4 seconds'), YOU MUST preserve that exact structured exercise in your rephrasing.\n"
        f"3. Do not add flowery language or toxic positivity ('It gets better!'). Stay grounded.\n"
        f"4. IDENTIFY the language the user is speaking in '{user_message}' (e.g., English, Malay). The rephrased response MUST be written fluently in the EXACT SAME LANGUAGE as the user's message.\n\n"
        f"Just output the rephrased sentence directly, without any conversational filler or quotation marks."
    )
    
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", text).strip(' \n"')
    except Exception as e:
        print(f"Ollama API Error: {e} - Ensure Ollama is running and '{OLLAMA_MODEL}' is downloaded.")
        return text
