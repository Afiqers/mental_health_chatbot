import json
import os
import time
import urllib.request
import urllib.error

try:
    # Centralised config when run inside the app.
    from config import Config
    LLM_PROVIDER = Config.LLM_PROVIDER
    GEMINI_API_KEY = Config.GEMINI_API_KEY
    GEMINI_MODEL = Config.GEMINI_MODEL
    ANTHROPIC_MODEL = Config.ANTHROPIC_MODEL
    ANTHROPIC_MAX_TOKENS = Config.ANTHROPIC_MAX_TOKENS
    OLLAMA_MODEL = Config.OLLAMA_MODEL
    OLLAMA_URL = Config.OLLAMA_URL
    OLLAMA_TIMEOUT = Config.OLLAMA_TIMEOUT
except Exception:
    # Standalone fallback (e.g. running this file directly in a test).
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    ANTHROPIC_MODEL = "claude-haiku-4-5"
    ANTHROPIC_MAX_TOKENS = 400
    OLLAMA_MODEL = "llama3.2"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_TIMEOUT = 30

# Common Malay function words and discourse markers for language detection
_MALAY_MARKERS = {
    "saya", "aku", "kamu", "awak", "dia", "kami", "kita", "mereka",
    "tak", "tidak", "tiada", "bukan", "jangan",
    "dan", "atau", "tapi", "tetapi", "sebab", "kerana", "supaya",
    "dengan", "untuk", "kepada", "dari", "daripada", "dalam", "pada",
    "ini", "itu", "yang", "pun", "juga", "sudah", "dah", "akan",
    "boleh", "nak", "mahu", "hendak", "perlu", "ada", "ade",
    "sangat", "amat", "lebih", "agak", "macam", "mcm", "macam",
    "rasa", "rase", "tolong", "please", "terima", "kasih",
    "hai", "helo", "selamat", "baik", "okay", "ok",
    "kenapa", "mengapa", "bagaimana", "mana", "bila", "siapa",
    "lah", "leh", "kan", "lor", "meh", "yer", "ye",
}

# Map emotion labels to Solace sentiment tiers for prompt guidance.
_SENTIMENT_GUIDE = {
    "NORMAL":   "POSITIVE / NEUTRAL — engage supportively, explore what's going well",
    "NEUTRAL":  "POSITIVE / NEUTRAL — engage supportively, explore what's going well",
    "STRESS":   "MILD_DISTRESS — validate feelings, gently explore emotions, offer grounding coping strategies",
    "ANXIETY":  "MILD_DISTRESS — validate feelings, gently explore emotions, offer grounding coping strategies",
    "DEPRESSION": "MODERATE_DISTRESS — prioritize emotional validation, slow down, ask what they need",
    "SUICIDAL": "HIGH_DISTRESS / CRISIS — acknowledge pain directly, share crisis resources, encourage professional help",
    "HIGH_RISK": "HIGH_DISTRESS / CRISIS — acknowledge pain directly, share crisis resources, encourage professional help",
}

# Target reply length. Keeps responses from being curt one-liners or rambling
# paragraphs — a consistent, readable size for a chat bubble.
_LENGTH_RULE = (
    "LENGTH: Write 2 to 4 sentences, roughly 40-75 words. Long enough to feel "
    "warm and specific, short enough to read at a glance. Never reply with a "
    "single short line, and never write more than four sentences or multiple "
    "paragraphs."
)


def detect_language(text: str) -> str:
    """Returns 'Malay' if the text appears to be Malay, else 'English'."""
    if not text:
        return "English"
    words = set(text.lower().split())
    malay_hits = words & _MALAY_MARKERS
    # If at least 1 Malay marker found OR >25% of words match, treat as Malay
    if malay_hits and (len(malay_hits) >= 1 or len(malay_hits) / len(words) > 0.25):
        return "Malay"
    return "English"


def _format_history(history: list) -> str:
    """Render the last few turns as a readable transcript for the prompt."""
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Solace"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "\n\nConversation so far:\n" + "\n".join(lines) + "\n"


def _solace_system_prompt(user_language: str) -> str:
    """Persona + absolute language constraint, evaluated first by the model."""
    return (
        "You are Solace, a compassionate and emotionally intelligent mental health "
        "support companion. Your role is to provide empathetic, non-judgmental support "
        "that helps users process their emotions.\n\n"
        f"ABSOLUTE LANGUAGE RULE: The user is writing in {user_language}. "
        f"You MUST respond entirely in {user_language}. "
        "Do NOT switch languages, mix languages, or respond in English if the user "
        "wrote in Malay. This rule overrides everything else."
    )


def _call_llm(system_prompt: str, prompt: str, temperature: float = 0.7):
    """Dispatch to the configured LLM backend. Returns text, or None on failure
    (so callers fall back to the curated template bank).

    For the default "gemini" provider, if Gemini fails or is rate-limited we
    automatically fall back to the local Ollama model so replies keep flowing.
    """
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system_prompt, prompt, temperature)
    if LLM_PROVIDER == "claude":
        return _call_claude(system_prompt, prompt, temperature)
    # Default: Gemini first, then local Ollama as a fallback.
    result = _call_gemini(system_prompt, prompt, temperature)
    if result is not None:
        return result
    return _call_ollama(system_prompt, prompt, temperature)


# Lazily-constructed Anthropic client (reused across calls). None until first
# successful construction; -1 marks a permanent failure (e.g. missing API key)
# so we don't retry on every message.
_claude_client = None


def _call_claude(system_prompt: str, prompt: str, temperature: float = 0.7):
    """Call the Claude API (Anthropic). Returns the reply text, or None.

    Uses Haiku by default (cheapest model) with a small max_tokens cap, since
    replies are short — both keep token cost low. The system prompt is tiny
    (well under Haiku's 4096-token cache minimum), so prompt caching wouldn't
    engage; we don't set cache_control here.
    """
    global _claude_client
    if _claude_client == -1:
        return None  # already known to be unavailable
    try:
        if _claude_client is None:
            import anthropic
            _claude_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        message = _claude_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in message.content if b.type == "text").strip(" \n\"")
        return text or None
    except Exception as e:
        # Missing/invalid key, network error, rate limit, etc. Mark unavailable
        # only for auth/config issues; transient errors just fail this one call.
        import anthropic
        if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            _claude_client = -1
            print(f"Claude API unavailable ({e}). Falling back to template responses. "
                  f"Set ANTHROPIC_API_KEY to enable.")
        else:
            print(f"Claude API error: {e}")
        return None


# After a Gemini rate-limit (429) we skip Gemini for a short window and use the
# Ollama fallback instead of retrying (and failing) on every message.
_gemini_cooldown_until = 0.0
GEMINI_COOLDOWN_SECONDS = 60


def _call_gemini(system_prompt: str, prompt: str, temperature: float = 0.7):
    """Call the Google Gemini API (free tier). Returns text, or None on failure.

    On a 429 (quota / rate limit) it starts a short cooldown so the caller falls
    straight through to Ollama until the limit resets.
    """
    global _gemini_cooldown_until
    if not GEMINI_API_KEY:
        return None
    if time.time() < _gemini_cooldown_until:
        return None  # recently rate-limited — let the caller use the fallback

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
    data = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 300},
    }
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        candidates = result.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip(" \n\"")
        return text or None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            _gemini_cooldown_until = time.time() + GEMINI_COOLDOWN_SECONDS
            print(f"Gemini rate limit (429). Falling back to Ollama for {GEMINI_COOLDOWN_SECONDS}s.")
        else:
            print(f"Gemini API error: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


def _call_ollama(system_prompt: str, prompt: str, temperature: float = 0.7):
    """Call the local Ollama model. Returns the text, or None on any failure."""
    data = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        # num_predict caps generation length as a safety net; ~220 tokens is
        # generous for a 2-4 sentence reply but stops the model from rambling.
        "options": {"temperature": temperature, "num_predict": 220},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
            text = (result.get("response") or "").strip(" \n\"")
            return text or None
    except Exception as e:
        print(f"Ollama API Error: {e} - Ensure Ollama is running and '{OLLAMA_MODEL}' is downloaded.")
        return None


def generate_contextual_reply(user_message: str, emotion: str, confidence: float,
                              history: list = None) -> str:
    """Generate a context-aware reply grounded in the whole conversation.

    Unlike rephrase_response (which dresses up a fixed template), this writes a
    fresh reply that builds on what the user has shared across turns, steered by
    the detected emotional state.

    Returns the generated text, or None if the LLM is unavailable — in which case
    the caller should fall back to the template bank.
    """
    history = history or []
    user_language = detect_language(user_message)
    history_section = _format_history(history)
    sentiment_tier = _SENTIMENT_GUIDE.get(emotion, "MILD_DISTRESS — validate and listen carefully")

    prompt = (
        f"{history_section}\n"
        f"The user just said: '{user_message}'.\n"
        f"Their detected emotional state for this message is: {emotion} — "
        f"Sentiment tier: {sentiment_tier}.\n\n"

        f"Write Solace's next reply in the conversation.\n\n"

        f"## How to respond\n"
        f"- Respond with warmth, empathy and validation FIRST, before any suggestion.\n"
        f"- Be CONTEXTUAL: build naturally on what the user has shared earlier in the "
        f"conversation. If they are answering a question you asked or continuing a topic "
        f"(e.g. a short 'yes', 'not really', 'it's my exams'), respond to it in that context "
        f"rather than treating it as a brand-new statement.\n"
        f"- You may gently reflect what they've told you, but do it like a caring friend — "
        f"do NOT say robotic phrases like 'You said' or 'Earlier you mentioned'.\n"
        f"- Mirror their emotional tone: calm if they're calm, gentle if they're distressed.\n"
        f"- For MILD/MODERATE distress, you may offer ONE simple, concrete coping idea "
        f"(a breathing or grounding exercise, a tiny next step) — but only after validating.\n"
        f"- {_LENGTH_RULE}\n"
        f"- Speak warmly, with no clinical jargon.\n"
        f"- Never diagnose, prescribe, or claim to replace professional care.\n"
        f"- Never be dismissive or toxic-positive (e.g. 'Everything will be fine!').\n"
        f"- End with ONE gentle, open-ended question that moves the conversation forward.\n\n"

        f"REMINDER: Respond entirely in {user_language}, in 2-4 sentences (about 40-75 words). "
        f"Output only Solace's reply — no quotation marks, no preamble, no labels."
    )

    return _call_llm(_solace_system_prompt(user_language), prompt)


def rephrase_response(text: str, emotion: str, user_message: str = "", history: list = None) -> str:
    """
    Rephrases a fixed template into the Solace persona using the local Ollama model.
    Used as a fallback when contextual generation is unavailable. On failure it
    returns the original `text` unchanged so the user always gets a reply.

    history: list of {"role": "user"|"bot", "content": "..."} dicts (last N turns).
    """
    history = history or []
    history_section = _format_history(history)
    sentiment_tier = _SENTIMENT_GUIDE.get(emotion, "MILD_DISTRESS — validate and listen carefully")
    user_language = detect_language(user_message)

    prompt = (
        f"The user said: '{user_message}'.\n"
        f"Their detected emotional state is: {emotion} — Sentiment tier: {sentiment_tier}.\n"
        f"{history_section}"
        f"Your task is to rephrase the following response to match the Solace persona: '{text}'.\n\n"

        f"## Solace Core Behavior\n"
        f"- Always respond with warmth, empathy, and validation FIRST before offering any advice.\n"
        f"- Mirror the user's emotional tone — calm if they're calm, gentle if they're distressed.\n"
        f"- Use active listening: reflect what you sense, then ask one gentle open-ended follow-up question.\n"
        f"- Never diagnose, prescribe, or claim to replace professional mental health care.\n"
        f"- {_LENGTH_RULE}\n"
        f"- Avoid clinical jargon — speak like a caring, warm human companion.\n"
        f"- Never be dismissive, toxic-positive (e.g., 'Semua akan baik-baik saja!'), or give unsolicited advice.\n\n"

        f"## Critical Rules\n"
        f"1. You are an AI companion, NOT a human. Use 'I' only when referring to yourself as a supportive AI.\n"
        f"2. If the text contains a consent question (e.g., 'Would you like to try...') or a structured "
        f"interactive exercise (e.g., 'name 3 things', 'breathe in for 4 seconds'), YOU MUST preserve "
        f"that structured exercise in your rephrasing — translated into {user_language}.\n"
        f"3. Let the conversation context shape your tone naturally without quoting it back robotically.\n"
        f"4. End with a gentle open-ended question to keep the conversation going — UNLESS the emotion is "
        f"SUICIDAL or HIGH_RISK, in which case stay focused entirely on their safety.\n\n"

        f"REMINDER: Respond entirely in {user_language}, in 2-4 sentences (about 40-75 words). "
        f"Output the rephrased response directly. No quotation marks, no preamble, no filler phrases."
    )

    result = _call_llm(_solace_system_prompt(user_language), prompt)
    return result if result else text
