# backend/config.py
"""Central configuration for the Mindbot backend.

Values can be overridden via environment variables so the same code runs
in a local demo and (later) on a server without edits.
"""

import os
from datetime import timedelta

# Project root = one level above /backend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class Config:
    # --- Flask / security ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-mindbot-secret-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-mindbot-jwt-change-me")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=7)

    # --- Database ---
    # Single SQLite file holds users, conversations and messages.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'mindbot.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- LLM backend for response generation ---
    # "ollama" (local, free) or "claude" (Anthropic API, needs key). Defaults to Ollama.
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

    # Claude API. Haiku 4.5 is the cheapest model (lowest $/token) — chosen to
    # minimise token cost. Set ANTHROPIC_API_KEY in the environment to enable;
    # if unset, the app falls back to the curated response bank.
    ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    # Replies are 2-4 sentences, so a small cap keeps token usage (and cost) low.
    ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "400"))

    # Local Ollama (used only when LLM_PROVIDER=ollama).
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    # First call after idle cold-loads the model into RAM and can be slow.
    OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

    # How many recent turns of history to feed the rephraser.
    HISTORY_TURNS = 6
