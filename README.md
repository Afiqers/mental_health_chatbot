# Mindbot — NLP-Based Sentiment Analysis Mental Health Chatbot

A bilingual (English / Malay) mental-health support chatbot built around an
NLP sentiment-analysis pipeline. It classifies the emotional state behind each
message, replies with an empathetic, LLM-rephrased response, and tracks the
user's mood over time on a personal dashboard.

> ⚠️ Mindbot is an AI companion for educational/FYP purposes — **not** a
> substitute for professional care. Crisis resources (Talian Kasih 15999,
> Befrienders KL 03-7627 2929) are surfaced automatically when self-harm
> language is detected.

---

## Features

- **Bilingual sentiment / emotion classification** — a fine-tuned DistilmBERT
  model, trained on a balanced English + Malay dataset, classifies messages into
  `ANXIETY`, `DEPRESSION`, `SUICIDAL`, `NORMAL` and returns a full probability
  distribution. ~80% accuracy on **both** languages (see the evaluation below).
- **Continuous wellbeing score** — the distribution is collapsed into a single
  score in `[-1, +1]` (see [`backend/sentiment.py`](backend/sentiment.py)), which
  powers the mood trend analytics.
- **Hybrid safety design** — an ML-primary classifier with a transparent rule
  layer: a deterministic **crisis safety net** for explicit self-harm language,
  and a **lexicon backstop** that rescues the model's rare false-negatives
  (clear distress mislabelled as NORMAL — the dangerous direction here).
- **Empathetic responses** — a curated response bank rephrased by a local LLM
  (Ollama `llama3.2`) into the "Solace" persona, matching the user's language
  and emotional tier. Falls back to the canned response if Ollama is offline.
- **Accounts & persistence** — JWT auth (bcrypt-hashed passwords), multiple
  conversations per user, full chat history stored in SQLite.
- **Mood dashboard** — sentiment-over-time line chart, emotion-distribution
  doughnut, and headline stats (average mood, trend, high-risk flags).
- **Model evaluation** — [`backend/evaluate.py`](backend/evaluate.py) produces a
  confusion matrix, per-class precision/recall/F1, and accuracy/macro-F1.

---

## Architecture

```
Frontend (vanilla JS SPA, Chart.js)
        │  REST + JWT
        ▼
Flask backend (blueprints)
  ├── auth.py             register / login / me
  ├── routes_chat.py      /chat + conversation CRUD
  ├── routes_analytics.py /analytics/{summary,mood,emotions}
  │
  ├── model.py            BERT classifier + crisis net + lexicon assist
  ├── sentiment.py        distribution → wellbeing score
  ├── response_generator.py   emotion → empathetic reply
  ├── llm_rephraser.py    Ollama rephrasing + language detection
  └── database.py         User / Conversation / Message (SQLite)
```

Request flow for `/chat`: `classify_text` → `score_from_distribution` →
`generate_response` (→ `rephrase_response`) → persist user + bot messages → JSON.

---

## Setup

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

The fine-tuned model lives in `backend/local_model/` (git-ignored). If absent,
the app falls back to a public model from Hugging Face.

### 2. Ollama (for response rephrasing)

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
```

If Ollama isn't running, the app still works — it serves the canned responses
without rephrasing.

### 3. Run

```bash
# Terminal 1 — backend (from backend/)
python app.py            # serves http://127.0.0.1:5000

# Terminal 2 — frontend (from project root)
python -m http.server 8080 --directory frontend
```

Open <http://127.0.0.1:8080>, create an account, and start chatting.

---

## Training & evaluating the classifier

Retrain the bilingual model (English + Malay) from the raw datasets:

```bash
python train_bilingual.py        # from project root; saves to backend/local_model/
```

This also writes `backend/eval_output/bilingual_report.txt` with **separate
English vs Malay** test metrics (evidence of bilingual capability).

Evaluate the current model and produce a confusion matrix:

```bash
cd backend
python evaluate.py               # held-out split; writes to backend/eval_output/
```

Artifacts: `confusion_matrix.png`, `classification_report.txt`, `metrics.json`.

**Held-out results (DistilmBERT, bilingual):** ~79% overall accuracy
(English 80%, Malay 78%), macro-F1 ≈ 0.79.

---

## Configuration

All settings are environment-overridable (see [`backend/config.py`](backend/config.py)):
`SECRET_KEY`, `JWT_SECRET_KEY`, `DATABASE_URL`, `OLLAMA_MODEL`, `OLLAMA_URL`,
`OLLAMA_TIMEOUT`.

---

## Tech stack

Flask · Flask-JWT-Extended · Flask-SQLAlchemy · Flask-Bcrypt · Hugging Face
Transformers (BERT) · PyTorch · Ollama (llama3.2) · scikit-learn · Chart.js
