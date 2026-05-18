import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import classify_text
from response_generator import generate_response

app = Flask(__name__)
CORS(app)

# --- Chat Endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    # Last N turns: [{"role": "user"|"bot", "content": "..."}]
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    emotion, confidence, is_high_risk = classify_text(user_message)
    response = generate_response(
        user_message=user_message,
        emotion=emotion,
        confidence=confidence,
        history=history
    )

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "is_high_risk": is_high_risk,
        "response": response
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
