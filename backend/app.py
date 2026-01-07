from flask import Flask, request, jsonify
from flask_cors import CORS
from model import classify_text
from response_generator import generate_response

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    emotion, confidence = classify_text(user_message)
    response = generate_response(emotion, confidence)

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True)
