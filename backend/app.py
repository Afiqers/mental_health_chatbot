# backend/app.py
"""Mindbot backend — application entry point.

Run from the /backend directory:   python app.py
"""

from flask import Flask, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

from config import Config
from database import db
from auth import auth_bp, init_auth
from routes_chat import chat_bp
from routes_analytics import analytics_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app)
    db.init_app(app)
    bcrypt = Bcrypt(app)
    JWTManager(app)
    init_auth(bcrypt)

    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(analytics_bp)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    with app.app_context():
        db.create_all()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
