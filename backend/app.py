import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import classify_text
from response_generator import generate_response

app = Flask(__name__)
CORS(app)

# --- DB & Auth Setup ---
basedir = os.path.abspath(os.path.dirname(__file__))
# Ensure data directory exists
os.makedirs(os.path.join(basedir, '../data'), exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, '../data/app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-dev-key-change-in-prod')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    chats = db.relationship('ChatMessage', backref='user', lazy=True)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False) # 'user' or 'bot'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    emotion = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)

# Create tables if they don't exist
with app.app_context():
    db.create_all()

# --- Auth Endpoints ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"message": "Username and password are required"}), 400
        
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Username already exists"}), 409
        
    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=hashed_pw)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"message": "User created successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity=str(user.id))
        return jsonify({"access_token": access_token, "username": user.username}), 200
        
    return jsonify({"message": "Invalid username or password"}), 401

@app.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    current_user_id = get_jwt_identity()
    messages = ChatMessage.query.filter_by(user_id=current_user_id).order_by(ChatMessage.timestamp.asc()).all()
    history = []
    for msg in messages:
        history.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "emotion": msg.emotion,
            "confidence": msg.confidence
        })
    return jsonify({"history": history}), 200

# --- Chat Endpoint ---
@app.route("/chat", methods=["POST"])
@jwt_required(optional=True)
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    emotion, confidence, is_high_risk = classify_text(user_message)
    response = generate_response(emotion)

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
