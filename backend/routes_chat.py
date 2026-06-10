# backend/routes_chat.py
"""Chat + conversation management endpoints (all require auth).

  POST   /chat                       -> analyse a message, persist, reply
  GET    /conversations              -> list the user's conversations
  POST   /conversations              -> create a new (empty) conversation
  GET    /conversations/<id>         -> fetch one conversation with messages
  DELETE /conversations/<id>         -> delete a conversation
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from database import db, Conversation, Message
from model import classify_text
from response_generator import generate_response
from sentiment import score_from_distribution

chat_bp = Blueprint("chat", __name__)


def _owned_conversation(conv_id, user_id):
    """Fetch a conversation only if it belongs to the user."""
    return Conversation.query.filter_by(id=conv_id, user_id=user_id).first()


def _title_from(text):
    text = text.strip().replace("\n", " ")
    return (text[:40] + "…") if len(text) > 40 else (text or "New conversation")


@chat_bp.route("/chat", methods=["POST"])
@jwt_required()
def chat():
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Resolve (or create) the conversation.
    conversation = None
    if conversation_id:
        conversation = _owned_conversation(conversation_id, user_id)
    if conversation is None:
        conversation = Conversation(user_id=user_id, title=_title_from(user_message))
        db.session.add(conversation)
        db.session.flush()  # assign an id

    is_first_message = len(conversation.messages) == 0

    # Build history from stored messages (last N turns) for the rephraser.
    history = [
        {"role": m.role, "content": m.content}
        for m in conversation.messages[-6:]
    ]

    # Most recent user emotion → lets the classifier resolve short follow-ups.
    prev_emotion = next(
        (m.emotion for m in reversed(conversation.messages)
         if m.role == "user" and m.emotion),
        None,
    )

    # ── NLP analysis ──
    emotion, confidence, is_high_risk, distribution = classify_text(
        user_message, prev_emotion=prev_emotion
    )
    sentiment_score = score_from_distribution(distribution)

    # ── Generate the empathetic reply ──
    bot_reply = generate_response(
        user_message=user_message,
        emotion=emotion,
        confidence=confidence,
        is_first_message=is_first_message,
        history=history,
    )

    # ── Persist both messages ──
    db.session.add(Message(
        conversation_id=conversation.id,
        role="user",
        content=user_message,
        emotion=emotion,
        confidence=confidence,
        sentiment_score=sentiment_score,
        is_high_risk=is_high_risk,
    ))
    db.session.add(Message(
        conversation_id=conversation.id,
        role="bot",
        content=bot_reply,
    ))
    db.session.commit()

    return jsonify({
        "conversation_id": conversation.id,
        "emotion": emotion,
        "confidence": confidence,
        "sentiment_score": sentiment_score,
        "is_high_risk": is_high_risk,
        "distribution": distribution,
        "response": bot_reply,
    })


@chat_bp.route("/conversations", methods=["GET"])
@jwt_required()
def list_conversations():
    user_id = int(get_jwt_identity())
    convs = (
        Conversation.query.filter_by(user_id=user_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return jsonify({"conversations": [c.to_dict() for c in convs]})


@chat_bp.route("/conversations", methods=["POST"])
@jwt_required()
def create_conversation():
    user_id = int(get_jwt_identity())
    conv = Conversation(user_id=user_id, title="New conversation")
    db.session.add(conv)
    db.session.commit()
    return jsonify({"conversation": conv.to_dict()}), 201


@chat_bp.route("/conversations/<int:conv_id>", methods=["GET"])
@jwt_required()
def get_conversation(conv_id):
    user_id = int(get_jwt_identity())
    conv = _owned_conversation(conv_id, user_id)
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"conversation": conv.to_dict(include_messages=True)})


@chat_bp.route("/conversations/<int:conv_id>", methods=["DELETE"])
@jwt_required()
def delete_conversation(conv_id):
    user_id = int(get_jwt_identity())
    conv = _owned_conversation(conv_id, user_id)
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    db.session.delete(conv)
    db.session.commit()
    return jsonify({"status": "deleted"})
