# backend/routes_analytics.py
"""Mood analytics endpoints (require auth).

  GET /analytics/summary   -> headline stats + mood band
  GET /analytics/mood?days=30  -> daily average sentiment time series
  GET /analytics/emotions  -> emotion distribution (counts)
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from database import Conversation, Message
from sentiment import mood_band

analytics_bp = Blueprint("analytics", __name__, url_prefix="/analytics")


def _user_messages(user_id, role="user", since=None):
    """All messages for a user across conversations, optionally filtered."""
    q = (
        Message.query.join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Conversation.user_id == user_id)
    )
    if role:
        q = q.filter(Message.role == role)
    if since:
        q = q.filter(Message.created_at >= since)
    return q.order_by(Message.created_at.asc()).all()


@analytics_bp.route("/summary", methods=["GET"])
@jwt_required()
def summary():
    user_id = int(get_jwt_identity())
    msgs = _user_messages(user_id)
    scores = [m.sentiment_score for m in msgs if m.sentiment_score is not None]

    avg = round(sum(scores) / len(scores), 4) if scores else 0.0
    # Trend: average of last 5 vs previous 5 analysed messages.
    recent = scores[-5:]
    prior = scores[-10:-5]
    trend = 0.0
    if recent and prior:
        trend = round((sum(recent) / len(recent)) - (sum(prior) / len(prior)), 4)

    high_risk_count = sum(1 for m in msgs if m.is_high_risk)
    conv_count = Conversation.query.filter_by(user_id=user_id).count()

    return jsonify({
        "total_messages": len(msgs),
        "total_conversations": conv_count,
        "average_sentiment": avg,
        "mood_band": mood_band(avg),
        "trend": trend,  # >0 improving, <0 declining
        "high_risk_messages": high_risk_count,
    })


@analytics_bp.route("/mood", methods=["GET"])
@jwt_required()
def mood_timeseries():
    user_id = int(get_jwt_identity())
    days = min(max(int(request.args.get("days", 30)), 1), 365)
    since = datetime.now(timezone.utc) - timedelta(days=days)

    msgs = _user_messages(user_id, since=since)

    buckets = defaultdict(list)
    for m in msgs:
        if m.sentiment_score is None or m.created_at is None:
            continue
        day = m.created_at.strftime("%Y-%m-%d")
        buckets[day].append(m.sentiment_score)

    series = [
        {
            "date": day,
            "average_sentiment": round(sum(v) / len(v), 4),
            "count": len(v),
        }
        for day, v in sorted(buckets.items())
    ]
    return jsonify({"series": series, "days": days})


@analytics_bp.route("/emotions", methods=["GET"])
@jwt_required()
def emotion_distribution():
    user_id = int(get_jwt_identity())
    msgs = _user_messages(user_id)
    counts = Counter(m.emotion for m in msgs if m.emotion)
    return jsonify({
        "distribution": [
            {"emotion": emo, "count": n}
            for emo, n in counts.most_common()
        ]
    })
