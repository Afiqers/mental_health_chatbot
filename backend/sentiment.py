# backend/sentiment.py
"""Continuous sentiment / wellbeing scoring.

The classifier outputs a probability distribution over emotional states.
Rather than throwing that away and keeping only the top label, we collapse
the whole distribution into a single continuous "wellbeing score" in the
range [-1.0, +1.0]:

    score = Σ  P(class) * valence(class)

  +1.0  => clearly positive / settled
   0.0  => neutral / mixed
  -1.0  => severe distress / crisis

This score is what we persist per message and aggregate into the mood
dashboard, so the "analysis" in *sentiment analysis* is actually visible.
"""

# Valence weight for each emotional state.
VALENCE = {
    "NORMAL": 1.0,
    "NEUTRAL": 0.6,
    "GREETING": 0.5,
    "FAREWELL": 0.5,
    "STRESS": -0.3,
    "ANXIETY": -0.45,
    "DEPRESSION": -0.7,
    "SUICIDAL": -1.0,
    "HIGH_RISK": -1.0,
    "UNKNOWN": 0.0,
}


def score_from_distribution(distribution: dict) -> float:
    """Weighted sentiment score from a {emotion: probability} mapping."""
    if not distribution:
        return 0.0
    total = sum(distribution.values()) or 1.0
    score = sum(VALENCE.get(emo, 0.0) * prob for emo, prob in distribution.items())
    return round(score / total, 4)


def score_from_label(emotion: str, confidence: float = 1.0) -> float:
    """Fallback score when only a single label is available (e.g. keyword hit).

    Blends the label's valence toward neutral by how confident we are.
    """
    valence = VALENCE.get(emotion, 0.0)
    return round(valence * max(0.0, min(1.0, confidence)), 4)


def mood_band(score: float) -> str:
    """Human-readable band for a wellbeing score (used by the dashboard)."""
    if score >= 0.5:
        return "Positive"
    if score >= 0.1:
        return "Stable"
    if score > -0.4:
        return "Low"
    if score > -0.8:
        return "Distressed"
    return "Crisis"
