"""
probe_model.py — tests a wide range of phrases against the emotion classifier
and prints what the model actually detects.
Run from the backend/ folder: python ../probe_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from model import classify_text

candidates = {
    "ANXIETY (English)": [
        "I can't stop worrying about everything",
        "I feel so nervous all the time",
        "I have a panic attack almost every day",
        "My heart races and I can't breathe properly",
        "I'm terrified of what might happen",
        "I feel restless and can't sit still",
        "Everything feels out of control and I'm scared",
        "I keep overthinking and I can't calm down",
        "I have this constant feeling of dread",
        "I'm anxious about the future",
    ],
    "ANXIETY (Malay)": [
        "Saya tidak boleh berhenti risau tentang semua perkara",
        "Jantung saya berdegup kencang dan saya rasa takut",
        "Saya rasa gelisah sepanjang masa",
        "Saya selalu rasa bimbang",
        "Saya takut sesuatu yang buruk akan berlaku",
    ],
    "DEPRESSION (English)": [
        "I feel completely empty inside",
        "Nothing brings me joy anymore",
        "I can't get out of bed most days",
        "I feel like a burden to everyone around me",
        "I have no energy and no motivation",
        "Everything feels hopeless",
        "I've been crying for no reason",
        "I feel numb and disconnected from life",
        "I don't see the point in anything",
        "I've lost interest in things I used to love",
    ],
    "DEPRESSION (Malay)": [
        "Saya rasa sangat sedih dan tidak berdaya",
        "Tiada apa yang boleh buat saya gembira",
        "Saya rasa sunyi dan kosong",
        "Saya tidak ada semangat langsung",
        "Hidup saya rasa tidak bermakna",
    ],
    "SUICIDAL (English)": [
        "I want to end my life",
        "I don't want to be alive anymore",
        "I've been thinking about killing myself",
        "I feel like everyone would be better off without me",
        "I have a plan to hurt myself",
        "I can't see any reason to keep going",
        "Life is not worth living",
        "I want to disappear forever",
    ],
    "SUICIDAL (Malay)": [
        "Saya nak bunuh diri",
        "Saya rasa lebih baik saya mati",
        "Saya tak nak hidup lagi",
        "Saya rasa semua orang lebih baik tanpa saya",
        "Saya nak hilang dari dunia ini",
    ],
    "NORMAL (English)": [
        "I had a pretty good day today",
        "I'm feeling okay overall",
        "Things are going fine at work",
        "I just wanted to check in",
        "I feel neutral right now",
    ],
}

results = {}
for category, phrases in candidates.items():
    results[category] = []
    for phrase in phrases:
        emotion, confidence, is_high_risk = classify_text(phrase)
        results[category].append((phrase, emotion, confidence, is_high_risk))

# Print results
for category, rows in results.items():
    expected = category.split(" ")[0]
    print(f"\n{'='*65}")
    print(f"  Expected: {category}")
    print(f"{'='*65}")
    for phrase, emotion, confidence, is_high_risk in rows:
        match = "[OK]   " if emotion == expected else "[MISS] "
        risk  = " CRISIS" if is_high_risk else ""
        print(f"  {match} {emotion:<12} {confidence:.0%}  \"{phrase[:50]}\"")
