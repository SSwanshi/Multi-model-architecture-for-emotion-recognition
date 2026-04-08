import re
from transformers import pipeline

# -------- PRETRAINED TEXT EMOTION MODEL --------
# Trained on ~200k English texts, recognizes 7 emotions
try:
    _text_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None  # return scores for ALL emotions, not just top 1
    )
    USE_TEXT_MODEL = True
    print("Text emotion model loaded successfully")
except Exception as e:
    USE_TEXT_MODEL = False
    print(f"Text model not available ({e}), using keyword fallback")


# -------- EMOTION KEYWORD DICTIONARY --------
# Maps words/phrases to emotions with weights
# Higher weight = stronger signal for that emotion

EMOTION_KEYWORDS = {
    "happy": [
        # Direct
        ("happy", 1.0), ("happiness", 1.0), ("joy", 1.0), ("joyful", 1.0),
        ("excited", 0.9), ("exciting", 0.8), ("excitement", 0.9),
        ("amazing", 0.8), ("wonderful", 0.8), ("fantastic", 0.9),
        ("great", 0.7), ("awesome", 0.8), ("excellent", 0.7),
        ("love", 0.7), ("loving", 0.7), ("loved", 0.7),
        ("glad", 0.8), ("pleased", 0.7), ("delighted", 0.9),
        ("cheerful", 0.8), ("thrilled", 0.9), ("ecstatic", 1.0),
        ("grateful", 0.7), ("thankful", 0.6), ("blessed", 0.7),
        ("fun", 0.6), ("enjoy", 0.7), ("enjoying", 0.7),
        ("smile", 0.6), ("laughing", 0.7), ("laugh", 0.6),
        ("celebrate", 0.8), ("celebration", 0.8), ("party", 0.5),
        ("yay", 1.0), ("woohoo", 1.0), ("hurray", 1.0),
    ],
    "sad": [
        ("sad", 1.0), ("sadness", 1.0), ("unhappy", 0.9),
        ("disappointed", 0.9), ("disappointing", 0.8), ("disappointment", 0.9),
        ("depressed", 1.0), ("depression", 1.0),
        ("miserable", 1.0), ("hopeless", 0.9), ("helpless", 0.8),
        ("cry", 0.8), ("crying", 0.9), ("tears", 0.7),
        ("grief", 1.0), ("grieve", 0.9), ("grieving", 0.9),
        ("heartbroken", 1.0), ("broken", 0.6),
        ("lonely", 0.9), ("alone", 0.6), ("isolated", 0.8),
        ("miss", 0.6), ("missing", 0.7), ("loss", 0.7),
        ("terrible", 0.7), ("horrible", 0.7), ("awful", 0.7),
        ("hurt", 0.7), ("pain", 0.7), ("painful", 0.8),
        ("tired", 0.5), ("exhausted", 0.6), ("drained", 0.7),
        ("worthless", 0.9), ("useless", 0.8),
    ],
    "angry": [
        ("angry", 1.0), ("anger", 1.0), ("mad", 0.9),
        ("furious", 1.0), ("rage", 1.0), ("outraged", 1.0),
        ("frustrated", 0.8), ("frustrating", 0.7), ("frustration", 0.8),
        ("annoyed", 0.8), ("annoying", 0.7), ("irritated", 0.8),
        ("hate", 0.9), ("hating", 0.9), ("hatred", 1.0),
        ("disgusting", 0.8), ("disgusted", 0.8),
        ("stupid", 0.7), ("idiot", 0.8), ("ridiculous", 0.7),
        ("unfair", 0.7), ("wrong", 0.5), ("betrayed", 0.9),
        ("offensive", 0.7), ("rude", 0.7),
        ("shut up", 1.0), ("stop it", 0.7),
    ],
    "fearful": [
        ("scared", 1.0), ("fear", 1.0), ("fearful", 1.0),
        ("afraid", 0.9), ("terrified", 1.0), ("terror", 1.0),
        ("panic", 1.0), ("panicking", 1.0), ("panicked", 1.0),
        ("anxious", 0.8), ("anxiety", 0.9), ("nervous", 0.7),
        ("worried", 0.8), ("worry", 0.7), ("worrying", 0.7),
        ("dread", 0.9), ("dreading", 0.9),
        ("horror", 0.9), ("horrified", 1.0),
        ("unsafe", 0.8), ("danger", 0.8), ("dangerous", 0.7),
        ("threat", 0.7), ("threatened", 0.8),
    ],
    "neutral": [
        ("okay", 0.6), ("fine", 0.5), ("alright", 0.5),
        ("normal", 0.6), ("usual", 0.5), ("regular", 0.5),
        ("whatever", 0.6), ("nothing", 0.5),
    ],
    "calm": [
        ("calm", 1.0), ("peaceful", 1.0), ("relaxed", 0.9),
        ("relaxing", 0.8), ("serene", 1.0), ("tranquil", 1.0),
        ("comfortable", 0.7), ("content", 0.8), ("satisfied", 0.7),
        ("quiet", 0.6), ("still", 0.5), ("gentle", 0.6),
        ("breathing", 0.5), ("meditate", 0.7), ("meditation", 0.8),
    ],
    "disgusted": [
        ("disgusting", 1.0), ("disgusted", 1.0), ("disgust", 1.0),
        ("gross", 0.9), ("nasty", 0.8), ("revolting", 1.0),
        ("sick", 0.7), ("yuck", 1.0), ("eww", 1.0),
        ("awful", 0.6), ("repulsive", 1.0),
    ],
}

# Build reverse lookup: word → (emotion, weight)
_WORD_MAP: dict[str, list[tuple[str, float]]] = {}
for emotion, pairs in EMOTION_KEYWORDS.items():
    for word, weight in pairs:
        if word not in _WORD_MAP:
            _WORD_MAP[word] = []
        _WORD_MAP[word].append((emotion, weight))


# -------- NEGATION HANDLING --------
NEGATIONS = {"not", "no", "never", "don't", "doesn't", "didn't",
             "won't", "can't", "isn't", "aren't", "wasn't"}

NEGATION_FLIP = {
    "happy":    "sad",
    "sad":      "happy",
    "angry":    "calm",
    "calm":     "angry",
    "fearful":  "neutral",
    "neutral":  "neutral",
    "disgusted":"neutral",
}


def _keyword_score(text: str) -> dict[str, float]:
    """
    Scans text for emotion keywords with negation awareness.
    Returns a score dict e.g. {"happy": 0.8, "sad": 0.3, ...}
    """
    text  = text.lower().strip()
    words = re.findall(r"[\w']+", text)
    scores: dict[str, float] = {e: 0.0 for e in EMOTION_KEYWORDS}

    i = 0
    while i < len(words):
        word = words[i]

        # Check for multi-word phrases first (e.g. "shut up")
        if i + 1 < len(words):
            phrase = word + " " + words[i + 1]
            if phrase in _WORD_MAP:
                # Check negation in window of 3 words before phrase
                window = words[max(0, i - 3): i]
                negated = any(w in NEGATIONS for w in window)
                for emotion, weight in _WORD_MAP[phrase]:
                    if negated:
                        flipped = NEGATION_FLIP.get(emotion, emotion)
                        scores[flipped] = max(scores[flipped], weight * 0.8)
                    else:
                        scores[emotion] = max(scores[emotion], weight)
                i += 2
                continue

        # Single word
        if word in _WORD_MAP:
            window = words[max(0, i - 3): i]
            negated = any(w in NEGATIONS for w in window)
            for emotion, weight in _WORD_MAP[word]:
                if negated:
                    flipped = NEGATION_FLIP.get(emotion, emotion)
                    scores[flipped] = max(scores[flipped], weight * 0.8)
                else:
                    scores[emotion] = max(scores[emotion], weight)
        i += 1

    return scores


# -------- LABEL MAP --------
# Maps HuggingFace model labels to your emotion set
_HF_LABEL_MAP = {
    "joy":      "happy",
    "sadness":  "sad",
    "anger":    "angry",
    "fear":     "fearful",
    "disgust":  "disgusted",
    "surprise": "happy",   # treat surprise as happy-adjacent
    "neutral":  "neutral",
}


# -------- MAIN FUNCTION --------
def predict_text_emotion(text: str):
    """
    Combines pretrained model + keyword matching for best results.
    Returns (emotion, confidence, transcript_shown)
    """
    if not text or not text.strip():
        return "neutral", 0.0, ""

    text = text.strip()
    print(f"  [NLP] transcript: '{text}'")

    # Step 1: keyword scores (always run)
    kw_scores = _keyword_score(text)
    kw_best   = max(kw_scores, key=kw_scores.get)
    kw_conf   = kw_scores[kw_best]

    # Step 2: model scores (if available)
    if USE_TEXT_MODEL:
        try:
            raw = _text_classifier(text)[0]  # list of {label, score}
            model_scores = {
                _HF_LABEL_MAP.get(r["label"].lower(), "neutral"): r["score"]
                for r in raw
            }
            # Merge: keyword match boosts model score
            final_scores = {}
            for emotion in EMOTION_KEYWORDS:
                model_s = model_scores.get(emotion, 0.0)
                kw_s    = kw_scores.get(emotion, 0.0)
                # Keyword is strong signal — weight it heavily when present
                if kw_s > 0.5:
                    final_scores[emotion] = 0.4 * model_s + 0.6 * kw_s
                else:
                    final_scores[emotion] = model_s

            best_emotion = max(final_scores, key=final_scores.get)
            best_conf    = round(min(final_scores[best_emotion], 1.0), 3)
            print(f"  [NLP] model+keywords → {best_emotion} ({best_conf:.2f})")
            return best_emotion, best_conf, text

        except Exception as e:
            print(f"  [NLP] model error ({e}), using keywords only")

    # Fallback: keywords only
    if kw_conf > 0.0:
        print(f"  [NLP] keywords only → {kw_best} ({kw_conf:.2f})")
        return kw_best, round(kw_conf, 3), text

    return "neutral", 0.3, text


# -------- TEST --------
if __name__ == "__main__":
    tests = [
        "I am so excited about this",
        "I feel really disappointed today",
        "I am not happy at all",
        "This is making me so angry",
        "I feel calm and peaceful",
        "I am terrified of what might happen",
        "everything is fine I guess",
    ]
    for t in tests:
        emotion, conf, _ = predict_text_emotion(t)
        print(f"'{t}' → {emotion} ({conf:.2f})\n")