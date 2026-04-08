import cv2
import numpy as np
from fer import FER
from collections import Counter

detector = FER(mtcnn=True)

_last_bbox     = None
_frame_counter = 0
_last_emotion  = "neutral"
_last_conf     = 0.0
DETECT_EVERY_N = 5
MIN_CONFIDENCE = 0.35

# ---- Face buffer for averaging during speech window ----
_face_buffer: list[tuple[str, float]] = []
_buffering   = False   # True while audio is being recorded


def start_face_buffering():
    """Call when speech starts — begin collecting face emotions."""
    global _face_buffer, _buffering
    _face_buffer = []
    _buffering   = True


def stop_face_buffering() -> tuple[str, float]:
    """
    Call when speech ends — returns the dominant face emotion
    seen during the entire speech window.
    """
    global _buffering
    _buffering = False

    if not _face_buffer:
        return _last_emotion, _last_conf

    # Weighted majority vote — confidence is the weight
    scores: dict[str, float] = {}
    for emotion, conf in _face_buffer:
        scores[emotion] = scores.get(emotion, 0.0) + conf

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    conf  = round(scores[best] / total, 3) if total > 0 else 0.0
    return best, conf


def detect_face_emotion(frame: np.ndarray):
    global _last_bbox, _frame_counter, _last_emotion, _last_conf
    _frame_counter += 1

    run_full_detect = (
        (_frame_counter % DETECT_EVERY_N == 0) or
        (_last_bbox is None)
    )

    if run_full_detect:
        result = detector.detect_emotions(frame)
        if not result:
            _last_bbox = None
            return _last_emotion, _last_conf

        face     = result[0]
        _last_bbox = face["box"]
        emotions = face["emotions"]
    else:
        if _last_bbox is None:
            return _last_emotion, _last_conf

        x, y, w, h = _last_bbox
        pad = 20
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 48 or crop.shape[1] < 48:
            _last_bbox = None
            return _last_emotion, _last_conf

        result = detector.detect_emotions(crop)
        if not result:
            return _last_emotion, _last_conf
        emotions = result[0]["emotions"]

    emotion    = max(emotions, key=emotions.get)
    confidence = emotions[emotion]

    if confidence < MIN_CONFIDENCE:
        return _last_emotion, _last_conf

    _last_emotion = emotion
    _last_conf    = round(confidence, 3)

    # Buffer face emotion during speech window
    if _buffering:
        _face_buffer.append((_last_emotion, _last_conf))

    return _last_emotion, _last_conf