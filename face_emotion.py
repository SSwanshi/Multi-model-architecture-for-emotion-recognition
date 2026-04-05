import cv2
import numpy as np
from fer import FER

detector = FER(mtcnn=True)

# Cache state
_last_bbox = None          # (x, y, w, h) of last detected face
_frame_counter = 0
DETECT_EVERY_N = 5         # run full MTCNN detection only every 5 frames


def detect_face_emotion(frame: np.ndarray):
    global _last_bbox, _frame_counter
    _frame_counter += 1

    run_full_detect = (_frame_counter % DETECT_EVERY_N == 0) or (_last_bbox is None)

    if run_full_detect:
        result = detector.detect_emotions(frame)
        if not result:
            _last_bbox = None
            return "neutral", 0.0

        face = result[0]
        _last_bbox = face["box"]          # cache it
        emotions = face["emotions"]
    else:
        # Fast path: crop from cached bbox, run only the classifier (no MTCNN)
        if _last_bbox is None:
            return "neutral", 0.0

        x, y, w, h = _last_bbox
        # Add a small padding to handle slight head movement
        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            _last_bbox = None
            return "neutral", 0.0

        result = detector.detect_emotions(crop)
        if not result:
            return "neutral", 0.0
        emotions = result[0]["emotions"]

    emotion = max(emotions, key=emotions.get)
    confidence = emotions[emotion]
    return emotion, round(confidence, 3)