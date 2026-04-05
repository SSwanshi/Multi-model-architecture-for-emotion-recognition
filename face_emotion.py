import cv2
import numpy as np
from fer import FER

detector = FER(mtcnn=True)

_last_bbox = None
_frame_counter = 0
_last_emotion = "neutral"
_last_conf = 0.0
DETECT_EVERY_N = 5
MIN_CONFIDENCE = 0.35  # ignore detections below this


def detect_face_emotion(frame: np.ndarray):
    global _last_bbox, _frame_counter, _last_emotion, _last_conf
    _frame_counter += 1

    run_full_detect = (_frame_counter % DETECT_EVERY_N == 0) or (_last_bbox is None)

    if run_full_detect:
        result = detector.detect_emotions(frame)
        if not result:
            _last_bbox = None
            return _last_emotion, _last_conf  # return last known instead of neutral

        face = result[0]
        _last_bbox = face["box"]
        emotions = face["emotions"]
    else:
        if _last_bbox is None:
            return _last_emotion, _last_conf

        x, y, w, h = _last_bbox
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            _last_bbox = None
            return _last_emotion, _last_conf

        result = detector.detect_emotions(crop)
        if not result:
            return _last_emotion, _last_conf
        emotions = result[0]["emotions"]

    emotion = max(emotions, key=emotions.get)
    confidence = emotions[emotion]

    # Ignore low confidence — keep last result instead
    if confidence < MIN_CONFIDENCE:
        return _last_emotion, _last_conf

    _last_emotion = emotion
    _last_conf = round(confidence, 3)
    return _last_emotion, _last_conf