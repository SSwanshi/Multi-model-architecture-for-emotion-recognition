import cv2
import time
import threading
from collections import deque, Counter

from face_emotion import detect_face_emotion
from speech_emotion import predict_speech_emotion, get_ser_backend, USE_SPEECHBRAIN
from audio_stream import record_audio

# -------- AUDIO THREAD STATE --------
_audio_lock = threading.Lock()
_speech_emotion = "neutral"
_speech_conf = 0.0
_audio_running = True

AUDIO_INTERVAL = 0.2


def _audio_loop():
    """Runs in background thread — never blocks the video loop."""
    global _speech_emotion, _speech_conf
    while _audio_running:
        try:
            audio = record_audio()
            emotion, conf = predict_speech_emotion(audio)
            with _audio_lock:
                _speech_emotion = emotion
                _speech_conf = conf
            print(f"[Speech] {emotion} ({conf:.2f})")
        except Exception as e:
            import traceback
            print(f"Audio thread error: {e}")
            traceback.print_exc()
        time.sleep(AUDIO_INTERVAL)


# -------- SMOOTHING --------
emotion_buffer = deque(maxlen=7)


def smooth_emotion(emotion: str) -> str:
    emotion_buffer.append(emotion)
    return Counter(emotion_buffer).most_common(1)[0][0]


# -------- FUSION --------
FACE_WEIGHT = 0.55
SPEECH_WEIGHT = 0.45


def fuse_emotions(face_emotion, face_conf, speech_emotion, speech_conf):
    if face_emotion == speech_emotion:
        return face_emotion, max(face_conf, speech_conf)

    face_score = face_conf * FACE_WEIGHT
    speech_score = speech_conf * SPEECH_WEIGHT

    if face_score >= speech_score:
        return face_emotion, face_conf
    return speech_emotion, speech_conf


# -------- DISPLAY --------
EMOTION_COLORS = {
    "happy":     (0, 200, 100),
    "angry":     (0, 60, 220),
    "sad":       (180, 100, 30),
    "fearful":   (160, 0, 160),
    "disgusted": (0, 160, 160),
    "neutral":   (160, 160, 160),
    "calm":      (100, 200, 180),
    "surprise":  (0, 180, 220),
}


def get_color(emotion: str):
    return EMOTION_COLORS.get(emotion, (200, 200, 200))


def draw_overlay(frame, face_emotion, face_conf, speech_emotion, speech_conf,
                 final_emotion, final_conf, backend):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 125), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    fc = get_color(face_emotion)
    sc = get_color(speech_emotion)
    ec = get_color(final_emotion)

    cv2.putText(frame, f"Face:   {face_emotion:<12} {face_conf:.2f}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fc, 2)
    cv2.putText(frame, f"Speech: {speech_emotion:<12} {speech_conf:.2f}",
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)

    # Backend badge top-right
    badge = "SpeechBrain" if USE_SPEECHBRAIN else "MFCC"
    cv2.putText(frame, badge, (w - 130, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Final emotion bar
    label = f"  {final_emotion.upper()}  ({final_conf:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    cv2.rectangle(frame, (8, 68), (16 + tw, 108), ec, -1)
    cv2.putText(frame, label, (12, 97),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    return frame


# -------- MAIN LOOP --------
def run_multimodal():
    global _audio_running

    # Print backend once at startup
    print(f"Speech backend: {get_ser_backend()}")
    print("Multimodal Emotion System started — press Q to quit")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Start audio in background thread
    audio_thread = threading.Thread(target=_audio_loop, daemon=True)
    audio_thread.start()

    backend = get_ser_backend()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_emotion, face_conf = detect_face_emotion(frame)

        with _audio_lock:
            speech_emotion = _speech_emotion
            speech_conf = _speech_conf

        final_emotion, final_conf = fuse_emotions(
            face_emotion, face_conf,
            speech_emotion, speech_conf
        )
        final_emotion = smooth_emotion(final_emotion)

        frame = draw_overlay(
            frame,
            face_emotion, face_conf,
            speech_emotion, speech_conf,
            final_emotion, final_conf,
            backend
        )

        cv2.imshow("Multimodal Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    _audio_running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_multimodal()