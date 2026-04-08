import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HTTPX_LOG_LEVEL"]        = "error"

import logging
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import cv2
import time
import threading
from collections import deque, Counter

from face_emotion   import detect_face_emotion, start_face_buffering, stop_face_buffering
from speech_emotion import predict_speech_emotion, get_ser_backend, USE_SPEECHBRAIN
from audio_stream   import record_audio, transcribe_audio, record_until_silence
from text_emotion   import predict_text_emotion

# -------- SHARED STATE --------
_audio_lock      = threading.Lock()
_speech_emotion  = "neutral"
_speech_conf     = 0.0
_text_emotion    = "neutral"
_text_conf       = 0.0
_face_emotion    = "neutral"   # face averaged over speech window
_face_conf       = 0.0
_last_transcript = ""
_audio_running   = True
_system_ready    = False       # True after first speech cycle completes


def _audio_loop():
    global _speech_emotion, _speech_conf
    global _text_emotion, _text_conf
    global _face_emotion, _face_conf
    global _last_transcript, _system_ready

    while _audio_running:
        try:
            # Signal face module to start buffering
            start_face_buffering()

            # Block here until user finishes speaking
            audio, duration = record_until_silence()

            # Get face emotion averaged over the speech window
            face_e, face_c = stop_face_buffering()

            if duration < 0.3:
                # Too short — likely silence, skip
                time.sleep(0.1)
                continue

            # Speech emotion (acoustic)
            s_emotion, s_conf = predict_speech_emotion(audio)

            # Transcribe + NLP
            transcript = transcribe_audio(audio)
            if transcript:
                print(f"  [Whisper] '{transcript}'")
                t_emotion, t_conf, shown = predict_text_emotion(transcript)
            else:
                t_emotion, t_conf, shown = "neutral", 0.0, ""

            with _audio_lock:
                _speech_emotion  = s_emotion
                _speech_conf     = s_conf
                _text_emotion    = t_emotion
                _text_conf       = t_conf
                _face_emotion    = face_e
                _face_conf       = face_c
                _last_transcript = shown
                _system_ready    = True

            print(f"  [Face/window] {face_e} ({face_c:.2f}) | "
                  f"[Speech] {s_emotion} ({s_conf:.2f}) | "
                  f"[NLP] {t_emotion} ({t_conf:.2f})")

        except Exception as e:
            import traceback
            print(f"Audio thread error: {e}")
            traceback.print_exc()
            stop_face_buffering()
            time.sleep(0.5)


# -------- SMOOTHING --------
emotion_buffer = deque(maxlen=5)

def smooth_emotion(emotion: str) -> str:
    emotion_buffer.append(emotion)
    return Counter(emotion_buffer).most_common(1)[0][0]


# -------- 3-WAY FUSION --------
def fuse_emotions(face_emotion, face_conf,
                  speech_emotion, speech_conf,
                  text_emotion, text_conf):
    scores: dict[str, float] = {}

    def add(emotion, conf, weight):
        scores[emotion] = scores.get(emotion, 0.0) + conf * weight

    # Priority: NLP (0.50) > Face (0.30) > Speech (0.20)
    add(text_emotion,   text_conf,   0.50)
    add(face_emotion,   face_conf,   0.30)
    add(speech_emotion, speech_conf, 0.20)

    # Extra boost if NLP is very confident (clear keyword match)
    if text_conf > 0.7:
        add(text_emotion, text_conf, 0.15)

    # Boost if NLP and face agree
    if text_emotion == face_emotion:
        add(text_emotion, (text_conf + face_conf) / 2, 0.10)

    best_emotion = max(scores, key=scores.get)
    best_conf    = round(min(scores[best_emotion], 1.0), 3)
    return best_emotion, best_conf

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


def draw_overlay(frame,
                 face_emotion, face_conf,
                 speech_emotion, speech_conf,
                 text_emotion, text_conf,
                 transcript,
                 final_emotion, final_conf,
                 system_ready):

    h, w  = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 165), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if not system_ready:
        cv2.putText(frame, "Listening... speak a sentence",
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)
        return frame

    fc = get_color(face_emotion)
    sc = get_color(speech_emotion)
    tc = get_color(text_emotion)
    ec = get_color(final_emotion)

    cv2.putText(frame,
                f"Face   (window): {face_emotion:<12} {face_conf:.2f}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, fc, 2)
    cv2.putText(frame,
                f"Speech (acoustic): {speech_emotion:<12} {speech_conf:.2f}",
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58, sc, 2)
    cv2.putText(frame,
                f"NLP    (words):  {text_emotion:<12} {text_conf:.2f}",
                (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, tc, 2)

    # Transcript
    shown = transcript[:58] + "..." if len(transcript) > 58 else transcript
    cv2.putText(frame, f"\"{shown}\"",
                (12, 108), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (200, 200, 200), 1)

    # Final emotion bar
    label = f"  {final_emotion.upper()}  ({final_conf:.2f})"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    cv2.rectangle(frame, (8, 118), (16 + tw, 158), ec, -1)
    cv2.putText(frame, label, (12, 148),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    # Backend + listening indicator
    badge = "SpeechBrain" if USE_SPEECHBRAIN else "MFCC"
    cv2.putText(frame, badge,
                (w - 130, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


# -------- MAIN LOOP --------
def run_multimodal():
    global _audio_running

    print(f"Speech backend : {get_ser_backend()}")
    print("Starting system — speak a complete sentence, pause, repeat")
    print("Press Q to quit")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    audio_thread = threading.Thread(target=_audio_loop, daemon=True)
    audio_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Live face detection (still runs every frame for display)
        live_face_emotion, live_face_conf = detect_face_emotion(frame)

        # Read latest synced results
        with _audio_lock:
            speech_emotion  = _speech_emotion
            speech_conf     = _speech_conf
            text_emotion    = _text_emotion
            text_conf       = _text_conf
            # Use window-averaged face for fusion, live face for display
            window_face_e   = _face_emotion
            window_face_c   = _face_conf
            transcript      = _last_transcript
            ready           = _system_ready

        # Fuse using window-averaged face (synced with speech)
        final_emotion, final_conf = fuse_emotions(
            window_face_e,  window_face_c,
            speech_emotion, speech_conf,
            text_emotion,   text_conf,
        )
        final_emotion = smooth_emotion(final_emotion)

        frame = draw_overlay(
            frame,
            window_face_e,  window_face_c,
            speech_emotion, speech_conf,
            text_emotion,   text_conf,
            transcript,
            final_emotion,  final_conf,
            ready,
        )

        # Show live face emotion in corner separately
        cv2.putText(frame,
                    f"Live face: {live_face_emotion}",
                    (12, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, get_color(live_face_emotion), 1)

        cv2.imshow("Multimodal Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    _audio_running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_multimodal()