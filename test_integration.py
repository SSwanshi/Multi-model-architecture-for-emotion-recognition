import sounddevice as sd
import numpy as np
import cv2
import threading

from audio_stream   import record_audio, transcribe_audio
from speech_emotion import predict_speech_emotion
from text_emotion   import predict_text_emotion
from face_emotion   import detect_face_emotion

SAMPLE_RATE = 16000

def test_single_cycle():
    """Simulates one complete cycle of the main loop."""
    print("\n" + "="*50)
    print("INTEGRATION TEST — one full cycle")
    print("="*50)

    # 1. Capture face
    cap   = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()

    if ret:
        face_emotion, face_conf = detect_face_emotion(frame)
        print(f"Face   : {face_emotion} ({face_conf:.2f})")
    else:
        print("Face   : FAILED - no frame captured")

    # 2. Capture audio
    print("Speak now for 2 seconds...")
    audio = sd.rec(int(2 * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)

    # 3. Speech emotion
    speech_emotion, speech_conf = predict_speech_emotion(audio)
    print(f"Speech : {speech_emotion} ({speech_conf:.2f})")

    # 4. Transcribe + NLP
    transcript             = transcribe_audio(audio)
    text_emotion, text_conf, _ = predict_text_emotion(transcript)
    print(f"Whisper: '{transcript}'")
    print(f"NLP    : {text_emotion} ({text_conf:.2f})")

    # 5. Manual fusion check
    print("\nAll modalities:")
    print(f"  Face   → {face_emotion:<12} conf={face_conf:.2f}  weight=0.40")
    print(f"  Speech → {speech_emotion:<12} conf={speech_conf:.2f}  weight=0.25")
    print(f"  NLP    → {text_emotion:<12} conf={text_conf:.2f}  weight=0.35")

    # Who wins
    scores = {
        face_emotion:   face_conf   * 0.40,
        speech_emotion: speech_conf * 0.25,
        text_emotion:   text_conf   * 0.35,
    }
    winner = max(scores, key=scores.get)
    print(f"\n  Fusion result → {winner}")


if __name__ == "__main__":
    test_single_cycle()