import cv2
import time
from collections import deque, Counter

from face_emotion import detect_face_emotion
from speech_emotion import predict_speech_emotion
from audio_stream import record_audio


# -------- SMOOTHING --------
emotion_buffer = deque(maxlen=10)


def smooth_emotion(emotion):
    emotion_buffer.append(emotion)
    return Counter(emotion_buffer).most_common(1)[0][0]


# -------- FUSION --------
def fuse_emotions(face_emotion, face_conf, speech_emotion, speech_conf):

    # If both agree → strong result
    if face_emotion == speech_emotion:
        return face_emotion, max(face_conf, speech_conf)

    # Otherwise weighted decision
    if face_conf >= speech_conf:
        return face_emotion, face_conf
    else:
        return speech_emotion, speech_conf


def run_multimodal():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 10)

    print("🚀 Multimodal Emotion System Started")
    print("Press 'q' to exit")

    last_audio_time = 0
    speech_emotion = "neutral"
    speech_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        frame = cv2.resize(frame, (640, 480))

        # -------- FACE --------
        face_emotion, face_conf = detect_face_emotion(frame)

        # -------- SPEECH --------
        if time.time() - last_audio_time > 2:
            audio = record_audio()
            speech_emotion, speech_conf = predict_speech_emotion(audio)
            last_audio_time = time.time()

        # -------- FUSION --------
        final_emotion, final_conf = fuse_emotions(
            face_emotion, face_conf,
            speech_emotion, speech_conf
        )

        # -------- SMOOTH --------
        final_emotion = smooth_emotion(final_emotion)

        # -------- DEBUG PRINT --------
        print(f"Face: {face_emotion}, Speech: {speech_emotion}, Final: {final_emotion}")

        # -------- DISPLAY --------
        cv2.putText(frame, f"Face: {face_emotion} ({face_conf:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Speech: {speech_emotion} ({speech_conf:.2f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"FINAL: {final_emotion} ({final_conf:.2f})",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow("Multimodal Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_multimodal()