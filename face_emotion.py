from fer import FER

# Load pretrained FER model
detector = FER(mtcnn=True)


def detect_face_emotion(frame):
    result = detector.detect_emotions(frame)

    if len(result) == 0:
        return "neutral", 0.0

    emotions = result[0]["emotions"]
    emotion = max(emotions, key=emotions.get)
    confidence = emotions[emotion]

    return emotion, confidence