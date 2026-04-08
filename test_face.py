import cv2
from face_emotion import detect_face_emotion

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Face emotion test — make these expressions one by one")
print("Press Q to quit\n")

expressions = ["neutral", "happy (smile)", "angry (frown+squint)",
               "sad (frown)", "surprised (raised brows)"]
print("Try:", " → ".join(expressions))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion, conf = detect_face_emotion(frame)

    cv2.putText(frame, f"{emotion} ({conf:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)

    cv2.imshow("Face Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()