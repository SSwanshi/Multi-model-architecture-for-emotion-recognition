import sounddevice as sd
import numpy as np
from audio_stream   import transcribe_audio
from text_emotion   import predict_text_emotion

SAMPLE_RATE = 16000
DURATION    = 3

test_phrases = [
    "I am so excited and happy today",
    "I feel really sad and disappointed",
    "This is making me so angry and frustrated",
    "I feel calm and relaxed",
    "I am terrified and scared",
]

print("=== Automated phrase test ===")
print("Speak each phrase when prompted\n")

for phrase in test_phrases:
    input(f"Press Enter then say: '{phrase}'")
    
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)

    transcript           = transcribe_audio(audio)
    emotion, conf, shown = predict_text_emotion(transcript)

    print(f"  Expected : {phrase}")
    print(f"  Heard    : '{transcript}'")
    print(f"  Detected : {emotion} ({conf:.2f})")
    match = "✓" if any(w in transcript.lower() 
                       for w in phrase.lower().split()) else "✗"
    print(f"  Whisper  : {match}\n")