import numpy as np
from speech_emotion import predict_speech_emotion, get_ser_backend

print(f"Backend: {get_ser_backend()}\n")

# Simulate different audio profiles
def make_audio(energy_level, sr=16000, duration=1.5):
    """Generate synthetic audio at different energy levels."""
    samples = int(sr * duration)
    audio   = np.random.randn(samples).astype(np.float32)
    return audio * energy_level

tests = [
    ("silence",      0.00001),
    ("whisper",      0.001),
    ("normal speech",0.005),
    ("loud speech",  0.02),
    ("shouting",     0.05),
]

for label, energy in tests:
    audio            = make_audio(energy)
    emotion, conf    = predict_speech_emotion(audio)
    print(f"{label:<16} energy={energy:.5f} → {emotion} ({conf:.2f})")