import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 1.5  # reduced from 2s — faster response

sd.default.device = 14

def record_audio():
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    audio = np.squeeze(audio)
    energy = np.mean(np.abs(audio))
    print(f"Audio Energy: {energy:.5f}")
    return audio