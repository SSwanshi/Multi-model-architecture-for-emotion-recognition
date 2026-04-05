import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 2

# ✅ Use Airdopes mic
sd.default.device = 1


def record_audio():
    print("🎤 Recording audio...")

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)

    sd.wait()

    audio = np.squeeze(audio)

    energy = np.mean(np.abs(audio))
    print("Audio Energy:", energy)

    return audio