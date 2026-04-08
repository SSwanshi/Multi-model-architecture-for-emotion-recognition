import sounddevice as sd
import numpy as np
from audio_stream import transcribe_audio

SAMPLE_RATE = 16000
DURATION    = 3

print("Speak now for 3 seconds...")
audio = sd.rec(int(DURATION * SAMPLE_RATE),
               samplerate=SAMPLE_RATE,
               channels=1, dtype="float32")
sd.wait()
audio = np.squeeze(audio)

transcript = transcribe_audio(audio)
print(f"You said: '{transcript}'")