# test_mic.py
import sounddevice as sd
import numpy as np

print("Available devices:")
print(sd.query_devices())

sd.default.device = 1
print("\nRecording 3 seconds — speak normally...")
audio = sd.rec(3 * 16000, samplerate=16000, channels=1, dtype="float32")
sd.wait()
audio = np.squeeze(audio)

print(f"Max amplitude : {np.max(np.abs(audio)):.5f}")
print(f"Mean energy   : {np.mean(audio**2):.5f}")
print(f"RMS            : {np.sqrt(np.mean(audio**2)):.5f}")