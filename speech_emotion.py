import torch
import librosa
import numpy as np
from torch import nn


# -------- MODEL --------
class SERCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        return self.fc2(x)


# Load model
model = SERCNN(num_classes=7)
model.eval()


EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted"
]


# -------- AUDIO → SPECTROGRAM --------
def compute_mel_spectrogram_3ch(audio, sr=16000, n_mels=128):
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    max_len = 128

    if spec_db.shape[1] < max_len:
        pad_width = ((0, 0), (0, max_len - spec_db.shape[1]))
        spec_db = np.pad(spec_db, pad_width, mode="constant")
    else:
        spec_db = spec_db[:, :max_len]

    # Convert to 3 channel
    spec_3ch = np.repeat(spec_db[np.newaxis, :, :], 3, axis=0)

    return spec_3ch


# -------- PREDICTION --------
import numpy as np


EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted"
]


def predict_speech_emotion(audio_array):

    # Normalize audio
    audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-6)

    energy = np.mean(np.abs(audio_array))
    print("Energy:", energy)

    # ✅ Better calibrated thresholds
    if energy < 0.02:
        return "neutral", 0.6
    elif energy < 0.05:
        return "calm", 0.7
    elif energy < 0.10:
        return "happy", 0.8
    elif energy < 0.20:
        return "angry", 0.85
    else:
        return "angry", 0.9