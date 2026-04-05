import numpy as np
import torch

# -------- SPEECHBRAIN SETUP --------
import sys
from unittest.mock import MagicMock
sys.modules["k2"] = MagicMock()

try:
    from speechbrain.inference.classifiers import EncoderClassifier
    _sb_model = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion"
    )
    USE_SPEECHBRAIN = True
    print("SpeechBrain loaded successfully")
except Exception as e:
    USE_SPEECHBRAIN = False
    print(f"SpeechBrain not available ({e}), using numpy fallback")

EMOTIONS_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
}

# -------- BACKEND INFO --------
def get_ser_backend():
    if USE_SPEECHBRAIN:
        return "SpeechBrain (wav2vec2 pretrained)"
    else:
        return "Numpy fallback (no librosa)"


# -------- PURE NUMPY FEATURES --------

def _compute_energy(audio: np.ndarray) -> float:
    return float(np.mean(audio ** 2))

def _compute_zcr(audio: np.ndarray) -> float:
    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.diff(signs)
    return float(np.mean(np.abs(crossings) > 0))

def _compute_speech_rate(audio: np.ndarray, sr: int = 16000) -> float:
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    frames = [audio[i: i + frame_len]
              for i in range(0, len(audio) - frame_len, hop)]
    if not frames:
        return 0.0
    rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    threshold = np.mean(rms) + 0.5 * np.std(rms)
    peaks, in_peak = 0, False
    for val in rms:
        if val > threshold and not in_peak:
            peaks += 1
            in_peak = True
        elif val <= threshold:
            in_peak = False
    return peaks / (len(audio) / sr) if len(audio) > 0 else 0.0

def _compute_energy_variance(audio: np.ndarray, sr: int = 16000) -> float:
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    frames = [audio[i: i + frame_len]
              for i in range(0, len(audio) - frame_len, hop)]
    if not frames:
        return 0.0
    rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    return float(np.std(rms))


# -------- CLASSIFIER --------

def _classify(audio: np.ndarray, sr: int = 16000):
    """
    audio — original unmodified samples straight from the mic.
    All loudness decisions use the REAL amplitude, not normalized.
    """
    # Real loudness — this is what Audio Energy prints in audio_stream.py
    real_energy = float(np.mean(np.abs(audio)))

    # For rate/zcr/var we normalize so features are scale-independent
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-6)
    zcr         = _compute_zcr(audio_norm)
    speech_rate = _compute_speech_rate(audio_norm, sr)
    energy_var  = _compute_energy_variance(audio_norm, sr)

    print(f"  [Features] real_energy={real_energy:.5f} zcr={zcr:.3f} "
          f"rate={speech_rate:.1f}/s var={energy_var:.4f}")

    # Silence — matches "Audio Energy: 0.00001" readings
    if real_energy < 0.0005:
        return "neutral", 0.55

    # Angry — your loudest was Audio Energy ~0.02+
    if real_energy > 0.015:
        return "angry", 0.82

    # Happy — fast speech: energy_var on normalized audio is reliable
    # Fast speech = more energy bursts = higher variance
    # Calm/sad = steady = low variance
    if energy_var > 0.25 and real_energy > 0.001:
        return "happy", 0.72
    if speech_rate > 3 and real_energy > 0.002:
        return "happy", 0.68

    # Sad — very quiet, slow
    if real_energy < 0.002 and speech_rate < 1.5:
        return "sad", 0.70

    # Calm — quiet but present
    if real_energy < 0.005 and energy_var < 0.20:
        return "calm", 0.62

    return "neutral", 0.55


# -------- MAIN PREDICTION --------

def predict_speech_emotion(audio_array: np.ndarray, sr: int = 16000):
    if audio_array is None or len(audio_array) == 0:
        return "neutral", 0.0

    # Hard silence gate using same metric as audio_stream.py prints
    real_energy = float(np.mean(np.abs(audio_array)))
    if real_energy < 0.0001:
        print("  [Features] Silence detected")
        return "neutral", 0.55

    if USE_SPEECHBRAIN:
        try:
            audio_norm = audio_array / (np.max(np.abs(audio_array)) + 1e-6)
            wav_tensor = torch.tensor(audio_norm, dtype=torch.float32).unsqueeze(0)
            out = _sb_model.classify_batch(wav_tensor)
            label = out[3][0]
            score = float(out[1].exp().max())
            emotion = EMOTIONS_MAP.get(label, label)
            return emotion, round(score, 3)
        except Exception as e:
            print(f"SpeechBrain inference error: {e}, using fallback")

    # Pass original unmodified array — no normalization before _classify
    return _classify(audio_array, sr)


# -------- TEST --------
if __name__ == "__main__":
    print(f"USE_SPEECHBRAIN: {USE_SPEECHBRAIN}")
    print(f"Backend: {get_ser_backend()}")
    dummy = np.random.randn(16000).astype(np.float32) * 0.1
    print(f"Test prediction: {predict_speech_emotion(dummy)}")