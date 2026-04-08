import os
import numpy as np
import sounddevice as sd
import whisper
import webrtcvad
import collections

SAMPLE_RATE  = 16000
sd.default.device = 1

print("Loading Whisper model...")
_whisper_model = whisper.load_model("small")
print("Whisper loaded")

# VAD settings
VAD_AGGRESSIVENESS = 2        # 0=least aggressive, 3=most aggressive
VAD_FRAME_MS       = 30       # ms per VAD frame (10, 20, or 30 only)
VAD_FRAME_SAMPLES  = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples

MIN_SPEECH_FRAMES  = 5        # ignore clips shorter than this (~150ms)
MAX_SPEECH_SECONDS = 8        # hard cap — classify after 8s regardless
SILENCE_FRAMES     = 20       # frames of silence before considering speech done (~600ms)
PRE_ROLL_FRAMES    = 8        # frames to keep before speech starts (context)

_vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


def _is_speech(frame_bytes: bytes) -> bool:
    """Returns True if this 30ms frame contains speech."""
    try:
        return _vad.is_speech(frame_bytes, SAMPLE_RATE)
    except Exception:
        return False


def record_until_silence() -> tuple[np.ndarray, float]:
    """
    Records audio using a ring buffer + VAD.
    Starts collecting when speech is detected.
    Stops when silence follows speech.
    Returns (audio_array, duration_seconds).
    """
    # Ring buffer holds PRE_ROLL_FRAMES frames before speech starts
    ring_buffer  = collections.deque(maxlen=PRE_ROLL_FRAMES)
    voiced_frames = []

    in_speech        = False
    silence_count    = 0
    speech_frame_count = 0
    total_frames     = 0
    max_frames       = int(MAX_SPEECH_SECONDS * 1000 / VAD_FRAME_MS)

    print("Listening... (speak now)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",          # VAD needs int16
        blocksize=VAD_FRAME_SAMPLES,
    ) as stream:

        while total_frames < max_frames:
            frame_raw, _ = stream.read(VAD_FRAME_SAMPLES)
            frame_int16  = frame_raw[:, 0]            # shape (480,)
            frame_bytes  = frame_int16.tobytes()
            total_frames += 1

            is_speech = _is_speech(frame_bytes)

            if not in_speech:
                ring_buffer.append(frame_int16)
                if is_speech:
                    in_speech     = True
                    silence_count = 0
                    # Include pre-roll context
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                    print("Speech detected — recording...")
            else:
                voiced_frames.append(frame_int16)
                speech_frame_count += 1

                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1

                # Stop if enough silence after speech
                if (silence_count >= SILENCE_FRAMES and
                        speech_frame_count >= MIN_SPEECH_FRAMES):
                    print(f"Speech ended ({speech_frame_count * VAD_FRAME_MS / 1000:.1f}s)")
                    break

    if not voiced_frames:
        return np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32), 0.0

    # Combine all frames and convert to float32
    audio_int16 = np.concatenate(voiced_frames).astype(np.float32)
    audio_float = audio_int16 / 32768.0         # int16 → float32 [-1, 1]
    duration    = len(audio_float) / SAMPLE_RATE

    energy = float(np.mean(np.abs(audio_float)))
    print(f"Audio captured: {duration:.1f}s  energy={energy:.5f}")

    return audio_float, duration


def record_audio():
    """Backward-compatible wrapper — returns audio array only."""
    audio, _ = record_until_silence()
    return audio


def transcribe_audio(audio: np.ndarray) -> str:
    if audio is None or len(audio) == 0:
        return ""

    energy = float(np.mean(np.abs(audio)))
    if energy < 0.0001:
        return ""

    # Already float32 normalized from record_until_silence
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-6)

    try:
        result = _whisper_model.transcribe(
            audio_norm,
            fp16=False,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.3,
            logprob_threshold=-1.5,
        )
        transcript = result["text"].strip()

        # Filter Whisper hallucinations
        hallucinations = {
            "thank you", "thanks for watching", "bye", "you",
            ".", ",", "!", "the", "a", "thanks", "thank you.",
            "you.", "bye.", "goodbye", "see you"
        }
        if transcript.lower().strip(".!, ") in hallucinations:
            return ""
        if len(transcript) < 3:
            return ""

        return transcript

    except Exception as e:
        print(f"Whisper error: {e}")
        return ""