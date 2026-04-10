# Emotion Multimodal Project

This project is a real-time multimodal emotion recognition system that combines Face, Speech, and Text analysis to determine the user's emotional state.

## Project Structure

- **Face Analysis**: `face_emotion.py`, `test_face.py`
- **Speech/Audio Analysis**: `audio_stream.py`, `speech_emotion.py`, `test_speech.py`, `test_wishper.py`, `test_mic.py`
- **Text/NLP Analysis**: `text_emotion.py`, `test_nlp_live.py`
- **Main/Integration**: `main_multimodal.py`, `test_integration.py`

## Prerequisites

- **Python 3.8 or higher** recommended.
- A functional webcam (for face emotion detection).
- A functional microphone (for speech/audio stream).

## Installation Guide

Follow these steps to set up the project on your machine.

### 1. Create a Virtual Environment (Recommended)
It's highly recommended to use a virtual environment to avoid dependency conflicts.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Required Packages
Install all the necessary Python dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` includes libraries like `opencv-python`, `fer`, `torch`, `librosa`, `numpy`, `sounddevice`, and `scipy`.*

### 3. Additional System Dependencies
Depending on your Operating System, the audio input library (`sounddevice`) might require additional system packages:
- **Windows / macOS:** Automatically handled by `pip` in most cases.
- **Linux (Debian/Ubuntu):** You may need to install `portaudio` before using the microphone.
  ```bash
  sudo apt-get install portaudio19-dev
  ```

### 4. Extra Dependencies (For Speech/Text/Whisper)
If you execute specific test scripts that use OpenAI's Whisper or transformers which are not fully listed in `requirements.txt`, you may also need to run:
```bash
pip install openai-whisper transformers
```

## Running the Application

Ensure your models (`pretrained_models/` and `wav2vec2_checkpoints/`) are present in the directory.

You can test individual modules:
- Face: `python test_face.py`
- Speech: `python test_speech.py`
- Mic Stream: `python test_mic.py`
- Text: `python test_nlp_live.py`

To run the full integrated application:
```bash
python main_multimodal.py
```