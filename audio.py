import streamlit as st
import soundcard as sc
import numpy as np
import whisper
import tempfile
import wave
import os

# Load the Whisper model (Base is fast and decent)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🔊 System Audio Transcriber")
st.write("Record what's playing on your computer and turn it into text.")

# Session state to handle recording status
if "recording" not in st.session_state:
    st.session_state.recording = False

# Setup loopback speaker (System Output)
speaker = sc.default_speaker()

def record_audio():
    fs = 44100  # Sample rate
    seconds = 10 # We'll record in chunks or a fixed duration for this demo
    
    with speaker.recorder(samplerate=fs) as mic:
        st.info("Recording system audio...")
        # Record audio data
        data = mic.record(numframes=fs * seconds)
        return data, fs

col1, col2 = st.columns(2)

if col1.button("Start 10s Recording"):
    # Note: For a true 'Start/Stop' toggle, you'd need a background thread.
    # This example captures a fixed 10-second snippet for stability.
    audio_data, samplerate = record_audio()
    
    # Save to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        file_path = tmp_file.name
        # soundcard returns float32, need to convert for WAV
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(samplerate)
            # Convert float32 to int16
            audio_int16 = (audio_data[:, 0] * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

    st.success("Recording complete! Transcribing...")

    # Transcribe using Whisper
    result = model.transcribe(file_path)
    
    st.subheader("Transcription:")
    st.write(result["text"])
    
    # Cleanup
    os.remove(file_path)

if col2.button("Clear"):
    st.rerun()