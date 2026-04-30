import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import os

# Load Whisper
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎙️ Audio Transcriber")
st.write("Record your voice or system audio (via your mic) to transcribe.")

# This creates a recording button in the browser
audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key='recorder'
)

if audio:
    st.audio(audio['bytes'])
    
    # Save the recorded bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio['bytes'])
        tmp_path = tmp_file.name

    with st.spinner("Transcribing..."):
        # Transcribe
        result = model.transcribe(tmp_path)
        st.subheader("Transcription:")
        st.success(result["text"])

    # Cleanup
    os.remove(tmp_path)
