import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import os

# 1. Properly cache the model loading
@st.cache_resource
def load_whisper_model():
    # 'tiny' is best for Streamlit Cloud's memory limits
    return whisper.load_model("tiny")

model = load_whisper_model()

st.title("🎙️ Audio Transcriber")
st.write("Record your voice or system audio (via your mic) to transcribe.")

# 2. Browser-based recording
audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key='recorder'
)

if audio:
    # Display the audio player so you can hear what was recorded
    st.audio(audio['bytes'])
    
    # 3. Create a temporary file to hold the audio bytes
    # Whisper's transcribe function requires a file path string
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio['bytes'])
        tmp_path = tmp_file.name

    try:
        with st.spinner("Transcribing..."):
            # 4. Perform transcription
            # fp16=False prevents a common CPU-only warning on cloud servers
            result = model.transcribe(tmp_path, fp16=False)
            
            st.subheader("Transcription:")
            # Use a text area so users can easily copy the text
            st.text_area("Result:", value=result["text"], height=200)
            st.success("Done!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # 5. Ensure the temp file is deleted even if transcription fails
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
