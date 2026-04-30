import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import os

# 1. Load the model (Tiny is fastest for Windows/Cloud testing)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

model = load_whisper_model()

st.set_page_config(page_title="Windows System Audio Transcriber", page_icon="🔊")

st.title("🔊 Windows System Audio Transcriber")
st.markdown("""
### 🛠️ Setup Instructions for Windows:
1. Enable **Stereo Mix** in your Windows Sound Control Panel.
2. Set **Stereo Mix** as your default recording device.
3. Use the recorder below. The browser will capture whatever is playing on your PC.
""")

# 2. Capture audio from the browser
# mic_recorder captures the default recording device (Stereo Mix)
audio = mic_recorder(
    start_prompt="▶️ Start Recording System Sound",
    stop_prompt="⏹️ Stop & Transcribe",
    key='windows_recorder'
)

if audio:
    # Display the captured audio to verify it worked
    st.audio(audio['bytes'])
    
    # 3. Create temp file for Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio['bytes'])
        tmp_path = tmp_file.name

    try:
        with st.spinner("Whisper is listening to the recording..."):
            # Transcribe (fp16=False ensures it runs on CPU without errors)
            result = model.transcribe(tmp_path, fp16=False)
            
            st.subheader("Transcription:")
            # Display result in a nice box
            st.text_area("Final Text", value=result["text"], height=250)
            
            # 4. Download Option
            st.download_button(
                label="📥 Download Transcription (.txt)",
                data=result["text"],
                file_name="system_audio_transcription.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"Transcription failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
