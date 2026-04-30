import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import os

# 1. Using 'base' for better accuracy with VoIP/Zendesk quality
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

st.set_page_config(page_title="Zendesk AU Support Transcriber", page_icon="🎧")

st.title("🎧 Zendesk Call Transcriber (AU)")
st.write("Route your Zendesk audio through **VB-Cable** or **Stereo Mix** to transcribe.")

# 2. Capture audio
audio = mic_recorder(
    start_prompt="▶️ Start Recording Zendesk Playback",
    stop_prompt="⏹️ Stop & Transcribe",
    key='zendesk_recorder'
)

if audio:
    st.audio(audio['bytes'])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio['bytes'])
        tmp_path = tmp_file.name

    try:
        with st.spinner("Analyzing Zendesk recording..."):
            # 3. Optimized transcription for Support Context + Aussie Accent
            # initial_prompt: includes support-desk keywords to help the AI 'focus'
            result = model.transcribe(
                tmp_path, 
                fp16=False, 
                language="en",
                initial_prompt="This is a Zendesk customer support call in Australian English. Keywords: ticket, issue, account, email, support, Melbourne, Sydney."
            )
            
            st.subheader("Transcription:")
            st.info(result["text"])
            
            # 4. Download button for the support log
            st.download_button(
                label="📥 Download as Support Log",
                data=f"Zendesk Transcription:\n\n{result['text']}",
                file_name="zendesk_transcription.txt"
            )
            
    except Exception as e:
        st.error(f"Error processing call: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
