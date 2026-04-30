import streamlit as st
from streamlit_mic_recorder import mic_recorder
import assemblyai as aai
import tempfile
import os

# Set your API Key here
aai.settings.api_key = "YOUR_ASSEMBLYAI_API_KEY"

st.set_page_config(page_title="High-Accuracy AU Transcriber", page_icon="🦘")

st.title("🦘 AssemblyAI: Aussie Zendesk Transcriber")
st.write("Using Universal-1 model for maximum accuracy.")

audio = mic_recorder(
    start_prompt="▶️ Start Recording",
    stop_prompt="⏹️ Stop & Transcribe",
    key='aai_recorder'
)

if audio:
    st.audio(audio['bytes'])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio['bytes'])
        tmp_path = tmp_file.name

    try:
        with st.spinner("AssemblyAI is analyzing the Australian accent..."):
            
            # Updated Configuration for modern AssemblyAI SDK
            config = aai.TranscriptionConfig(
                # help the AI with 'support' context and Aussie slang
                word_boost=["Zendesk", "ticket", "refund", "Melbourne", "Sydney"],
                # In newer versions, we use boost_param or just the word_boost list
                # 'language_code' handles the Aussie accent
            language_code="en_au" 
)

            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(tmp_path, config=config)

            if transcript.status == aai.TranscriptStatus.error:
                st.error(transcript.error)
            else:
                st.subheader("Transcription:")
                st.success(transcript.text)
                
                # Bonus: Automatic Summary (AssemblyAI is great at this)
                if st.checkbox("Show Summary"):
                    st.write(transcript.export_subtitles_vtt()) # Or other insights

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
