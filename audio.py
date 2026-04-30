import streamlit as st
from streamlit_mic_recorder import mic_recorder
import assemblyai as aai
import tempfile
import os

# Set your API Key here
aai.settings.api_key = "b9bb4fd14d1d4d56ac08701e7ccc6918"

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
            
            # Configure the transcription
            config = aai.TranscriptionConfig(
                # Help the AI with 'support' context and Aussie slang
                word_boost=["Zendesk", "ticket", "refund", "Melbourne", "Sydney"],
                boost_score="high",
                # AssemblyAI handles AU English automatically, 
                # but you can specify it if needed
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
