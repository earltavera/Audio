import streamlit as st
from streamlit_mic_recorder import mic_recorder
import assemblyai as aai
import tempfile
import os

# --- CONFIGURATION ---
# Replace with your actual API key from the AssemblyAI dashboard
aai.settings.api_key = "YOUR_ASSEMBLYAI_API_KEY"

st.set_page_config(page_title="Zendesk AU Transcriber", page_icon="🎧", layout="wide")

# --- UI HEADER ---
st.title("🎧 Zendesk Australian Call Transcriber")
st.markdown("""
### 🛠️ Windows Setup:
1. Set your **Windows Output** to **VB-Cable Input** (or use Stereo Mix).
2. Play the Zendesk audio.
3. Click **Start Recording** and ensure your browser is using the **Virtual Cable/Stereo Mix** as the mic.
""")

# --- THE RECORDER ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Capture Audio")
    audio = mic_recorder(
        start_prompt="▶️ Start Recording System Audio",
        stop_prompt="⏹️ Stop & Transcribe",
        key='zendesk_recorder'
    )

# --- PROCESSING ---
if audio:
    with col2:
        st.subheader("Audio Preview")
        st.audio(audio['bytes'])
    
    # Save the captured audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio['bytes'])
        tmp_path = tmp_file.name

    try:
        with st.spinner("🚀 AI is analyzing the Australian accent and speaker roles..."):
            
            # Configure for AU English + Support Context + Speaker Labels
            config = aai.TranscriptionConfig(
                language_code="en_au",
                speaker_labels=True,  # Separates Agent and Customer
                word_boost=["Zendesk", "ticket", "account", "refund", "Melbourne", "Sydney"],
                punctuation_src="automatic",
                format_text=True
            )

            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(tmp_path, config=config)

            if transcript.status == aai.TranscriptStatus.error:
                st.error(f"AssemblyAI Error: {transcript.error}")
            else:
                st.divider()
                st.subheader("📄 Final Transcription")
                
                # Display results with Speaker Labels
                transcript_text = ""
                if transcript.utterances:
                    for utterance in transcript.utterances:
                        line = f"**Speaker {utterance.speaker}:** {utterance.text}\n\n"
                        st.write(line)
                        transcript_text += line
                else:
                    st.write(transcript.text)
                    transcript_text = transcript.text
                
                # --- DOWNLOAD BUTTON ---
                st.download_button(
                    label="📥 Download Transcription (.txt)",
                    data=transcript_text.replace("**", ""), # Remove bolding for the text file
                    file_name="zendesk_au_transcription.txt",
                    mime="text/plain"
                )
                
    except Exception as e:
        st.error(f"Application Error: {e}")
    finally:
        # Cleanup the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
