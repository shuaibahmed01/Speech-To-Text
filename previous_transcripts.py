import streamlit as st
from datetime import datetime

st.title("Previous Transcripts")

if 'transcripts' in st.session_state and st.session_state.transcripts:
    for idx, transcript_data in enumerate(st.session_state.transcripts):
        with st.expander(f"{transcript_data['timestamp']} - {transcript_data['filename']}"):
            st.write(transcript_data['transcript'])
            
            # Add download button for each transcript
            pdf = create_pdf(transcript_data['transcript'])
            st.download_button(
                f"Download PDF for {transcript_data['filename']}", 
                pdf, 
                file_name=f"transcript_{idx+1}.pdf", 
                mime="application/pdf"
            )
else:
    st.write("No previous transcripts found.")

# Import the create_pdf function from main.py
from main import create_pdf