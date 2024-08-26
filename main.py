import streamlit as st
import whisper
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os
from tempfile import NamedTemporaryFile
import pickle
from datetime import datetime

load_dotenv()



# Load Whisper model
model = whisper.load_model("base")  # or use "small", "medium", "large" depending on your needs

# Initialize LangChain components
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
vectorstore = None


# Add this near the top of the file
if 'transcripts' not in st.session_state:
    st.session_state.transcripts = []

# Streamlit app
def main():
    st.title("Audio Transcription and PDF Generation with Search")

    page = st.sidebar.selectbox("Pages:", ("Generate Transcription", "Transcript History"))

    if page == "Generate Transcription":
        transcribe_page()
    elif page == "Transcript History":
        transcribe_history_page()
        

def transcribe_page():
    st.subheader("Generate Transcription")
    # Upload audio file
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()

        # Transcribe audio
        transcript = transcribe_audio(audio_bytes)
        
        if uploaded_file is not None:
            # Show transcript
            st.subheader("Transcript")
            st.write(transcript)
            
            # Store the transcript
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.transcripts.append({"timestamp": timestamp, "filename": uploaded_file.name, "transcript": transcript})

            # Generate and show search bar
            global vectorstore
            if vectorstore is None:
                vectorstore = build_vectorstore(transcript)
            else:
                add_to_vectorstore(transcript)
            
            st.subheader("Search in Transcript")
            query = st.text_input("Enter your search query:")
            
            if query:
                results = search_transcript(query)
                st.write("### Search Results")
                for result in results:
                    st.write(result)

            # Download PDF button
            if st.button("Download Transcript as PDF"):
                pdf = create_pdf(transcript)
                st.download_button("Download PDF", pdf, file_name="transcript.pdf", mime="application/pdf")

def transcribe_history_page():
    st.subheader("Transcript History")
    if not st.session_state.transcripts:
        st.write("No transcripts available.")
    else:
        for transcript in st.session_state.transcripts:
            with st.expander(f"{transcript['filename']} - {transcript['timestamp']}"):
                st.write(transcript['transcript'])
                pdf = create_pdf(transcript['transcript'])
                st.download_button("Download PDF", pdf, file_name=f"transcript_{transcript['timestamp']}.pdf", mime="application/pdf")

# Add this new function
def add_to_vectorstore(text):
    global vectorstore
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    vectorstore.add_texts(chunks)
    
def transcribe_audio(audio_bytes):
    # Save the audio bytes to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name
        

    # Transcribe audio
    result = model.transcribe(temp_audio_file_path)
    return result['text']

def create_pdf(text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add text to PDF
    c.drawString(72, height - 72, "Transcript:")
    c.drawString(72, height - 100, text)
    c.save()

    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def build_vectorstore(text):
    # Split text into chunks for indexing
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    # Create the vector store
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore

def search_transcript(query):
    if vectorstore is None:
        return []
    
    # Perform the search
    results = vectorstore.similarity_search(query)
    return [result for result in results]

if __name__ == "__main__":
    main()
