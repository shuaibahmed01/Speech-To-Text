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
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import json
load_dotenv()

UPLOAD_FOLDER = "uploads"
TRANSCRIPT_FILE = "transcripts.json"
VECTORSTORE_FILE = "vectorstore.pkl"

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load transcripts from file
def load_transcripts():
    if os.path.exists(TRANSCRIPT_FILE):
        with open(TRANSCRIPT_FILE, 'r') as f:
            return json.load(f)
    return []

# Save transcripts to file
def save_transcripts(transcripts):
    with open(TRANSCRIPT_FILE, 'w') as f:
        json.dump(transcripts, f)

# Load vectorstore from file
def load_vectorstore():
    if os.path.exists(VECTORSTORE_FILE):
        with open(VECTORSTORE_FILE, 'rb') as f:
            return pickle.load(f)
    return None

# Save vectorstore to file
def save_vectorstore(vectorstore):
    with open(VECTORSTORE_FILE, 'wb') as f:
        pickle.dump(vectorstore, f)

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
        
# Modify the global vectorstore initialization
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def transcribe_page():
    st.subheader("Generate Transcription")
    # Upload audio file
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio_bytes = uploaded_file.read()

        # Transcribe audio
        transcript = transcribe_audio(audio_bytes)
        # Show transcript
        st.subheader("Transcript")
        st.write(transcript)
        
        # Store the transcript
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.transcripts.append({"timestamp": timestamp, "filename": uploaded_file.name, "transcript": transcript})

        # Generate and show search bar
        global vectorstore
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = build_vectorstore(transcript)
        else:
            add_to_vectorstore(transcript)
        

        # Save transcripts after adding new one
        save_transcripts(st.session_state.transcripts)

        # Save vectorstore after updating
        save_vectorstore(st.session_state.vectorstore)

        # Download PDF button
        if st.button("Download Transcript as PDF"):
            pdf = create_pdf(transcript)
            st.download_button("Download PDF", pdf, file_name="transcript.pdf", mime="application/pdf")

def transcribe_history_page():
    st.subheader("Transcript History")
    if not st.session_state.transcripts:
        st.write("No transcripts available.")
    else:
        for i, transcript in enumerate(st.session_state.transcripts):
            with st.expander(f"{transcript['filename']} - {transcript['timestamp']}"):
                st.write(transcript['transcript'])
                pdf = create_pdf(transcript['transcript'])
                st.download_button("Download PDF", pdf, file_name=f"transcript_{transcript['timestamp']}.pdf", mime="application/pdf")
                if st.button(f"Chat with Bot", key=f"chat_button_{i}"):
                    st.session_state.current_transcript = transcript['transcript']
                    st.session_state.show_chat = True

    if 'show_chat' in st.session_state and st.session_state.show_chat:
        chat_with_bot()

def chat_with_bot():
    st.sidebar.title("Chat with Bot")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create two columns with adjusted ratios
    col1, col2 = st.sidebar.columns([4, 1])

    # Place user input in the first (wider) column
    with col1:
        user_input = st.text_input(
            label="",
            key="user_input",
            placeholder="Type your message here..."
        )

    # Place clear chat button in the second (narrower) column
    with col2:
        clear_chat = st.button("Clear", key="clear_chat")

    # Handle user input
    if user_input:
        response = get_bot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        # Clear the input box after sending
        st.session_state.user_input = ""

    # Handle clear chat button
    if clear_chat:
        st.session_state.chat_history = []
        st.experimental_rerun()
    # Reverse the chat history and display it
    for role, message in reversed(st.session_state.chat_history):
        st.sidebar.write(f"<b>{role}</b>: {message}", unsafe_allow_html=True)


def get_bot_response(query):
    if st.session_state.vectorstore is None:
        return "I'm sorry, but there's no transcript data available to answer your question."

    chat = ChatOpenAI(model="gpt-4o",temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Define the prompt template
    template = """You are an AI assistant tasked with answering questions about audio transcripts. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create the retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=st.session_state.vectorstore.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    result = qa_chain({"question": query, "chat_history": []})
    return result['answer']

# Add this new function
def add_to_vectorstore(text):
    save_vectorstore(st.session_state.vectorstore)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    st.session_state.vectorstore.add_texts(chunks)

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
