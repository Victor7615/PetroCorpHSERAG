import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
import numpy as np
import PyPDF2
import os
from dotenv import load_dotenv

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Hybrid HSE Compliance Assistant",
    page_icon="⚖️",
    layout="wide"
)

# --- LOAD SECRETS ---
load_dotenv()

def get_secret(key):
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return None

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🚨 OPENAI_API_KEY is not configured. Please set it in your .env file locally or in Render secrets for deployment.")
    st.stop()

# --- CACHED RESOURCES ---
@st.cache_resource
def load_resources(pdf_path):
    """
    Loads and caches the open-source components: embedding model, document, and FAISS index.
    Initializes the OpenAI client.
    """
    # 1. Load and Chunk the Document
    st.write("Loading knowledge base...")
    raw_text = PyPDF2.PdfReader(pdf_path).pages[0].extract_text() # Simplified for single page
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(raw_text)]

    # 2. Load the Open-Source Embedding Model
    st.write("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Create Embeddings and FAISS Index (Local)
    st.write("Creating local vector index...")
    embeddings = embedding_model.encode([doc.page_content for doc in documents])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    # 4. Initialize the OpenAI Client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    st.success("All resources loaded successfully!")
    return {
        "documents": documents,
        "embedding_model": embedding_model,
        "index": index,
        "openai_client": openai_client
    }

# --- MAIN APP LOGIC ---
st.title("Global PetroCorp Compliance Assistant ⚖️")
st.markdown("This assistant provides answers grounded in official corporate policy documents")

PDF_FILE_PATH = "C:\\Users\\USER\\Downloads\\OpenSourceRAG\\PetroSafe Global Holdings.pdf"

if not os.path.exists(PDF_FILE_PATH):
    st.error(f"Required file is missing: {PDF_FILE_PATH}")
    st.stop()

try:
    resources = load_resources(PDF_FILE_PATH)
    documents = resources["documents"]
    embedding_model = resources["embedding_model"]
    index = resources["index"]
    openai_client = resources["openai_client"]
except Exception as e:
    st.error(f"Failed to load resources. Error: {e}")
    st.stop()

def get_rag_response(question):
    """Orchestrates the hybrid RAG process."""
    # 1. Embed question locally
    question_embedding = embedding_model.encode([question])

    # 2. Search FAISS index locally
    D, I = index.search(np.array(question_embedding), k=3)
    context = " ".join([documents[i].page_content for i in I[0]])
    
    # 3. Build prompt for OpenAI
    system_prompt = f"""
    You are "HSE Assist," a specialized AI assistant. Your primary purpose is to help employees understand and comply with the company's Health, Safety, and Environment (HSE) policies.
    Your answers must be grounded in and consistent with the context provided below. If the context doesn't cover a topic, state that the specific information is not available in the policy and recommend contacting a human HSE supervisor.
    
    CONTEXT:
    ---
    {context}
    ---
    """
    
    # 4. Generate final answer using OpenAI API
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o", # Or another model like gpt-3.5-turbo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1
    )
    return chat_completion.choices[0].message.content

# --- STREAMLIT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    disclaimer = "I am HSE Assist, an AI guide to the HSE Policy. How can I help you today?"
    st.session_state.messages.append({"role": "assistant", "content": disclaimer})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a compliance question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response..."):
            response = get_rag_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})