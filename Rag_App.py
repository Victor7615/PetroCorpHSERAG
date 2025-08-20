import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from openai import OpenAI
import faiss
import numpy as np
import PyPDF2
import os
from dotenv import load_dotenv

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Global PetroCorp Compliance Assistant",
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

# Load credentials from .env or Streamlit's secrets manager
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = get_secret("OPENAI_EMBEDDING_MODEL")
OPENAI_CHAT_MODEL = get_secret("OPENAI_CHAT_MODEL")

# Verify that all necessary secrets have been loaded
if not all([OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL]):
    st.error("🚨 Critical secrets are missing. Please configure OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, and OPENAI_CHAT_MODEL.")
    st.stop()

# --- CACHED RESOURCES ---
@st.cache_resource
def setup_rag_pipeline(pdf_path):
    """
    Loads resources, creates embeddings via OpenAI, and builds a local FAISS index.
    This function is cached to run only once per session.
    """
    # 1. Initialize the OpenAI Client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 2. Load and Chunk the Document
    st.write("Loading knowledge base...")
    raw_text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            raw_text += page.extract_text() if page.extract_text() else ""
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(raw_text)]

    # 3. Create Embeddings with OpenAI API and Build FAISS Index
    st.write("Creating embeddings...")
    doc_contents = [doc.page_content for doc in documents]
    response = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=doc_contents)
    embeddings = [item.embedding for item in response.data]
    
    st.write("Creating local vector index ...")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings, dtype=np.float32))

    st.success("All resources loaded and indexed successfully!")
    return {
        "documents": documents,
        "openai_client": openai_client,
        "index": index
    }

# --- MAIN APP LOGIC ---
st.title("Global PetroCorp Compliance Assistant ⚖️")
st.markdown("This assistant provides answers grounded in official corporate policy documents.")

PDF_FILE_PATH = "PetroSafe Global Holdings.pdf"

if not os.path.exists(PDF_FILE_PATH):
    st.error(f"Required knowledge base file is missing: {PDF_FILE_PATH}")
    st.stop()

# Load all resources using the setup function
try:
    resources = setup_rag_pipeline(PDF_FILE_PATH)
    documents = resources["documents"]
    openai_client = resources["openai_client"]
    index = resources["index"]
except Exception as e:
    st.error(f"Failed to load RAG pipeline. Error: {e}")
    st.stop()


def get_rag_response(question):
    """
    Orchestrates the RAG process using OpenAI for embeddings/chat and FAISS for search.
    """
    # 1. Embed the user's question using OpenAI API
    response = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[question])
    question_embedding = response.data[0].embedding

    # 2. Search the local FAISS index for relevant document chunks
    D, I = index.search(np.array([question_embedding], dtype=np.float32), k=3)
    context = " ".join([documents[i].page_content for i in I[0]])
    
    # 3. Build the prompt for the OpenAI chat model
    system_prompt = f"""
    You are "HSE Assist," a specialized AI assistant for PetroSafe Global Holdings. Your primary purpose is to help all employees and contractors understand and comply with the company's Health, Safety, and Environment (HSE) policies and procedures. You are a supportive and knowledgeable resource designed to make safety information accessible and clear.

    1. Core Knowledge Base:
    Your primary source of truth is the PetroSafe Global Holdings HSE Policy, Document ID: PSG-HSE-POL-001. All your answers must be grounded in and consistent with the context provided below. If the context doesn't cover a topic, state that the specific information is not available in the policy and recommend contacting a human HSE supervisor.

    2. Interaction Style & Tone:
    Your tone must be supportive, clear, patient, and professional. Always prioritize safety. Reinforce the company's commitment to "zero harm." Base your answers directly on the text of the policy document provided in the context.

    3. Crucial Guardrails & Limitations:
    You do not approve permits, authorize work, or conduct risk assessments. Your role is to inform, not to authorize. If a user asks for permission, you must refuse and direct them to the proper human authority. If a query is unclear or describes a complex situation not explicitly covered in the policy, do not invent an answer; instead, direct the user to their supervisor. You are aware that the current date is August 20, 2025.
    CONTEXT:
    ---
    {context}
    ---
    """
    
    # 4. Generate the final answer using the OpenAI Chat API
    chat_completion = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
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
    disclaimer = "I am HSE Assist, your AI guide to the HSE Policy. My purpose is to provide information from this policy. I am not a substitute for professional judgment or your supervisor's direction. How can I help you today?"
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