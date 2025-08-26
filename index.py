import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from openai import OpenAI
import faiss
import numpy as np
import os
import pickle
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

print("Starting index creation process...")

# 1. Initialize OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 2. Load and Chunk the Document
pdf_path = "PetroSafe Global Holdings.pdf"
print(f"Loading document: {pdf_path}")
raw_text = ""
with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        raw_text += page.extract_text() if page.extract_text() else ""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(raw_text)]
print(f"Document split into {len(documents)} chunks.")

# 3. Create Embeddings with OpenAI API
print("Creating embeddings via OpenAI API... (This may take a moment)")
doc_contents = [doc.page_content for doc in documents]
response = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=doc_contents)
embeddings = [item.embedding for item in response.data]

# 4. Build and Save FAISS Index
print("Building and saving FAISS index...")
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, "index.faiss")
print("FAISS index saved to index.faiss")

# 5. Save the processed documents
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)
print("Documents saved to documents.pkl")

print("\nIndex creation complete!")