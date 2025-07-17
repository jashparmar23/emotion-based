# rag_loader.py

import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import pickle

# === STEP 1: Load Documents ===
docs_folder = "docs"  # Create a folder named 'docs' and put your PDFs, DOCX, TXT inside

all_documents = []

for filename in os.listdir(docs_folder):
    file_path = os.path.join(docs_folder, filename)
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        continue  # Skip unknown file types
    docs = loader.load()
    all_documents.extend(docs)

print(f"✅ Loaded {len(all_documents)} documents")

# === STEP 2: Split into chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_documents)

print(f"✅ Split into {len(chunks)} text chunks")

# === STEP 3: Create Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === STEP 4: Store in FAISS Vector Store ===
vector_store = FAISS.from_documents(chunks, embedding_model)

# Save the FAISS index
faiss_folder = "vectorstore"
os.makedirs(faiss_folder, exist_ok=True)
vector_store.save_local(faiss_folder)

print("✅ Embeddings generated and saved to FAISS index!")
