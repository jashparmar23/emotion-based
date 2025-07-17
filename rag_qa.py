# ask_llm.py

from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# === Load Embedding Model ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load FAISS Vector Store (with deserialization allowed since it's self-generated) ===
vector_store = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# === Search Relevant Chunks ===
def search_docs(query, vector_store, top_k=3):
    docs = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]

# === Build Prompt ===
def build_prompt(context_chunks, question):
    context = "\n".join(context_chunks)
    return f"""### Instruction:
You are a helpful assistant that only answers questions related to brain tumors using the given context.
If the question is not related to the provided context, say: "❌ Sorry, I can only help with brain tumor-related questions."

### Context:
{context}

### Question:
{question}

### Response:"""

# === Query TinyLlama Model ===
def ask_llm(prompt):
    llm = Llama(
        model_path="models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=8,      # Adjust based on your CPU
        n_gpu_layers=20   # Adjust based on GPU (4GB VRAM should support ~20 layers)
    )
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=400,
        temperature=0.7,
        stop=["###"]
    )
    return response["choices"][0]["text"].strip()

# === MAIN ===
if __name__ == "__main__":
    question = input("❓ Ask a question: ")

    top_chunks = search_docs(question, vector_store)
    prompt = build_prompt(top_chunks, question)
    answer = ask_llm(prompt)

    print("\n🤖 Answer:")
    print(answer)
