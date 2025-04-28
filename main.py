import fitz  
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv(override=False)
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-large")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings, model

def store_vectors_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_chunks(query, model, index, chunks, top_k=5):
    query_vec = model.encode([query])
    _, indices = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""### Instruction:\nAnswer the question based on the context.\n\n### Context:\n{context}\n\n### Question:\n{query}\n\n### Answer:"""

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.2, "max_new_tokens": 200}
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()[0]['generated_text'].split("### Answer:")[-1].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

def main():
    pdf_path = "sample.pdf"
    query = input("Ask your question: ")

    print("Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("embedding chunks...")
    embeddings, model = embed_chunks(chunks)

    print("vectors in FAISS...")
    index = store_vectors_faiss(np.array(embeddings))

    print("serching relevant chunks...")
    relevant_chunks = search_chunks(query, model, index, chunks)

    #print(relevant_chunks)

    print("Generating answer...")
    answer = generate_answer(query, relevant_chunks)

    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
