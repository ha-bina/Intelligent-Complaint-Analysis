import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FAISS index and metadata
index = faiss.read_index("vector_store/index.faiss")
with open("vector_store/index.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model (same as Task 2)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM for generation (you can swap this with LangChain or OpenAI if needed)
generator = pipeline("text-generation", model="gpt2", max_new_tokens=200)

# Prompt template
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

# Retrieval function
def retrieve_top_k(question, k=5):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, k)
    retrieved_chunks = [metadata[i] for i in indices[0]]
    retrieved_texts = [metadata[i].get("text", "N/A") for i in indices[0]]
    return retrieved_chunks, retrieved_texts

# RAG pipeline function
def rag_pipeline(question, k=5):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, k)

    context_chunks = []
    sources = []
    for i in indices[0]:
        meta = metadata[i]
        context_chunks.append(meta.get("text", ""))
        sources.append(f"(Product: {meta.get('Product')}, ID: {meta.get('Complaint ID')})")

    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = generator(prompt)[0]["generated_text"]
    answer = response.split("Answer:")[-1].strip()

    return answer, sources[:2]  # return only top 2 sources for brevity
