import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load vector store
index = faiss.read_index("vector_store/index.faiss")
with open("vector_store/index.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2", max_new_tokens=200)

# Prompt template
TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Use the following complaint excerpts to answer the question.
If the context is insufficient, say you don’t have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

def rag_response(question, k=5):
    q_embed = embedder.encode([question])
    _, indices = index.search(q_embed, k)

    chunks = []
    sources = []

    for i in indices[0]:
        meta = metadata[i]
        text = meta.get("text", "[text missing]")
        chunks.append(text)
        sources.append(f"• {text[:250]}... (Product: {meta.get('Product')}, ID: {meta.get('Complaint ID')})")

    prompt = TEMPLATE.format(context="\n\n".join(chunks), question=question)
    result = generator(prompt)[0]['generated_text'].split("Answer:")[-1].strip()

    return result, sources

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="CrediTrust RAG Chat", layout="wide")
st.title("💬 Ask a Question about Consumer Complaints")

query = st.text_input("Enter your question")

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask")
with col2:
    clear = st.button("Clear")

if ask and query:
    with st.spinner("Generating answer..."):
        answer, source_chunks = rag_response(query)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source Chunks")
        for source in source_chunks:
            st.markdown(f"> {source}")
elif clear:
    st.experimental_rerun()
