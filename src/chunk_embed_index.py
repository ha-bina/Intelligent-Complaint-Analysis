import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import os
import pickle

# -----------------------
# Paths
# -----------------------
DATA_PATH = "data/processed_data/filtered_complaints.csv"
VECTOR_DIR = "vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

# -----------------------
# Load Cleaned Data
# -----------------------
df = pd.read_csv(DATA_PATH)
texts = df["cleaned_narrative"].tolist()
metadata = df[["Product", "Complaint ID"]].to_dict(orient="records")

# -----------------------
# Chunking Strategy
# -----------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # balance: long enough for context, short enough for accurate embedding
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

all_chunks = []
chunk_metadata = []

for i, text in enumerate(texts):
    chunks = text_splitter.split_text(text)
    all_chunks.extend(chunks)
    chunk_metadata.extend([{
        "Product": metadata[i]["Product"],
        "Complaint ID": metadata[i]["Complaint ID"],
        "Chunk Index": j
    } for j in range(len(chunks))])

print(f"Total chunks created: {len(all_chunks)}")

# -----------------------
# Embedding
# -----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks, show_progress_bar=True)

# -----------------------
# FAISS Indexing
# -----------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))

# Save metadata separately
with open(os.path.join(VECTOR_DIR, "index.pkl"), "wb") as f:
    pickle.dump(chunk_metadata, f)

print(f"\n✅ Vector store saved to: {VECTOR_DIR}/index.faiss and metadata in index.pkl")
