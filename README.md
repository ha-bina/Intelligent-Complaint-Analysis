# Intelligent Complaint Analysis

A modular pipeline for analyzing, cleaning, embedding, and indexing consumer complaint narratives using modern NLP and vector search techniques.

---

## Project Structure

```
intelligent-complaint-analysis/
│
├── data/                  # Raw and processed data
├── notebooks/
│   └── eda_preprocessing.ipynb   # EDA and preprocessing notebook
├── src/
│   └── chunk_embed_index.py      # Chunking, embedding, and vector indexing
├── vector_store/          # Vector and metadata indices
├── app.py                 # (Optional) Main application entry point
├── rag_pipeline.py        # (Optional) RAG pipeline script
├── rag_evaluation.py      # (Optional) RAG evaluation script
├── test_rag.py            # (Optional) Tests for RAG pipeline
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Workflow

1. **Data Preparation & EDA**
   - Use `notebooks/eda_preprocessing.ipynb` to:
     - Load and explore your data (e.g., `path6/complaints.csv`)
     - Filter, clean, and save processed data to `path6/filtered_complaints.csv`

2. **Chunking, Embedding, and Indexing**
   - Use `src/chunk_embed_index.py` (or its function) to:
     - Split narratives into chunks
     - Generate embeddings with SentenceTransformers
     - Build a FAISS vector index and save metadata

   Example usage in a notebook or script:
   ```python
   from src.chunk_embed_index import build_vector_index
   build_vector_index(data_path="path6/filtered_complaints.csv", vector_dir="path6/vector_store")
   ```

3. **(Optional) Run Application or Pipelines**
   - Run main app:
     ```sh
     python app.py
     ```
   - Run RAG pipeline:
     ```sh
     python rag_pipeline.py
     ```
   - Run RAG evaluation:
     ```sh
     python rag_evaluation.py
     ```

4. **Testing**
   - Run tests:
     ```sh
     pytest test_rag.py
     ```
     or
     ```sh
     python test_rag.py
     ```

---

## Requirements

Install dependencies:
```sh
pip install -r requirements.txt
```
Or, from within a notebook:
```python
%pip install pandas matplotlib seaborn sentence-transformers langchain faiss-cpu
```



