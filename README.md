# Intelligent Complaint Analysis

A Python-based system for analyzing, embedding, and indexing complaint data using modern NLP and vector search techniques.

---

## Features

- **Data Preprocessing:** Clean and prepare complaint datasets for analysis.
- **Exploratory Data Analysis (EDA):** Jupyter notebooks for visualizing and understanding complaint data.
- **Chunking & Embedding:** Split complaints into chunks and generate vector embeddings.
- **Vector Store:** Efficient similarity search using FAISS and pickle-based indices.
- **Modular Codebase:** Organized into `src/` for easy extension and maintenance.
- **Unit Testing:** Automated tests in the `tests/` directory.

---

## Project Structure

```
intelligent-complaint-analysis/
│
├── data/                  # Raw and processed data
├── notebooks/             # Jupyter notebooks for EDA and preprocessing
├── src/
│   ├── chunk_embed_index.py   # Chunking, embedding, and indexing logic
│   └── ...                   # Other source files
├── tests/                 # Unit tests
├── vector_store/
│   ├── index.faiss        # FAISS vector index
│   └── index.pk1          # Pickle-based index
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/ha-bina/intelligent-complaint-analysis.git
cd intelligent-complaint-analysis
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Prepare Data

- Place your raw complaint data  in the `data/raw/` directory.

### 4. Run EDA and Preprocessing

- Open and run the notebook:  
  `notebooks/eda_preprocessing.ipynb`

### 5. Chunk, Embed, and Index

- Run the main script to process data and build the vector index:
  ```sh
  python src/chunk_embed_index.py
  ```
- This will generate/update `vector_store/index.faiss` and `vector_store/index.pk1`.

---

## Usage

- **Querying:** Use the vector store to find similar complaints or perform semantic search.
- **Extending:** Add new embedding models or chunking strategies by editing `src/chunk_embed_index.py`.

---

## Testing

Run all unit tests with:

```sh
pytest tests/
```
