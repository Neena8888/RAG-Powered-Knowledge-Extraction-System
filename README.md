# Data Ingestion, Chunking & Vector Search Engine

A production-ready data collection, recursive chunking, embedding generation, and ChromaDB vector search engine built for Parallax Labs Internship.

---

## 📌 Week 1 & Week 2 Objectives & Deliverables Completed

### Week 1: Ingestion & Data Cleaning
- **Environment & Setup Verification:** Configured dependencies (`pandas`, `tqdm`, `datasets`, `pytest`).
- **Data Acquisition:** Ingested **5,100 real-world technical documents** from Wikipedia API across core CS domains.
- **Data Cleaning & Sanitation:** Sanitized HTML tags, URLs, non-ASCII text, and redundant formatting.

### Week 2: Chunking, Embeddings & Vector DB
- **Recursive Text Chunking:** Implemented character-recursive chunking strategy (`chunk_size=500`, `chunk_overlap=50`) preserving semantic paragraph structure.
- **Embedding Generation & Benchmarking:** Generated 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` and logged embedding performance metrics.
- **ChromaDB Vector Store Setup:** Configured persistent ChromaDB vector store (`data/vector_db`) handling edge cases (duplicate deletion, micro-chunks filtering).
- **Retrieval Performance & Latency Testing:** Developed `src/test_retrieval.py` benchmarking script measuring query execution speeds across various query types.
- **Unit Testing:** Verified recursive splitting edge cases via `pytest` suite.

---

## 📊 Week 2 Performance & Benchmarking Summary

| Metric / Parameter | Result / Metric Value |
| :--- | :--- |
| **Recursive Chunking Strategy** | `chunk_size=500`, `chunk_overlap=50` |
| **Embedding Model Used** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database Engine** | Persistent `ChromaDB` (`data/vector_db`) |
| **Total Chunks Vectorized** | ~10,000+ chunks |
| **Avg Search Retrieval Latency** | **< 15 ms** per query |

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── raw/                # 5,100 raw Wikipedia documents
│   ├── processed/          # Cleaned dataset (clean_dataset.csv)
│   └── vector_db/          # Persistent ChromaDB vector database
├── src/
│   ├── __init__.py
│   ├── cleaning.py         # Data cleaning & sanitation pipeline
│   ├── utils.py            # Environment verification script
│   ├── vector_store.py     # Recursive chunking & ChromaDB ingestion
│   └── test_retrieval.py  # Retrieval latency benchmarking script
├── tests/
│   ├── test_cleaning.py    # Unit tests for text cleaning
│   └── test_chunking.py    # Unit tests for text chunking
├── fetch_data.py           # Bulk data acquisition script
├── requirements.txt        # Project dependencies
└── README.md               # Complete documentation