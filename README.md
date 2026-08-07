# Data Ingestion, Vector Search & RAG Generation Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system built with OpenRouter API integration, custom prompt engineering, API error handling, hallucination guardrails, and latency logging for Parallax Labs Internship.

---

## 📌 Progress & Deliverables Completed

### Week 1: Ingestion & Data Cleaning
- **Environment Setup:** Configured dependencies (`pandas`, `tqdm`, `datasets`, `pytest`).
- **Data Acquisition:** Ingested 5,100 real-world technical documents from Wikipedia API across Computer Science domains.
- **Data Cleaning:** Sanitized HTML tags, URLs, non-ASCII characters, and noise.

### Week 2: Chunking, Embeddings & Vector Store
- **Recursive Text Chunking:** Implemented character-recursive chunking (`chunk_size=500`, `chunk_overlap=50`).
- **Embedding Generation:** Leveraged `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional dense embeddings.
- **ChromaDB Vector Store:** Set up persistent vector storage (`data/vector_db`) with edge-case handling.
- **Retrieval Testing:** Benchmarked semantic search latency across multiple query profiles.

### Week 3: LLM Integration, Prompt Engineering & Guardrails
- **OpenRouter API Integration:** Connected OpenRouter LLM endpoints for context-grounded answer generation (`src/llm_client.py`).
- **Prompt Engineering:** Designed strict system instructions and context injection boundaries (`src/prompt_template.py`).
- **API Robustness & Error Handling:** Implemented exponential backoff retries handling rate limits (429), timeouts, and token limits (`src/llm_client.py`).
- **Hallucination & Domain Guardrails:** Built dual-layer filtering to block out-of-domain queries and verify context relevance before generation (`src/guardrails.py`).
- **End-to-End Latency Logging:** Benchmarked granular pipeline latency (Retrieval vs. Generation vs. Total Pipeline) in `src/rag_pipeline.py`.

---

## 📊 Performance & Latency Summary

| Metric / Stage | Benchmark Result |
| :--- | :--- |
| **Embedding Engine** | `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM Engine** | OpenRouter API (`google/gemini-2.5-flash`) |
| **Average Retrieval Latency** | **~30 - 50 ms** |
| **Average LLM Generation Latency** | **~1500 - 2000 ms** |
| **Out-of-Domain Filtering** | **100% Intercepted** via Guardrails |

---

## 📁 Repository Architecture

```text
.
├── data/
│   ├── raw/                # Raw acquired documents
│   ├── processed/          # Clean dataset (clean_dataset.csv)
│   └── vector_db/          # Persistent ChromaDB storage
├── src/
│   ├── __init__.py
│   ├── cleaning.py         # Text cleaning pipeline
│   ├── utils.py            # Environment setup check
│   ├── vector_store.py     # Recursive chunking & vector store
│   ├── llm_client.py       # OpenRouter API client with error retries
│   ├── prompt_template.py  # System prompt & context injector
│   ├── guardrails.py       # Hallucination & off-topic filter
│   ├── rag_pipeline.py     # End-to-end RAG runner & latency logger
│   └── test_rag.py         # Test suite runner
├── tests/
│   ├── test_cleaning.py    # Cleaning unit tests
│   └── test_chunking.py    # Chunking unit tests
├── .env.example            # Environment template
├── fetch_data.py           # Data acquisition script
├── requirements.txt        # Project dependencies
└── README.md               # Documentation