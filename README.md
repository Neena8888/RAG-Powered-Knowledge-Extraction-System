# Data Ingestion, Vector Search & RAG Generation Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system with OpenRouter LLM integration, prompt engineering, API error handling, hallucination guardrails, NLP Topic & Sentiment Analysis, and metadata-filtered vector retrieval built for Parallax Labs Internship.

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
- **Hallucination & Domain Guardrails:** Built dual-layer filtering to block out-of-domain queries and verify context relevance (`src/guardrails.py`).
- **End-to-End Latency Logging:** Benchmarked granular pipeline latency in `src/rag_pipeline.py`.

### Week 4: NLP Analysis (Topic Modeling & Sentiment Filtering)
- **Latent Dirichlet Allocation (LDA) Topic Modeling:** Discovered corpus themes across technical categories (`src/topic_modeler.py`).
- **Sentiment Analysis & Accuracy Evaluation:** Implemented lexicon sentiment scoring and validated accuracy on a labeled benchmark (`src/nlp_analyzer.py`).
- **Edge Case Safeguards:** Handled micro-documents, short texts, and domain jargon gracefully (`src/nlp_enricher.py`).
- **Metadata Enriched ChromaDB Vector Store:** Ingested topic and sentiment tags into ChromaDB for filtered retrieval (`src/nlp_enricher.py`).
- **Filtered Semantic Search:** Implemented metadata filtering (`topic`, `sentiment`) to improve context retrieval precision.

---

## 📊 Performance & Accuracy Benchmarks

| Evaluation Metric / Stage | Benchmark Result |
| :--- | :--- |
| **Embedding Engine** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Topic Modeling Strategy** | Latent Dirichlet Allocation (LDA) - 5 Themes |
| **Sentiment Validation Accuracy** | **83.33%** (on manual validation set) |
| **Metadata Filtered Search** | Precision-boosted vector retrieval via ChromaDB `where` clause |
| **Out-of-Domain Guardrail Filtering** | **100% Intercepted** via Guardrails |

---

## 📁 Repository Architecture

```text
.
├── data/
│   ├── raw/                # Raw acquired documents
│   ├── processed/          # Clean dataset (clean_dataset.csv)
│   └── vector_db/          # Persistent ChromaDB storage with NLP metadata
├── src/
│   ├── __init__.py
│   ├── cleaning.py         # Text cleaning pipeline
│   ├── utils.py            # Environment setup check
│   ├── vector_store.py     # Recursive chunking & vector store
│   ├── llm_client.py       # OpenRouter API client with error retries
│   ├── prompt_template.py  # System prompt & context injector
│   ├── guardrails.py       # Hallucination & off-topic filter
│   ├── rag_pipeline.py     # End-to-end RAG runner & latency logger
│   ├── topic_modeler.py    # LDA topic modeling
│   ├── nlp_analyzer.py     # Sentiment analysis & validation
│   ├── nlp_enricher.py     # Metadata enriched ChromaDB store
│   ├── test_nlp.py         # Week 4 NLP test runner
│   └── test_rag.py         # RAG pipeline test runner
├── tests/
│   ├── test_cleaning.py    # Cleaning unit tests
│   └── test_chunking.py    # Chunking unit tests
├── .env.example            # Environment template
├── fetch_data.py           # Data acquisition script
├── requirements.txt        # Project dependencies
└── README.md               # Documentation