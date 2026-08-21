"""
Automated Evaluation Suite for RAG Retrieval & Generation Quality.
Calculates Precision@K, Recall@K, and End-to-End Latency Metrics.
"""

import time
import chromadb
from chromadb.utils import embedding_functions
from src.rag_pipeline import run_rag_pipeline


def evaluate_retrieval_metrics(top_k=3, db_directory="data/vector_db"):
    """
    Computes Precision@K and Recall@K on a curated technical evaluation test set.
    """
    test_suite = [
        {
            "query": "artificial intelligence and machine learning algorithms",
            "relevant_keywords": ["artificial intelligence", "machine learning", "algorithm", "neural", "learning"]
        },
        {
            "query": "relational database management systems and SQL queries",
            "relevant_keywords": ["database", "relational", "sql", "table", "query", "schema"]
        },
        {
            "query": "computer network protocols and communication layers",
            "relevant_keywords": ["network", "protocol", "communication", "tcp", "ip", "packet"]
        }
    ]

    chroma_client = chromadb.PersistentClient(path=db_directory)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Connect to collection (supports both enriched and standard collections)
    try:
        collection = chroma_client.get_collection(name="enriched_knowledge_base", embedding_function=embed_fn)
    except Exception:
        collection = chroma_client.get_collection(name="knowledge_base", embedding_function=embed_fn)

    precisions = []
    recalls = []

    print("\n" + "=" * 60)
    print(f"       RETRIEVAL BENCHMARK: PRECISION@{top_k} & RECALL@{top_k}       ")
    print("=" * 60)

    for idx, item in enumerate(test_suite, 1):
        query = item["query"]
        relevant_keys = item["relevant_keywords"]
        
        results = collection.query(query_texts=[query], n_results=top_k)
        retrieved_docs = results["documents"][0] if results["documents"] else []

        relevant_retrieved_count = 0
        for doc in retrieved_docs:
            doc_lower = doc.lower()
            if any(kw in doc_lower for kw in relevant_keys):
                relevant_retrieved_count += 1

        # Precision@K = (Relevant Retrieved) / K
        p_at_k = (relevant_retrieved_count / top_k) if top_k > 0 else 0.0
        # Recall@K = (Relevant Retrieved) / (Total Relevant Threshold estimated)
        estimated_total_relevant = min(len(relevant_keys), top_k)
        r_at_k = (relevant_retrieved_count / estimated_total_relevant) if estimated_total_relevant > 0 else 0.0

        precisions.append(p_at_k)
        recalls.append(r_at_k)

        print(f"Test #{idx}: '{query[:40]}...'")
        print(f"  - Precision@{top_k}: {p_at_k:.2f} | Recall@{top_k}: {r_at_k:.2f}")

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    print("-" * 60)
    print(f"Mean Precision@{top_k} : {avg_precision * 100:.2f}%")
    print(f"Mean Recall@{top_k}    : {avg_recall * 100:.2f}%")
    print("=" * 60)

    return avg_precision, avg_recall


def evaluate_end_to_end_generation():
    """
    Evaluates end-to-end generation quality, guardrail effectiveness, and latency.
    """
    queries = [
        ("What is machine learning?", True),
        ("How do relational databases store structured data?", True),
        ("Give me the recipe for delicious chicken biryani", False)
    ]

    print("\n" + "=" * 60)
    print("        END-TO-END GENERATION QUALITY & LATENCY SUITE        ")
    print("=" * 60)

    latencies = []
    for q_text, is_in_domain in queries:
        start_t = time.time()
        res = run_rag_pipeline(user_query=q_text, top_k=2)
        total_time = (time.time() - start_t) * 1000
        latencies.append(total_time)

        print(f"\nQuery: '{q_text}'")
        print(f"  - Latency: {total_time:.2f} ms")
        print(f"  - Answer Snippet: {res.get('answer', '')[:100]}...")
        if not is_in_domain:
            guardrail_flagged = "out-of-domain" in res.get("answer", "").lower() or "guardrail" in str(res).lower()
            print(f"  - Out-of-Domain Intercepted: {'✅ YES' if guardrail_flagged else '❌ NO'}")

    avg_lat = sum(latencies) / len(latencies)
    print("\n" + "-" * 60)
    print(f"Average Generation Latency: {avg_lat:.2f} ms")
    print("=" * 60 + "\n")
    return avg_lat


if __name__ == "__main__":
    evaluate_retrieval_metrics(top_k=3)
    evaluate_end_to_end_generation()