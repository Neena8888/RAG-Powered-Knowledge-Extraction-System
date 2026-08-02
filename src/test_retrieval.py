import time
from src.vector_store import semantic_search


def run_retrieval_benchmarks():
    test_queries = [
        "What is artificial intelligence and machine learning?",
        "How do computer networks and cybersecurity protocols function?",
        "Explain relational database management systems and SQL.",
        "What are deep learning neural networks?",
        "How does software engineering principles apply to modern systems?",
    ]

    print("\n" + "=" * 50)
    print("       RETRIEVAL LATENCY BENCHMARK       ")
    print("=" * 50)

    total_time = 0

    for idx, query in enumerate(test_queries, 1):
        results, latency = semantic_search(query, top_k=2)
        total_time += latency

        print(f"\n[Query {idx}]: {query}")
        print(f"Latency  : {latency:.2f} ms")

        if results and results["documents"] and results["documents"][0]:
            title = results["metadatas"][0][0].get("title", "N/A")
            snippet = results["documents"][0][0][:100].replace("\n", " ")
            print(f"Top Match: [{title}] -> {snippet}...")

    avg_time = total_time / len(test_queries)
    print("\n" + "=" * 50)
    print(f"Average Search Latency: {avg_time:.2f} ms")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_retrieval_benchmarks()