"""
Week 4 NLP Pipeline Test Suite & Metadata Retrieval Benchmarking.
"""

from src.nlp_analyzer import evaluate_sentiment_accuracy
from src.nlp_enricher import build_enriched_vector_store, filtered_semantic_search


def run_week4_evaluations():
    print("\n" + "=" * 60)
    print("      WEEK 4: NLP ANALYSIS & METADATA SEARCH EVALUATION     ")
    print("=" * 60)

    # 1. Run Sentiment Analysis Accuracy Validation
    print("\n>>> TASK 1: SENTIMENT ANALYSIS VALIDATION ACCURACY <<<")
    acc = evaluate_sentiment_accuracy()

    # 2. Build Enriched Vector DB with Topic & Sentiment Metadata
    print("\n>>> TASK 2: BUILDING ENRICHED CHROMADB VECTOR STORE <<<")
    build_enriched_vector_store()

    # 3. Benchmark Filtered Retrieval Search
    print("\n>>> TASK 3: METADATA FILTERED VECTOR RETRIEVAL TEST <<<")
    test_query = "What are the core concepts of artificial intelligence and machine learning?"
    
    print(f"\n[Query]: '{test_query}'")
    print("Executing Filtered Search (Filter: Sentiment = Positive)...")
    
    try:
        results = filtered_semantic_search(
            query_text=test_query,
            sentiment_filter="Positive",
            top_k=2
        )
        
        if results and results.get("documents") and results["documents"][0]:
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                print(f"\n  Result #{i}:")
                print(f"  - Title: {meta.get('doc_title')}")
                print(f"  - Topic Theme: {meta.get('topic')}")
                print(f"  - Sentiment: {meta.get('sentiment')}")
                print(f"  - Content Snippet: {doc[:120]}...")
        else:
            print("  No matching filtered documents found.")
            
    except Exception as err:
        print(f"  Search error: {err}")

    print("\n" + "=" * 60)
    print("           WEEK 4 NLP PIPELINE EVALUATION COMPLETE          ")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_week4_evaluations()