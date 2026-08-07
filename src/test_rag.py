"""
Test Suite & End-to-End Execution Script for Week 3 RAG Pipeline.
"""

from src.rag_pipeline import run_rag_pipeline


def execute_test_suite():
    test_queries = [
        "What is artificial intelligence and machine learning?",  # In-Domain Query
        "Explain relational database management systems and SQL.", # In-Domain Query
        "What is the best recipe for cooking biryani?",            # Out-of-Domain Query
    ]

    print("\n========================================================")
    print("        WEEK 3 RAG SYSTEM BENCHMARKING SUITE           ")
    print("========================================================")

    for idx, query in enumerate(test_queries, 1):
        print(f"\n>>> TEST CASE {idx} <<<")
        run_rag_pipeline(query)


if __name__ == "__main__":
    execute_test_suite()