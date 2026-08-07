"""
End-to-End RAG Pipeline with Latency Logging & Guardrails.
Integrates Retrieval, Prompt Engineering, LLM Generation, and Safety Checks.
"""

import time
from src.vector_store import semantic_search
from src.llm_client import call_openrouter_llm_safe, RAGAPIError
from src.prompt_template import build_rag_prompt
from src.guardrails import (
    is_off_topic_query,
    validate_context_relevance,
    apply_hallucination_filter,
)


def run_rag_pipeline(user_query, top_k=3):
    """
    Executes full RAG workflow and measures granular latency metrics.
    """
    pipeline_start = time.time()
    print(f"\n========================================================")
    print(f"PROCESSING RAG QUERY: '{user_query}'")
    print(f"========================================================")

    # 1. Fast Off-Topic Guardrail Check
    if is_off_topic_query(user_query):
        total_time = (time.time() - pipeline_start) * 1000
        print("[Guardrail] Flagged as out-of-domain query.")
        return {
            "query": user_query,
            "answer": "Out-of-domain query detected. Please ask technical computer science questions related to the knowledge base.",
            "retrieval_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "total_latency_ms": total_time,
        }

    # 2. Step 1: Retrieval Phase (Vector Search)
    retrieval_start = time.time()
    search_results, _ = semantic_search(user_query, top_k=top_k)
    retrieval_latency = (time.time() - retrieval_start) * 1000

    # Validate Context Relevance
    is_relevant, context_chunks = validate_context_relevance(search_results)

    if not is_relevant or not context_chunks:
        total_time = (time.time() - pipeline_start) * 1000
        print("[Guardrail] Insufficient context relevance found in ChromaDB.")
        return {
            "query": user_query,
            "answer": "Based on the provided knowledge base, I do not have enough information to answer this query.",
            "retrieval_latency_ms": retrieval_latency,
            "generation_latency_ms": 0.0,
            "total_latency_ms": total_time,
        }

    # 3. Step 2: Prompt Engineering (Context Injection)
    system_prompt, user_prompt = build_rag_prompt(user_query, context_chunks)

    # 4. Step 3: Generation Phase (LLM Call with Error Handling)
    generation_start = time.time()
    try:
        raw_llm_response = call_openrouter_llm_safe(
            prompt_text=user_prompt,
            system_instruction=system_prompt,
        )
    except RAGAPIError as err:
        raw_llm_response = f"API Error: Unable to fetch response due to '{err}'"

    generation_latency = (time.time() - generation_start) * 1000

    # 5. Step 4: Hallucination Safeguard
    final_answer = apply_hallucination_filter(raw_llm_response, context_chunks)

    total_latency = (time.time() - pipeline_start) * 1000

    # Performance & Latency Summary Report
    print(f"\n--- LATENCY METRICS BREAKDOWN ---")
    print(f"Retrieval Time  : {retrieval_latency:.2f} ms")
    print(f"Generation Time : {generation_latency:.2f} ms")
    print(f"Total Pipeline  : {total_latency:.2f} ms")
    print(f"---------------------------------")
    print(f"ANSWER:\n{final_answer}\n")

    return {
        "query": user_query,
        "answer": final_answer,
        "retrieval_latency_ms": retrieval_latency,
        "generation_latency_ms": generation_latency,
        "total_latency_ms": total_latency,
    }


if __name__ == "__main__":
    # Test Query Execution
    test_q = "What is artificial intelligence and machine learning?"
    run_rag_pipeline(test_q)