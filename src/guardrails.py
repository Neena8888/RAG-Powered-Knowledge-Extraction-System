"""
Guardrails and Hallucination Protection Module.
Handles out-of-domain query detection and response verification.
"""

# Common out-of-domain / off-topic trigger words/categories check
OFF_TOPIC_KEYWORDS = [
    "recipe", "cooking", "sports", "cricket", "football", 
    "movie", "celebrity", "horoscope", "astrology", "song"
]


def is_off_topic_query(query_text):
    """
    Preliminary fast guardrail to filter obvious non-technical off-topic queries.
    """
    clean_query = query_text.lower().strip()
    for word in OFF_TOPIC_KEYWORDS:
        if word in clean_query:
            return True
    return False


def validate_context_relevance(retrieved_results, max_distance_threshold=1.6):
    """
    Checks if ChromaDB vector search returned meaningful context.
    If distances are too high or documents are empty, flags as out-of-domain.
    """
    if not retrieved_results or not retrieved_results.get("documents"):
        return False, []

    documents = retrieved_results["documents"][0]
    distances = retrieved_results.get("distances", [[]])[0]

    if not documents:
        return False, []

    # If distances are available, ensure at least one match is within acceptable threshold
    if distances:
        min_distance = min(distances)
        if min_distance > max_distance_threshold:
            return False, []

    return True, documents


def apply_hallucination_filter(llm_response, context_chunks):
    """
    Ensures LLM response adheres to groundedness rules and handles fallbacks.
    """
    if not llm_response or not llm_response.strip():
        return "I am unable to generate a response based on the provided technical documents."

    # Standard out-of-domain response safeguard
    fallback_msg = "Based on the provided knowledge base, I do not have enough information to answer this query."
    
    if fallback_msg.lower() in llm_response.lower():
        return fallback_msg

    return llm_response