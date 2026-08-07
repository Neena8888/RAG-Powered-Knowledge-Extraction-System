"""
Prompt Engineering Module for RAG Pipeline.
Implements system prompts and context injection logic.
"""

SYSTEM_PROMPT = """You are a precise, factual AI Assistant for a technical RAG Knowledge Extraction System.

CRITICAL INSTRUCTIONS:
1. Answer the user's question using ONLY the provided context blocks below.
2. Do NOT use external knowledge, assumptions, or extrapolations.
3. If the answer cannot be found or deduced directly from the context, state clearly: "Based on the provided knowledge base, I do not have enough information to answer this query."
4. Keep the answer concise, technical, clear, and directly relevant to the question."""


def build_rag_prompt(user_query, context_chunks):
    """
    Constructs a structured prompt by injecting retrieved text context blocks into the template.
    """
    if not context_chunks:
        formatted_context = "No relevant context found in knowledge base."
    else:
        formatted_blocks = []
        for idx, chunk in enumerate(context_chunks, 1):
            formatted_blocks.append(f"--- CONTEXT BLOCK {idx} ---\n{chunk.strip()}")
        formatted_context = "\n\n".join(formatted_blocks)

    user_prompt = f"""Use the following context snippets to answer the question at the end.

=== START OF CONTEXT ===
{formatted_context}
=== END OF CONTEXT ===

QUESTION: {user_query}

ANSWER:"""

    return SYSTEM_PROMPT, user_prompt