"""
NLP Metadata Enricher and Filtered Vector DB Ingestion Pipeline.
Handles short document edge cases and integrates topic/sentiment metadata into ChromaDB.
"""

import os
import pandas as pd
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

from src.topic_modeler import extract_corpus_topics
from src.nlp_analyzer import compute_rule_sentiment


def simple_character_chunker(text_content, chunk_size=500, chunk_overlap=50):
    """
    Splits text content into character-based recursive chunks with overlap.
    """
    if not text_content:
        return []
    
    chunks = []
    start = 0
    text_len = len(text_content)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text_content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += (chunk_size - chunk_overlap)

    return chunks


def handle_edge_case_and_clean(doc_text, min_length=20):
    """
    Handles short documents and technical jargon edge cases.
    """
    if not isinstance(doc_text, str) or len(doc_text.strip()) < min_length:
        return None
    return doc_text.strip()


def build_enriched_vector_store(
    csv_path="data/processed/clean_dataset.csv",
    db_directory="data/vector_db"
):
    """
    Applies topic modeling & sentiment analysis, then stores documents
    with enriched metadata inside ChromaDB.
    """
    print("\n[Step 1] Extracting Topic Themes & Sentiments...")
    df, topic_map = extract_corpus_topics(csv_path)

    os.makedirs(db_directory, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_directory)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        chroma_client.delete_collection(name="enriched_knowledge_base")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="enriched_knowledge_base",
        embedding_function=embed_fn
    )

    documents_batch, metadatas_batch, ids_batch = [], [], []
    chunk_counter = 0

    print("\n[Step 2] Processing Text Chunks & Ingesting Metadata to ChromaDB...")
    target_col = 'clean_text' if 'clean_text' in df.columns else df.columns[-1]

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching & Indexing"):
        raw_text = row[target_col]
        doc_id = str(row.get('id', f"doc_{idx}"))
        doc_title = str(row.get('title', 'Unknown Document'))
        topic = str(row.get('topic_theme', 'General/Technical'))

        valid_text = handle_edge_case_and_clean(raw_text)
        if not valid_text:
            continue

        doc_sentiment = compute_rule_sentiment(valid_text)

        chunks = simple_character_chunker(valid_text, chunk_size=500, chunk_overlap=50)

        for c_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_c{c_idx}"
            
            documents_batch.append(chunk_text)
            metadatas_batch.append({
                "doc_title": doc_title,
                "topic": topic,
                "sentiment": doc_sentiment,
                "chunk_index": c_idx
            })
            ids_batch.append(chunk_id)
            chunk_counter += 1

            if len(documents_batch) >= 200:
                collection.add(
                    documents=documents_batch,
                    metadatas=metadatas_batch,
                    ids=ids_batch
                )
                documents_batch, metadatas_batch, ids_batch = [], [], []

    if documents_batch:
        collection.add(
            documents=documents_batch,
            metadatas=metadatas_batch,
            ids=ids_batch
        )

    print(f"\nSuccessfully indexed {chunk_counter} enriched chunks with NLP metadata!\n")
    return collection


def filtered_semantic_search(query_text, topic_filter=None, sentiment_filter=None, top_k=3, db_directory="data/vector_db"):
    """
    Executes semantic vector search with metadata filtering.
    """
    chroma_client = chromadb.PersistentClient(path=db_directory)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_collection(
        name="enriched_knowledge_base",
        embedding_function=embed_fn
    )

    where_clause = {}
    if topic_filter:
        where_clause["topic"] = topic_filter
    if sentiment_filter:
        where_clause["sentiment"] = sentiment_filter

    if where_clause:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_clause
        )
    else:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

    return results


if __name__ == "__main__":
    build_enriched_vector_store()