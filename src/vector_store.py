import os
import time
import pandas as pd
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text_recursive(text, chunk_size=500, chunk_overlap=50):
    if not isinstance(text, str) or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def build_and_benchmark_vector_store(
    input_csv="data/processed/clean_dataset.csv",
    db_dir="data/vector_db",
    collection_name="knowledge_base",
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing file: {input_csv}")

    print(f"Reading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    os.makedirs(db_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_dir)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    documents, metadatas, ids = [], [], []
    total_chunks = 0

    print("\nProcessing text chunks and embeddings...")
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting Data"):
        doc_id = str(row.get("id", f"doc_{idx}"))
        title = str(row.get("title", ""))
        text = str(row.get("clean_text", ""))

        chunks = chunk_text_recursive(text, chunk_size=500, chunk_overlap=50)

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10:
                continue

            chunk_id = f"{doc_id}_c{chunk_idx}"
            documents.append(chunk)
            metadatas.append(
                {"parent_id": doc_id, "title": title, "chunk_index": chunk_idx}
            )
            ids.append(chunk_id)
            total_chunks += 1

            if len(documents) >= 500:
                collection.add(
                    documents=documents, metadatas=metadatas, ids=ids
                )
                documents, metadatas, ids = [], [], []

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    total_time = time.time() - start_time
    avg_chunk_time = (
        (total_time / total_chunks) * 1000 if total_chunks > 0 else 0
    )

    print("\n" + "=" * 40)
    print("      PROCESSING SUMMARY      ")
    print("=" * 40)
    print(f"Total Documents : {len(df)}")
    print(f"Total Chunks    : {total_chunks}")
    print(f"Elapsed Time    : {total_time:.2f}s")
    print(f"Avg Time/Chunk  : {avg_chunk_time:.2f} ms")
    print(f"Database Path   : {db_dir}")
    print("=" * 40 + "\n")


def semantic_search(query, top_k=3, db_dir="data/vector_db"):
    chroma_client = chromadb.PersistentClient(path=db_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_collection(
        name="knowledge_base", embedding_function=embedding_fn
    )

    t0 = time.time()
    results = collection.query(query_texts=[query], n_results=top_k)
    latency = (time.time() - t0) * 1000

    return results, latency


if __name__ == "__main__":
    build_and_benchmark_vector_store()