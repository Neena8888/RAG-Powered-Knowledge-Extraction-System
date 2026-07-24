import json
import os
import urllib.request
import pandas as pd
from tqdm import tqdm


def fetch_wikipedia_dataset(target_docs=5100):
    print(f"Fetching {target_docs}+ real-world documents in bulk...\n")

    # High-density tech topics to query
    queries = [
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Computer science",
        "Data science",
        "Software engineering",
        "Computer network",
        "Cybersecurity",
        "Algorithm",
        "Database",
        "Operating system",
        "Programming language",
        "Big data",
        "Cloud computing",
        "Information technology",
    ]
    records = []
    seen_ids = set()
    headers = {"User-Agent": "RAG-Knowledge-System/1.0 (academic_research)"}

    pbar = tqdm(total=target_docs, desc="Downloading Documents")

    for q in queries:
        if len(records) >= target_docs:
            break

        # Query Wikipedia API for 500 search results at once
        encoded_q = urllib.parse.quote(q)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&srlimit=500&format=json"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                search_results = data.get("query", {}).get("search", [])

                for item in search_results:
                    page_id = item.get("pageid")
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")

                    # Basic HTML tag strip from snippet
                    clean_text = (
                        snippet.replace('<span class="searchmatch">', "")
                        .replace("</span>", "")
                        .strip()
                    )

                    if page_id not in seen_ids and len(clean_text) > 30:
                        seen_ids.add(page_id)
                        records.append(
                            {
                                "id": f"wiki_{page_id}",
                                "title": title,
                                "text": clean_text,
                                "source": "Wikipedia",
                            }
                        )
                        pbar.update(1)

                        if len(records) >= target_docs:
                            break
        except Exception:
            continue

    pbar.close()

    df = pd.DataFrame(records)

    print("\n--- Basic Validation Summary ---")
    print(f"Total Rows Downloaded : {len(df)}")
    print(f"Null Values per Column:\n{df.isnull().sum()}")

    os.makedirs("data/raw", exist_ok=True)
    save_path = "data/raw/raw_dataset.csv"
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\n[SUCCESS] Raw dataset saved to '{save_path}'")


if __name__ == "__main__":
    fetch_wikipedia_dataset()