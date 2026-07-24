"""
Data Cleaning & Quality Reporting Module
"""

import os
import re
import pandas as pd


def sanitize_text(content: str) -> str:
    """Sanitizes text by removing HTML, URLs, non-ASCII, and normalizing whitespaces."""
    if not content or not isinstance(content, str):
        return ""

    # 1. Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", content)

    # 2. Strip URLs
    clean = re.sub(r"https?://\S+|www\.\S+", "", clean)

    # 3. Strip non-ASCII (Foreign language/emojis)
    clean = re.sub(r"[^\x00-\x7F]+", " ", clean)

    # 4. Remove special symbols
    clean = re.sub(r"[^\w\s.,!?-]", " ", clean)

    # 5. Normalize whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean


def generate_quality_report_and_clean_data(
    input_file: str = "data/raw/raw_dataset.csv",
    output_file: str = "data/processed/clean_dataset.csv",
):
    df = pd.read_csv(input_file)
    initial_count = len(df)

    # Apply Cleaning
    df["title"] = df["title"].fillna("").astype(str).apply(sanitize_text)
    df["clean_text"] = df["text"].fillna("").astype(str).apply(sanitize_text)

    # Filter Edge Cases (Drop texts shorter than 20 characters or duplicates)
    df = df[df["clean_text"].str.len() > 20].copy()
    df.drop_duplicates(subset=["clean_text"], inplace=True)

    # Select final columns
    df = df[["id", "title", "clean_text", "source"]]

    # Save Clean Output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")

    # Display Data Quality Report
    print("========================================")
    print("         DATA QUALITY REPORT            ")
    print("========================================")
    print(f"Total Raw Documents Input : {initial_count}")
    print(f"Cleaned & Retained Docs   : {len(df)}")
    print(f"Dropped Edge Cases/Dups   : {initial_count - len(df)}")
    print(f"Null Values Check         :\n{df.isnull().sum().to_string()}")
    print("========================================")
    print(f"Output File Saved At      : {output_file}\n")


if __name__ == "__main__":
    generate_quality_report_and_clean_data()