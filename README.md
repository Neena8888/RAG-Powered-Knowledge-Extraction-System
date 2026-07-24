# Data Ingestion & Cleaning Pipeline for Knowledge Extraction

A production-ready data collection, sanitization, and quality validation pipeline built for Week 1 of the Parallax Labs Internship.

---

## 📌 Project Overview

This repository contains the complete implementation for Week 1, focusing on acquiring a large-scale real-world technical dataset (5,000+ records), implementing robust text cleaning logic for edge cases, validating data quality, and exporting a clean dataset ready for chunking and embedding.

---

## 🎯 Week 1 Objectives & Deliverables Completed

- **Environment Setup & Verification:** Installed required packages (`pandas`, `numpy`, `requests`, `tqdm`, `datasets`, `pytest`) and created a verification script (`src/utils.py`) to test imports and core library functionality.
- **Large Data Acquisition:** Developed a bulk fetching engine (`fetch_data.py`) utilizing the Wikipedia MediaWiki API and stream concepts to acquire **5,100 real-world technical documents** across core Computer Science domains. Validated data for encoding issues and null values.
- **Robust Text Cleaning Logic:** Built custom sanitization pipeline (`src/cleaning.py`) handling edge cases such as HTML tags, URLs, non-ASCII/foreign characters, non-printable symbols, and multi-line whitespaces.
- **Unit Testing & Data Quality:** Developed unit tests (`tests/test_cleaning.py`) using `pytest` to test edge-case cleaning functions and generated automated Data Quality Reports.
- **Clean Dataset Export:** Processed raw inputs and generated a validated output CSV (`data/processed/clean_dataset.csv`) free of nulls, duplicates, and invalid records.

---

## 💡 Technical Approach & Methodology

1. **Environment Verification:** Implemented functional sanity checks for core dependencies (`pandas`, `datasets`, `pytest`) before data processing to ensure runtime stability.
2. **API Ingestion Engine:** Queried Wikipedia API across high-density technology domains (AI, Machine Learning, Cybersecurity, Databases) in bulk batches to collect over 5,000 unique records dynamically.
3. **Sequential Regex Sanitization:** Designed a custom step-by-step cleaner handling specific edge cases:
   - Stripping HTML/XML tags and web links (`http`/`https`).
   - Filtering non-ASCII and foreign language text to prevent downstream embedding issues.
   - Normalizing line breaks and excessive whitespace into clean single spaces.
4. **Quality Filtering & Edge Drop:** Dropped empty strings and records shorter than 20 characters post-sanitization along with duplicate entries.
5. **Deterministic Testing:** Verified edge cases (HTML, URLs, whitespace, non-ASCII) using `pytest` suite execution.

---

## 📊 Data Quality Report Summary

| Metric / Parameter | Value / Status |
| :--- | :--- |
| **Total Raw Input Documents** | 5,100 |
| **Cleaned & Retained Documents** | 5,095 |
| **Dropped Edge Cases / Duplicates** | 5 |
| **Missing / Null Values** | 0 across all columns |
| **Cleaned Output File Path** | `data/processed/clean_dataset.csv` |

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── raw/                # Raw downloaded dataset (5,100 documents)
│   └── processed/          # Clean dataset ready for chunking & embedding
├── src/
│   ├── __init__.py
│   ├── cleaning.py         # Text cleaning and data quality reporting module
│   └── utils.py            # Environment verification script
├── tests/
│   └── test_cleaning.py    # Pytest unit tests for cleaning logic
├── fetch_data.py           # Bulk data acquisition script
├── main.py
├── requirements.txt        # Project dependencies
└── README.md               # Documentation and execution guide