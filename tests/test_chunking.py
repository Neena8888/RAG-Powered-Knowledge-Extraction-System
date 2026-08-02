"""
Unit tests for text chunking strategy.
"""

from src.vector_store import chunk_text_recursive


def test_recursive_chunking_normal():
    sample_text = "Data science is an interdisciplinary field. " * 30
    chunks = chunk_text_recursive(sample_text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert isinstance(chunks[0], str)


def test_recursive_chunking_empty_and_null():
    assert chunk_text_recursive("") == []
    assert chunk_text_recursive(None) == []


def test_recursive_chunking_overlap():
    sample_text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
    chunks = chunk_text_recursive(sample_text, chunk_size=30, chunk_overlap=10)
    assert len(chunks) >= 2