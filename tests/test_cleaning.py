"""
Unit Tests for Data Cleaning Functions
Tests edge cases: HTML tags, URLs, non-ASCII/foreign characters, empty values, and formatting.
"""

from src.cleaning import sanitize_text


def test_html_tag_removal():
    raw_text = "<h1>Artificial Intelligence</h1> <p>This is <b>bold</b> text.</p>"
    expected = "Artificial Intelligence This is bold text."
    assert sanitize_text(raw_text) == expected


def test_url_stripping():
    raw_text = "Read more at https://wikipedia.org or http://arxiv.org for details."
    expected = "Read more at or for details."
    assert sanitize_text(raw_text) == expected


def test_non_ascii_and_mixed_languages():
    raw_text = "Machine Learning こんにちは Systems"
    expected = "Machine Learning Systems"
    assert sanitize_text(raw_text) == expected


def test_whitespace_and_empty_inputs():
    assert sanitize_text("   Too   many    spaces   ") == "Too many spaces"
    assert sanitize_text("") == ""
    assert sanitize_text(None) == ""