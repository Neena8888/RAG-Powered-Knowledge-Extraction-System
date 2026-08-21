"""
Unit Tests for FastAPI RAG Endpoints using Pytest and TestClient.
Verifies HTTP status codes (200, 400, 422, 500) and response structures.
"""

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify that the health check endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"
    assert "service" in data


def test_query_endpoint_valid_payload():
    """Verify that POST /query with a valid technical query returns 200 and proper schema."""
    payload = {
        "query": "What is machine learning?",
        "top_k": 2
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    # Check that any of the latency metrics are present in response
    assert any("latency" in k for k in data.keys())


def test_query_endpoint_empty_query_bad_request():
    """Verify that POST /query with whitespace-only string triggers 400 or 422 error."""
    payload = {
        "query": "   ",
        "top_k": 2
    }
    response = client.post("/query", json=payload)
    assert response.status_code in [400, 422]


def test_query_endpoint_invalid_schema():
    """Verify that POST /query with invalid payload types returns 422 Unprocessable Entity."""
    payload = {
        "top_k": "invalid_number"
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_filtered_search_endpoint():
    """Verify that POST /search/filtered returns 200 and filtered result structure."""
    payload = {
        "query": "artificial intelligence",
        "sentiment_filter": "Positive",
        "top_k": 2
    }
    response = client.post("/search/filtered", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "artificial intelligence"
    assert data["sentiment_filter"] == "Positive"
    assert "results" in data