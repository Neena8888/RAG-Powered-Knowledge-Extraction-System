"""
FastAPI Service for RAG Knowledge Extraction System.
Includes structured logging, request validation, and standard HTTP error handling.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag_pipeline import run_rag_pipeline
from src.nlp_enricher import filtered_semantic_search

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="RAG Knowledge Extraction API",
    description="Production-ready FastAPI service for semantic retrieval, NLP enrichment, and RAG generation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Schemas ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The user query text.")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve.")


class FilteredSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Search query string.")
    topic_filter: Optional[str] = Field(default=None, description="Topic metadata filter.")
    sentiment_filter: Optional[str] = Field(default=None, description="Sentiment metadata filter (e.g. Positive, Neutral, Negative).")
    top_k: Optional[int] = Field(default=3, ge=1, le=10)


# --- Middleware for Request Timing & Structured Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Completed {request.method} {request.url.path} - Status: {response.status_code} in {duration_ms:.2f}ms")
        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Unhandled Exception on {request.method} {request.url.path} after {duration_ms:.2f}ms: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error. Please review server logs."}
        )


# --- Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """Service health verification endpoint."""
    return {"status": "healthy", "service": "RAG Knowledge Extraction Engine"}


@app.post("/query", status_code=status.HTTP_200_OK, tags=["RAG Engine"])
async def execute_rag_query(payload: QueryRequest):
    """
    Executes full end-to-end RAG pipeline with safety guardrails and latency tracking.
    """
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be whitespace only."
        )

    try:
        result = run_rag_pipeline(user_query=clean_query, top_k=payload.top_k)
        return result
    except Exception as err:
        logger.error(f"Error during RAG pipeline execution: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(err)}"
        )


@app.post("/search/filtered", status_code=status.HTTP_200_OK, tags=["Vector Retrieval"])
async def execute_filtered_search(payload: FilteredSearchRequest):
    """
    Performs metadata-filtered semantic search against ChromaDB.
    """
    try:
        search_results = filtered_semantic_search(
            query_text=payload.query.strip(),
            topic_filter=payload.topic_filter,
            sentiment_filter=payload.sentiment_filter,
            top_k=payload.top_k
        )
        return {
            "query": payload.query,
            "topic_filter": payload.topic_filter,
            "sentiment_filter": payload.sentiment_filter,
            "results": search_results
        }
    except Exception as err:
        logger.error(f"Filtered search failure: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector retrieval search failed."
        )