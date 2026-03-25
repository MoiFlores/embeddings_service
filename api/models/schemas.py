from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class IngestResponse(BaseModel):
    document_id: str
    chunks_stored: int
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Filtro opcional por metadatos
    filter_filename: Optional[str] = None


class SearchResult(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
