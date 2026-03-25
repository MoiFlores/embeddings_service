"""
Wrapper sobre el cliente Qdrant.
Crea la colección si no existe y expone métodos de upsert y búsqueda.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from functools import lru_cache
from config import settings
import logging
import uuid

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_collection(client)
    return client


def _ensure_collection(client: QdrantClient) -> None:
    """Crea la colección si no existe."""
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Colección '{settings.qdrant_collection}' creada")
    else:
        logger.info(f"Colección '{settings.qdrant_collection}' ya existe")


def upsert_chunks(
    document_id: str,
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """
    Almacena chunks con sus embeddings en Qdrant.
    Retorna la cantidad de puntos insertados.
    """
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def search(
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.5,
    filter_filename: str | None = None,
) -> list[dict]:
    """
    Busca los chunks más similares al vector de query.
    Filtra opcionalmente por nombre de archivo.
    """
    client = get_client()

    query_filter = None
    if filter_filename:
        query_filter = Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filter_filename))]
        )

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "document_id": r.payload["document_id"],
            "filename": r.payload["filename"],
            "chunk_index": r.payload["chunk_index"],
            "text": r.payload["text"],
            "score": r.score,
        }
        for r in results
    ]
