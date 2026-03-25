from fastapi import APIRouter, HTTPException
from models.schemas import SearchRequest, SearchResponse, SearchResult
from services import embedder, vector_store
import logging

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Busca los chunks más relevantes para una query en lenguaje natural.
    """
    try:
        query_vector = embedder.embed_query(request.query)
    except Exception as e:
        logger.error(f"Error generando embedding de query: {e}")
        raise HTTPException(status_code=500, detail=f"Error en embedding: {str(e)}")

    try:
        raw_results = vector_store.search(
            query_vector=query_vector,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filter_filename=request.filter_filename,
        )
    except Exception as e:
        logger.error(f"Error en búsqueda Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")

    results = [SearchResult(**r) for r in raw_results]
    return SearchResponse(query=request.query, results=results, total=len(results))
