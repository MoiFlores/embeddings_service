"""
Singleton que carga el modelo UNA sola vez al iniciar la app
y reutiliza la instancia en todos los requests.
"""

from sentence_transformers import SentenceTransformer
from functools import lru_cache
from config import settings
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info(f"Cargando modelo: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Modelo cargado correctamente")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Genera embeddings para una lista de textos.
    Retorna lista de vectores float32.
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embedding para una query de búsqueda (un solo texto)."""
    return embed_texts([query])[0]
