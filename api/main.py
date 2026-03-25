from fastapi import FastAPI
from contextlib import asynccontextmanager
from routers import documents, search
from services.embedder import get_model
from services.vector_store import get_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar modelo y conexión a Qdrant al arrancar (no en el primer request)
    logger.info("Iniciando servicio...")
    get_model()       # Carga el modelo en memoria
    get_client()      # Verifica conexión con Qdrant y crea colección si no existe
    logger.info("Servicio listo")
    yield
    logger.info("Servicio detenido")


app = FastAPI(
    title="Document Embeddings Service",
    description="Ingesta documentos (PDF, imágenes, texto) y los indexa como embeddings en Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(search.router)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
