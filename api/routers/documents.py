from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import IngestResponse
from services import extractor, embedder, vector_store
import uuid
import logging

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {
    "application/pdf": extractor.extract_text_from_pdf,
    "image/png": extractor.extract_text_from_image,
    "image/jpeg": extractor.extract_text_from_image,
    "image/tiff": extractor.extract_text_from_image,
    "text/plain": extractor.extract_text_from_txt,
}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Recibe un archivo, extrae su texto, genera embeddings
    y los almacena en Qdrant.
    """
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de archivo no soportado: {content_type}. "
                   f"Permitidos: {list(ALLOWED_TYPES.keys())}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío")

    # 1. Extraer texto
    try:
        extract_fn = ALLOWED_TYPES[content_type]
        text = extract_fn(file_bytes)
    except Exception as e:
        logger.error(f"Error extrayendo texto de {file.filename}: {e}")
        raise HTTPException(status_code=422, detail=f"No se pudo extraer texto: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No se pudo extraer texto del documento")

    # 2. Dividir en chunks
    chunks = extractor.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="El documento no tiene contenido procesable")

    # 3. Generar embeddings
    try:
        embeddings = embedder.embed_texts(chunks)
    except Exception as e:
        logger.error(f"Error generando embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error en embeddings: {str(e)}")

    # 4. Almacenar en Qdrant
    document_id = str(uuid.uuid4())
    try:
        stored = vector_store.upsert_chunks(
            document_id=document_id,
            filename=file.filename or "unknown",
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error(f"Error almacenando en Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Error almacenando vectores: {str(e)}")

    return IngestResponse(
        document_id=document_id,
        chunks_stored=stored,
        message=f"Documento ingresado correctamente con {stored} chunks",
    )
