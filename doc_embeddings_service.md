# Document Embeddings Service — Especificación de Proyecto

> Prompt/spec completo para Cursor. Lee todo antes de generar código.

---

## Objetivo

Construir un microservicio en Python con FastAPI que:
1. Reciba documentos (PDF, imágenes, texto plano)
2. Extraiga el texto (OCR si es necesario)
3. Genere embeddings con un modelo ligero de HuggingFace
4. Almacene y consulte los embeddings en Qdrant
5. Todo corra dentro de Docker sin depender de APIs externas de pago

---

## Stack tecnológico

| Capa | Tecnología | Motivo |
|------|-----------|--------|
| API | FastAPI + Uvicorn | Async nativo, tipado, OpenAPI gratis |
| Extracción PDF | PyMuPDF (fitz) | Rápido, sin dependencias pesadas |
| OCR imágenes | pytesseract + Pillow | Ligero, open source |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 120 MB, multilingüe, CPU-friendly |
| Vector DB | Qdrant (Docker oficial) | REST + gRPC, filtros, sin costo |
| Infra | Docker Compose | Orquestación local y en servidor |

---

## Estructura de directorios

```
doc-embeddings-service/
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # Entrypoint FastAPI
│   ├── config.py                # Settings desde variables de entorno
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── documents.py         # POST /documents/ingest
│   │   └── search.py            # POST /search
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extractor.py         # Extracción de texto (PDF + OCR)
│   │   ├── embedder.py          # Generación de embeddings
│   │   └── vector_store.py      # Cliente Qdrant
│   │
│   └── models/
│       ├── __init__.py
│       └── schemas.py           # Pydantic models
│
└── scripts/
    └── create_collection.py     # Script de inicialización de Qdrant
```

---

## Variables de entorno

Archivo `.env.example` (crear `.env` a partir de este):

```env
# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=documents

# Modelo de embeddings (NO cambiar a menos que se ajuste VECTOR_SIZE)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_SIZE=384

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info

# Cache del modelo (mapeado a volumen Docker)
HF_HOME=/app/.cache/huggingface
TRANSFORMERS_CACHE=/app/.cache/huggingface
```

---

## docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: embeddings-api
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - hf_cache:/app/.cache/huggingface
    depends_on:
      qdrant:
        condition: service_healthy
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  hf_cache:       # Modelo descargado persiste entre reinicios
  qdrant_data:    # Embeddings almacenados persisten
```

---

## api/Dockerfile

```dockerfile
FROM python:3.11-slim

# Dependencias del sistema para tesseract y PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo en la imagen para evitar descarga en runtime
# Si usas volumen de caché, esto acelera el primer arranque
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
```

---

## api/requirements.txt

```txt
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
pydantic-settings==2.3.1

# Extracción de documentos
PyMuPDF==1.24.5
pytesseract==0.3.10
Pillow==10.3.0

# Embeddings
sentence-transformers==3.0.1
torch==2.3.1+cpu  --extra-index-url https://download.pytorch.org/whl/cpu

# Qdrant
qdrant-client==1.10.1
```

> **Nota para Cursor**: instalar torch CPU (no GPU) para mantener la imagen ligera. Si el servidor tiene GPU, cambiar a `torch==2.3.1` sin el sufijo `+cpu`.

---

## api/config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_size: int = 384

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## api/models/schemas.py

```python
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
```

---

## api/services/extractor.py

```python
"""
Extrae texto de PDFs e imágenes.
- PDF con texto digital: PyMuPDF (rápido, sin OCR)
- PDF escaneado / imagen: pytesseract
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from typing import Generator

CHUNK_SIZE = 500        # caracteres por chunk
CHUNK_OVERLAP = 50      # solapamiento entre chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrae texto de un PDF. Si una página no tiene texto, aplica OCR."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []

    for page in doc:
        text = page.get_text().strip()
        if text:
            full_text.append(text)
        else:
            # Página escaneada: renderizar y aplicar OCR
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img, lang="spa+eng")
            full_text.append(ocr_text.strip())

    doc.close()
    return "\n".join(full_text)


def extract_text_from_image(file_bytes: bytes) -> str:
    """OCR directo sobre una imagen."""
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img, lang="spa+eng")


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en chunks con solapamiento."""
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

    return [c for c in chunks if c]  # Filtrar chunks vacíos
```

---

## api/services/embedder.py

```python
"""
Singleton que carga el modelo UNA sola vez al iniciar la app
y reutiliza la instancia en todos los requests.
"""

from sentence_transformers import SentenceTransformer
from functools import lru_cache
from api.config import settings
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
```

---

## api/services/vector_store.py

```python
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
from api.config import settings
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
```

---

## api/routers/documents.py

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.models.schemas import IngestResponse
from api.services import extractor, embedder, vector_store
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
```

---

## api/routers/search.py

```python
from fastapi import APIRouter, HTTPException
from api.models.schemas import SearchRequest, SearchResponse, SearchResult
from api.services import embedder, vector_store
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
```

---

## api/main.py

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routers import documents, search
from api.services.embedder import get_model
from api.services.vector_store import get_client
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
```

---

## Endpoints disponibles

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/documents/ingest` | Subir y procesar documento |
| `POST` | `/search/` | Buscar por texto |
| `GET` | `/docs` | Swagger UI (FastAPI automático) |

### Ejemplo: ingestar un PDF

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@mi_documento.pdf"
```

### Ejemplo: buscar

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "¿qué dice el contrato sobre penalizaciones?", "top_k": 3}'
```

---

## Cómo correr el proyecto

```bash
# 1. Clonar / crear estructura de directorios
# 2. Copiar variables de entorno
cp .env.example .env

# 3. Levantar todo
docker compose up --build

# 4. Verificar que todo esté corriendo
curl http://localhost:8000/health        # API
curl http://localhost:6333/healthz       # Qdrant

# 5. Ver logs
docker compose logs -f api
```

---

## Consideraciones importantes para Cursor

- **No instalar GPU torch** a menos que el servidor tenga GPU. Usar siempre `torch+cpu` para mantener la imagen por debajo de 1.5 GB.
- **El modelo se descarga una sola vez** en el build de Docker y se cachea en el volumen `hf_cache`. No se vuelve a descargar en cada arranque.
- **Chunk size configurable**: los valores `CHUNK_SIZE=500` y `CHUNK_OVERLAP=50` en `extractor.py` son un buen punto de partida. Para documentos muy largos se puede aumentar a 1000/100.
- **Qdrant corre en modo single-node** (sin cluster). Si se necesita escalar, Qdrant tiene modo distribuido pero requiere licencia para producción con más de un nodo.
- **Idiomas OCR**: el Dockerfile instala `tesseract-ocr-spa` (español) y `tesseract-ocr-eng` (inglés). Agregar más idiomas con `tesseract-ocr-{lang}` si se necesita.
- **El modelo `paraphrase-multilingual-MiniLM-L12-v2`** produce vectores de dimensión **384**. Si se cambia a otro modelo, actualizar `VECTOR_SIZE` en `.env` y recrear la colección en Qdrant (`docker compose down -v` para borrar el volumen y recrearla).

---

## Modelo alternativo más ligero

Si 120 MB es aún mucho, usar:

```
all-MiniLM-L6-v2  →  80 MB, solo inglés, dim=384
```

Cambiar en `.env`:
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_SIZE=384
```