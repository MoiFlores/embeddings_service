# Document Embeddings Service

Microservicio en **Python + FastAPI** que permite ingestar documentos (PDF, imágenes, texto plano), extraer su texto (con OCR si es necesario), generar embeddings con un modelo de HuggingFace y almacenarlos/consultarlos en **Qdrant** (vector database). Todo corre dentro de Docker sin depender de APIs externas de pago.

---

## Stack Tecnológico

| Capa             | Tecnología                                                     | Motivo                                   |
|------------------|----------------------------------------------------------------|------------------------------------------|
| API              | FastAPI + Uvicorn                                              | Async nativo, tipado, OpenAPI automático  |
| Extracción PDF   | PyMuPDF (fitz)                                                 | Rápido, sin dependencias pesadas          |
| OCR imágenes     | pytesseract + Pillow                                           | Ligero, open source                       |
| Embeddings       | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  | 120 MB, multilingüe, CPU-friendly         |
| Vector DB        | Qdrant (Docker oficial)                                        | REST + gRPC, filtros, sin costo           |
| Infra            | Docker Compose                                                 | Orquestación local y en servidor          |

---

##  Estructura del Proyecto

```
embeddings_service/
├── docker-compose.yml          # Orquestación de servicios
├── .env.example                # Template de variables de entorno
├── .env                        # Variables de entorno activas (no se sube a git)
├── .gitignore
├── readme.md
├── doc_embeddings_service.md   # Especificación del proyecto
│
├── api/
│   ├── Dockerfile              # Imagen Docker de la API
│   ├── requirements.txt        # Dependencias Python
│   ├── main.py                 # Entrypoint FastAPI
│   ├── config.py               # Settings desde variables de entorno
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── documents.py        # POST /documents/ingest
│   │   └── search.py           # POST /search/
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extractor.py        # Extracción de texto (PDF + OCR)
│   │   ├── embedder.py         # Generación de embeddings
│   │   └── vector_store.py     # Cliente Qdrant
│   │
│   └── models/
│       ├── __init__.py
│       └── schemas.py          # Pydantic models
│
└── scripts/
    └── create_collection.py    # Script de inicialización de Qdrant
```

---

## Cómo Correr el Proyecto

### Prerrequisitos

- **Docker** y **Docker Compose** instalados en tu máquina.
  - [Instalar Docker Desktop (Windows/Mac)](https://www.docker.com/products/docker-desktop/)
  - [Instalar Docker Engine (Linux)](https://docs.docker.com/engine/install/)
- Mínimo **4 GB de RAM** disponibles para Docker (el modelo de embeddings requiere ~500 MB en memoria).

### Paso 1: Configurar variables de entorno

Copia el archivo de ejemplo y ajústalo si es necesario:

```bash
cp .env.example .env
```

> **Nota:** Las variables por defecto ya están configuradas para funcionar con Docker Compose. Solo modifica si necesitas cambiar puertos o el modelo de embeddings.

### Paso 2: Levantar los servicios

```bash
docker compose up --build
```

Esto hará lo siguiente:

1. **Construir la imagen de la API** — instala dependencias, tesseract OCR y descarga el modelo de embeddings (~120 MB).
2. **Levantar Qdrant** — base de datos vectorial en los puertos `6333` (REST) y `6334` (gRPC).
3. **Levantar la API** — disponible en el puerto `8000`.

> **La primera vez tarda varios minutos** porque se descarga el modelo de ML. Las siguientes ejecuciones serán mucho más rápidas gracias al caché en volumen Docker.

### Paso 3: Verificar que todo funciona

Espera a ver en los logs el mensaje `Servicio listo`, luego ejecuta:

```bash
# Verificar la API
curl http://localhost:8000/health
# Respuesta esperada: {"status":"ok"}

# Verificar Qdrant
curl http://localhost:6333/healthz
# Respuesta esperada: (texto o JSON indicando que está sano)
```

### Paso 4: Ejecutar en segundo plano (opcional)

Si quieres que los servicios corran en background:

```bash
docker compose up --build -d
```

Para ver los logs:

```bash
docker compose logs -f api       # Solo logs de la API
docker compose logs -f qdrant    # Solo logs de Qdrant
docker compose logs -f           # Todos los logs
```

### Detener los servicios

```bash
docker compose down
```

Para eliminar también los volúmenes (modelo cacheado y datos de Qdrant):

```bash
docker compose down -v
```

---

## Endpoints Disponibles

| Método | Ruta                  | Descripción                      |
|--------|-----------------------|----------------------------------|
| `GET`  | `/health`             | Health check                     |
| `POST` | `/documents/ingest`   | Subir y procesar un documento    |
| `POST` | `/search/`            | Buscar por texto (semántico)     |
| `GET`  | `/docs`               | Swagger UI (documentación auto)  |
| `GET`  | `/redoc`              | ReDoc (documentación alternativa)|

---

##  Ejemplos de Uso

### Ingestar un documento PDF

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@mi_documento.pdf"
```

**Respuesta:**

```json
{
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "chunks_stored": 15,
  "message": "Documento ingresado correctamente con 15 chunks"
}
```

### Ingestar una imagen (OCR)

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@foto_contrato.png"
```

### Ingestar un archivo de texto

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@notas.txt"
```

### Buscar en los documentos indexados

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "¿qué dice el contrato sobre penalizaciones?", "top_k": 3}'
```

**Respuesta:**

```json
{
  "query": "¿qué dice el contrato sobre penalizaciones?",
  "results": [
    {
      "document_id": "a1b2c3d4-...",
      "filename": "contrato.pdf",
      "chunk_index": 7,
      "text": "En caso de incumplimiento, se aplicarán penalizaciones del 5%...",
      "score": 0.87
    }
  ],
  "total": 1
}
```

### Buscar filtrando por nombre de archivo

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "cláusula de confidencialidad", "top_k": 5, "filter_filename": "contrato.pdf"}'
```

---

##  Swagger UI

Una vez levantado el servicio, accede a la documentación interactiva en:

**[http://localhost:8000/docs](http://localhost:8000/docs)**

Desde ahí puedes probar todos los endpoints directamente desde el navegador.

---

## Variables de Entorno

| Variable             | Valor por defecto                                               | Descripción                          |
|----------------------|-----------------------------------------------------------------|--------------------------------------|
| `QDRANT_HOST`        | `qdrant`                                                        | Host de Qdrant (nombre del servicio) |
| `QDRANT_PORT`        | `6333`                                                          | Puerto REST de Qdrant                |
| `QDRANT_COLLECTION`  | `documents`                                                     | Nombre de la colección de vectores   |
| `EMBEDDING_MODEL`    | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`   | Modelo de embeddings                 |
| `VECTOR_SIZE`        | `384`                                                           | Dimensión de los vectores            |
| `API_HOST`           | `0.0.0.0`                                                       | Host en el que escucha la API        |
| `API_PORT`           | `8000`                                                          | Puerto de la API                     |
| `LOG_LEVEL`          | `info`                                                          | Nivel de logging                     |

---

## Notas Importantes

- **Sin GPU requerida**: el modelo usa `torch+cpu` para mantener la imagen ligera (~1.5 GB). Si tienes GPU, puedes cambiar a `torch==2.3.1` en `requirements.txt`.
- **El modelo se descarga una sola vez** durante el build de Docker y se cachea en el volumen `hf_cache`.
- **Idiomas OCR**: se incluyen español (`spa`) e inglés (`eng`). Para agregar más idiomas, edita el `Dockerfile` y agrega `tesseract-ocr-{lang}`.
- **Chunk size**: por defecto son 500 caracteres con 50 de solapamiento. Para documentos largos se puede ajustar en `api/services/extractor.py`.
- **Si cambias de modelo de embeddings**, actualiza `VECTOR_SIZE` en `.env` y recrea los volúmenes con `docker compose down -v`.

---

## Script de Inicialización Manual

Si necesitas crear o recrear la colección de Qdrant manualmente (fuera de la API):

```bash
# Asegúrate de que Qdrant esté corriendo
pip install qdrant-client
python scripts/create_collection.py
```

---

## Troubleshooting

| Problema                                | Solución                                                                    |
|-----------------------------------------|-----------------------------------------------------------------------------|
| `Connection refused` al conectar con Qdrant | Verifica que Qdrant esté corriendo: `docker compose ps`                 |
| La API tarda mucho en arrancar          | Normal en la primera ejecución, el modelo se está cargando en memoria       |
| Error de OCR                           | Verifica que tesseract esté instalado en el contenedor (ya incluido)         |
| `No se pudo extraer texto`             | El archivo puede estar corrupto o tener un formato no soportado              |
| Memoria insuficiente                   | Asigna al menos 4 GB de RAM a Docker Desktop                                |
