"""
Script de inicialización de Qdrant.
Crea la colección 'documents' si no existe.

Uso:
    python scripts/create_collection.py

Nota: Este script se ejecuta fuera de Docker, por lo que usa localhost.
      La API crea la colección automáticamente al arrancar, pero este
      script es útil para inicialización manual o reseteo.
"""

import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))


def main():
    print(f"Conectando a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        print(f"La colección '{COLLECTION_NAME}' ya existe.")
        response = input("¿Desea recrearla? (s/N): ").strip().lower()
        if response == "s":
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Colección '{COLLECTION_NAME}' eliminada.")
        else:
            print("Operación cancelada.")
            sys.exit(0)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"Colección '{COLLECTION_NAME}' creada exitosamente.")
    print(f"  - Dimensión de vectores: {VECTOR_SIZE}")
    print(f"  - Distancia: COSINE")


if __name__ == "__main__":
    main()
