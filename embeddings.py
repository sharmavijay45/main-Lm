# embeddings.py
import os
from pathlib import Path
import uuid
import math
from typing import List
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# CONFIG (can be overridden by env or caller)
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Load model once globally (lazy)
_model = None

def get_model(model_name: str = DEFAULT_EMBED_MODEL) -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model

def get_qdrant_client(url: str = DEFAULT_QDRANT_URL, api_key: str = DEFAULT_QDRANT_API_KEY) -> QdrantClient:
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)

def read_pdf(file_path: str) -> str:
    """Extract text from PDF file using PyMuPDF."""
    text_parts = []
    doc = fitz.open(file_path)
    for page in doc:
        text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 128) -> List[str]:
    """
    Chunk text into overlapping chunks.
    chunk_size and overlap are measured in characters (good heuristic).
    """
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create collection if it doesn't exist. Preserves existing data."""
    print(f"Checking collection '{collection_name}'")
    try:
        info = client.get_collection(collection_name=collection_name)
        # Get vector size from collection config
        config = client.get_collection(collection_name=collection_name).config
        current_size = config.params.vectors.size
        if current_size != vector_size:
            raise ValueError(f"Collection exists but vector size mismatch ({current_size} != {vector_size}). Please use a different collection name.")
        print(f"Collection '{collection_name}' exists with correct vector size {vector_size}, proceeding with updates")
        return
    except ValueError as e:
        # Re-raise vector size mismatch error
        raise e
    except Exception as e:
        # Only create if collection really doesn't exist
        if "Collection `documents` already exists!" in str(e):
            print(f"Collection '{collection_name}' already exists, proceeding with updates")
            return
        print(f"Creating new collection '{collection_name}' with vector size {vector_size}")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except Exception as e:
            if "Collection `documents` already exists!" in str(e):
                print(f"Collection '{collection_name}' already exists, proceeding with updates")
                return
            raise e


def create_embeddings(
    pdf_dir: str,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    qdrant_api_key: str = DEFAULT_QDRANT_API_KEY,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = 800,
    chunk_overlap: int = 128,
    batch_size: int = 128,
) -> None:
    """
    Walk pdf_dir (recursively), read PDFs, chunk them, embed each chunk and upsert to Qdrant in batches.
    """
    model = get_model(embed_model_name)
    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    vector_dim = model.get_sentence_embedding_dimension()
    ensure_collection(client, collection_name, vector_dim)

    # ✅ recursively collect PDFs
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))

    total_chunks = 0
    points_batch = []

    for pdf_path in pdf_files:
        filename = pdf_path.name
        rel_path = str(pdf_path.relative_to(pdf_dir))  # so you know folder structure
        print(f"Processing: {rel_path}")
        try:
            text = read_pdf(str(pdf_path))
        except Exception as e:
            print(f"  ⚠️ Failed to read {filename}: {e}")
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        print(f"  → {len(chunks)} chunks")
        total_chunks += len(chunks)

        for i in range(0, len(chunks), 32):
            sub = chunks[i:i+32]
            vectors = model.encode(sub, show_progress_bar=False)
            for chunk_text_, vector in zip(sub, vectors):
                pts = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                    payload={
                        "content": chunk_text_,
                        "file": filename,
                        "path": rel_path,   # ✅ store relative folder path also
                    },
                )
                points_batch.append(pts)

            if len(points_batch) >= batch_size:
                client.upsert(collection_name=collection_name, points=points_batch)
                print(f"  Upserted batch of {len(points_batch)} points")
                points_batch = []

    if points_batch:
        client.upsert(collection_name=collection_name, points=points_batch)
        print(f"Upserted final batch of {len(points_batch)} points")

    print(f"Done. Total PDFs processed: {len(pdf_files)}. Total chunks approx: {total_chunks}")





