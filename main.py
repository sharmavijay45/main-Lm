# main.py
import argparse
import os
from embeddings import create_embeddings

def parse_args():
    ap = argparse.ArgumentParser(description="Create embeddings from PDFs and upload to Qdrant")
    ap.add_argument("--pdf-dir", type=str, default=os.getenv("PDF_DIR", r"C:\Users\abc1\Documents\Datasets"))
    ap.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-api-key", type=str, default=os.getenv("QDRANT_API_KEY", None))
    ap.add_argument("--collection", type=str, default=os.getenv("QDRANT_COLLECTION", "documents"))
    ap.add_argument("--model", type=str, default=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Configuration:")
    print(f"  PDF_DIR = {args.pdf_dir}")
    print(f"  QDRANT_URL = {args.qdrant_url}")
    print(f"  COLLECTION = {args.collection}")
    print(f"  MODEL = {args.model}")
    create_embeddings(
        pdf_dir=args.pdf_dir,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embed_model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
