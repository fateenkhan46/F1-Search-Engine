# backend/ingest_docs.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # force PyTorch path (no TF/Keras)

from backend.rag_local import ingest_dir

if __name__ == "__main__":
    n = ingest_dir("docs")
    print(f"chunks: {n}")
