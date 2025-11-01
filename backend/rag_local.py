# backend/rag_local.py
import os, glob, re, uuid
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from bs4 import BeautifulSoup
import html2text

CHROMA_PATH = "artifacts/chroma_f1"
COLLECTION = "f1_docs"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Embeddings ---
_sbert = SentenceTransformer(EMB_MODEL)
def embed(texts):  # returns list[list[float]]
    return _sbert.encode(texts, show_progress_bar=False).tolist()

# --- DB / Collection ---
def _collection():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        col = client.create_collection(COLLECTION)
    return col

# --- Simple loaders ---
def _load_pdf(path):
    reader = PdfReader(path)
    out = []
    for i, pg in enumerate(reader.pages):
        try:
            txt = pg.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            out.append({"text": txt.strip(), "source": f"{os.path.basename(path)}#p{i+1}"})
    return out

def _load_html(path):
    raw = open(path,"rb").read()
    soup = BeautifulSoup(raw, "html.parser")
    for s in soup(["script","style","noscript"]): s.extract()
    md = html2text.HTML2Text(); md.ignore_links=False
    text = md.handle(str(soup))
    return [{"text": text.strip(), "source": os.path.basename(path)}] if text.strip() else []

def _load_txt(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    return [{"text": text.strip(), "source": os.path.basename(path)}] if text.strip() else []

def _chunk(text, max_tokens=600, overlap=80):
    # crude splitter by sentences; good enough for MVP
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > max_tokens and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur += (" " + p if cur else p)
    if cur.strip():
        chunks.append(cur.strip())
    # add overlap
    out = []
    for i, c in enumerate(chunks):
        if i == 0: out.append(c); continue
        prev = chunks[i-1]
        tail = prev[-overlap:] if len(prev) > overlap else prev
        out.append((tail + " " + c).strip())
    return out

def ingest_dir(dirpath="docs"):
    col = _collection()
    files = sum([glob.glob(os.path.join(dirpath, ext)) for ext in ("*.pdf","*.txt","*.html","*.htm")], [])
    print(f"[ingest] dir={os.path.abspath(dirpath)} files={len(files)} -> {files}")
    to_add = []
    for f in files:
        print(f"[ingest] reading {os.path.basename(f)}")
        if f.lower().endswith(".pdf"):
            docs = _load_pdf(f)
        elif f.lower().endswith((".html",".htm")):
            docs = _load_html(f)
        else:
            docs = _load_txt(f)
        print(f"[ingest] extracted docs: {len(docs)}")
        for d in docs:
            for ch in _chunk(d["text"]):
                to_add.append((str(uuid.uuid4()), ch, d["source"]))
    print(f"[ingest] total chunks: {len(to_add)}")
    if to_add:
        print("[ingest] embedding… (first run downloads the model)")
        ids, texts, metas = zip(*[(i,t,{"source":s}) for i,t,s in to_add])
        embs = embed(list(texts))
        print("[ingest] writing to chroma…")
        col.add(ids=list(ids), documents=list(texts), metadatas=list(metas), embeddings=embs)
    print("[ingest] done")
    return len(to_add)


def retrieve(query, k=6):
    col = _collection()
    embs = embed([query])[0]
    res = col.query(query_embeddings=[embs], n_results=k, include=["documents","metadatas","distances"])
    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "score": float(res["distances"][0][i])
        })
    return docs
