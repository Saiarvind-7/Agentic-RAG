"""Read documents, split them into chunks and build both search indexes.

Run it directly:  python -m app.ingest
"""
import hashlib
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import embeddings
from app.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE, DOCS_DIR, INDEX_DIR

COLLECTION = "docs"
SUPPORTED = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    docs_dir = Path(docs_dir)
    docs = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = read_file(path).strip()
            if text:
                docs.append({"source": str(path.relative_to(docs_dir)), "text": text})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in docs:
        for index, piece in enumerate(splitter.split_text(doc["text"])):
            key = f'{doc["source"]}:{index}:{piece}'
            chunks.append(
                {
                    "id": hashlib.sha1(key.encode("utf-8")).hexdigest()[:16],
                    "source": doc["source"],
                    "chunk_index": index,
                    "text": piece,
                }
            )
    return chunks


def build_index(docs_dir: Path = DOCS_DIR) -> dict:
    """Rebuild everything from scratch. Returns a small summary."""
    import chromadb

    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(f"No .txt, .md or .pdf files with text found in {docs_dir}")
    chunks = chunk_documents(docs)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run, nothing to delete
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), 64):
        batch = chunks[start : start + 64]
        texts = [c["text"] for c in batch]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings.embed(texts),
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch],
        )

    # the keyword (BM25) index is rebuilt from this file when the app starts
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    return {"documents": len(docs), "chunks": len(chunks)}


if __name__ == "__main__":
    print(build_index())
