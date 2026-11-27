"""Settings. Everything can be overridden with environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# where documents are read from and where the index is stored
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "storage"))
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"

# models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

# retrieval
TOP_K = int(os.getenv("TOP_K", 5))              # chunks given to the LLM
CANDIDATES = int(os.getenv("CANDIDATES", 20))   # chunks each retriever returns before fusion
RRF_K = int(os.getenv("RRF_K", 60))             # reciprocal rank fusion constant

# agent behaviour
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 2))                  # retrieval rounds
MAX_REGENERATIONS = int(os.getenv("MAX_REGENERATIONS", 1))        # rewrites after a failed self-check
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", 0.7))
