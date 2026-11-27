"""FastAPI backend.  Run:  uvicorn app.api:app --reload"""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import AuthenticationError, RateLimitError
from pydantic import BaseModel, Field

from app.agent import RagAgent
from app.config import DOCS_DIR, LLM_MODEL, TOP_K
from app.ingest import SUPPORTED, build_index
from app.retriever import HybridRetriever, IndexNotBuilt

app = FastAPI(title="Agentic RAG", version="1.0")
_agent: RagAgent | None = None


def get_agent() -> RagAgent:
    global _agent
    if _agent is None:
        try:
            _agent = RagAgent(HybridRetriever())
        except IndexNotBuilt as error:
            raise HTTPException(status_code=409, detail=str(error))
    return _agent


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


@app.get("/health")
def health():
    return {"status": "ok", "llm_model": LLM_MODEL, "docs_dir": str(DOCS_DIR)}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Save documents into the docs folder. Call /ingest afterwards to index them."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(status_code=400, detail=f"{name or 'file'}: only .txt, .md and .pdf are supported")
        (DOCS_DIR / name).write_bytes(await file.read())
        saved.append(name)
    return {"saved": saved}


@app.post("/ingest")
def ingest():
    """Rebuild the vector and keyword indexes from the docs folder."""
    global _agent
    try:
        summary = build_index()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    _agent = None  # reload with the new index on the next question
    return summary


@app.post("/ask")
def ask(request: AskRequest):
    agent = get_agent()
    try:
        return agent.ask(request.question, top_k=request.top_k, mode=request.mode)
    except RateLimitError as error:
        raise HTTPException(
            status_code=502,
            detail="The LLM provider reports that the account has no remaining API credits. Add credits or configure another provider.",
        ) from error
    except AuthenticationError as error:
        raise HTTPException(
            status_code=502,
            detail="The LLM provider rejected the API key. Set a valid OPENAI_API_KEY in .env.",
        ) from error
    except RuntimeError as error:  # e.g. missing API key
        raise HTTPException(status_code=500, detail=str(error))
