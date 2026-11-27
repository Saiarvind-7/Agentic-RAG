"""Tests run without an API key or a model download: embeddings and the LLM are faked."""
import os
import tempfile
import zlib

# must be set before anything from `app` is imported
os.environ["INDEX_DIR"] = tempfile.mkdtemp(prefix="rag_test_")

import pytest  # noqa: E402

from app import embeddings, llm  # noqa: E402
from app.fusion import tokenize  # noqa: E402


def fake_embed(texts):
    """Bag-of-words hashing into 256 dimensions. Crude, but deterministic and offline."""
    vectors = []
    for text in texts:
        vec = [0.0] * 256
        for token in tokenize(text):
            vec[zlib.crc32(token.encode()) % 256] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


def fake_chat_json(system, user):
    if "needs_retrieval" in system:
        return {"needs_retrieval": True, "queries": [user]}
    if "sufficient" in system:
        return {"sufficient": True, "missing": ""}
    if "fact checker" in system:
        return {"score": 1.0, "unsupported_claims": []}
    return {}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(llm, "chat", lambda messages, json_mode=False: "Employees get 24 days of leave [1].")
