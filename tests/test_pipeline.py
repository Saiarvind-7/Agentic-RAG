import httpx
from openai import AuthenticationError, RateLimitError
from fastapi.testclient import TestClient

from app.agent import RagAgent
from app import api
from app.api import app
from app.ingest import build_index
from app.retriever import HybridRetriever


def test_index_and_hybrid_search():
    summary = build_index()
    assert summary["documents"] >= 3 and summary["chunks"] >= 3

    retriever = HybridRetriever()
    hits = retriever.search("How many characters must a password have, POL-305?", top_k=3)
    assert hits[0]["source"] == "it_security_policy.md"
    assert hits[0]["found_by"]  # at least one retriever found it


def test_keyword_search_finds_exact_policy_code():
    build_index()
    retriever = HybridRetriever()
    top = retriever.keyword_search("POL-204")
    assert retriever.chunks[top[0]]["source"] == "remote_work_policy.md"


def test_agent_returns_answer_sources_and_trace():
    build_index()
    result = RagAgent(HybridRetriever()).ask("How many days of annual leave do I get?")
    assert "24 days" in result["answer"]
    assert result["sources"] and result["self_eval"]["passed"]
    assert [step["step"] for step in result["trace"]] == ["plan", "retrieve", "self_check"]


def test_api_flow():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/ingest").json()["chunks"] >= 3
    reply = client.post("/ask", json={"question": "How many days of annual leave do I get?"})
    assert reply.status_code == 200 and reply.json()["sources"]
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_reports_invalid_llm_credentials(monkeypatch):
    class FailingAgent:
        def ask(self, question, top_k, mode):
            raise AuthenticationError(
                "Invalid API key",
                response=httpx.Response(401, request=httpx.Request("POST", "https://api.example.com")),
                body=None,
            )

    monkeypatch.setattr(api, "get_agent", lambda: FailingAgent())
    response = TestClient(app).post("/ask", json={"question": "test"})

    assert response.status_code == 502
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_api_reports_exhausted_llm_credits(monkeypatch):
    class FailingAgent:
        def ask(self, question, top_k, mode):
            raise RateLimitError(
                "No credits remaining",
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.example.com")),
                body=None,
            )

    monkeypatch.setattr(api, "get_agent", lambda: FailingAgent())
    response = TestClient(app).post("/ask", json={"question": "test"})

    assert response.status_code == 502
    assert "no remaining API credits" in response.json()["detail"]
