"""Checks the two loops: retry retrieval when context is thin, rewrite when the answer is unsupported."""
from app import llm
from app.agent import RagAgent
from app.ingest import build_index
from app.retriever import HybridRetriever


def test_retries_retrieval_when_context_is_insufficient(monkeypatch):
    build_index()
    grades = iter([{"sufficient": False, "missing": "carry over rules"}, {"sufficient": True, "missing": ""}])

    def scripted(system, user):
        if "needs_retrieval" in system:
            return {"needs_retrieval": True, "queries": ["annual leave"]}
        if "sufficient" in system:
            return next(grades)
        if "new short search queries" in system:
            return {"queries": ["carry over unused leave"]}
        return {"score": 1.0, "unsupported_claims": []}

    monkeypatch.setattr(llm, "chat_json", scripted)
    result = RagAgent(HybridRetriever()).ask("Can I carry over leave?")
    steps = [t["step"] for t in result["trace"]]
    assert result["attempts"] == 2 and steps.count("retrieve") == 2


def test_rewrites_answer_when_self_check_fails(monkeypatch):
    build_index()
    scores = iter([0.2, 1.0])
    answers = iter(["Employees get 40 days [1].", "Employees get 24 days [1]."])

    def scripted(system, user):
        if "needs_retrieval" in system:
            return {"needs_retrieval": True, "queries": ["annual leave"]}
        if "sufficient" in system:
            return {"sufficient": True, "missing": ""}
        return {"score": next(scores), "unsupported_claims": ["40 days"]}

    monkeypatch.setattr(llm, "chat_json", scripted)
    monkeypatch.setattr(llm, "chat", lambda messages, json_mode=False: next(answers))
    result = RagAgent(HybridRetriever()).ask("How many leave days?")
    assert result["answer"] == "Employees get 24 days [1]."
    assert [t["passed"] for t in result["trace"] if t["step"] == "self_check"] == [False, True]
