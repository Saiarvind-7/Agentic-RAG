"""The agent: plan -> retrieve -> grade -> (retry) -> answer -> self-check -> (rewrite).

Every step is a small LLM call or a retrieval call, and each one is recorded in a
trace so you can see what the agent did for a question.
"""
from app import llm
from app.config import FAITHFULNESS_THRESHOLD, MAX_ATTEMPTS, MAX_REGENERATIONS, RRF_K, TOP_K
from app.fusion import reciprocal_rank_fusion

PLAN_PROMPT = """You help answer questions using a private collection of documents.
Decide whether the question needs the documents, and write up to 3 short search queries
that would find the answer. Use needs_retrieval=false only for greetings or small talk.
Reply with JSON only: {"needs_retrieval": true, "queries": ["..."]}"""

GRADE_PROMPT = """You check whether retrieved text is enough to answer a question.
Reply with JSON only: {"sufficient": true or false, "missing": "what is still missing, or an empty string"}"""

REWRITE_PROMPT = """The first search did not find everything needed to answer the question.
Write up to 2 new short search queries aimed at the missing information. Use different
words from the earlier queries.
Reply with JSON only: {"queries": ["..."]}"""

ANSWER_PROMPT = """Answer the question using ONLY the numbered context below.
Cite the sources you use like [1] or [2]. If the context does not contain the answer,
say that you could not find it in the documents. Do not guess and do not add outside facts.
Keep the answer short and direct."""

CHECK_PROMPT = """You are a strict fact checker. You get a context and an answer.
Split the answer into its factual claims and decide which of them the context supports.
Reply with JSON only:
{"score": <supported claims divided by total claims, from 0 to 1>, "unsupported_claims": ["..."]}
If the answer only says the information was not found, the score is 1."""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, start=1))


def clean_queries(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [q.strip() for q in value if isinstance(q, str) and q.strip()][:limit]


class RagAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    # --- helpers ---------------------------------------------------------
    def _retrieve(self, queries: list[str], top_k: int, mode: str) -> list[dict]:
        """Search once per query and merge the lists with rank fusion."""
        per_query = [self.retriever.search(q, top_k=top_k, mode=mode) for q in queries]
        if len(per_query) == 1:
            return per_query[0]
        by_id = {c["id"]: c for results in per_query for c in results}
        fused = reciprocal_rank_fusion([[c["id"] for c in r] for r in per_query], k=RRF_K)
        return [by_id[chunk_id] for chunk_id, _ in fused[:top_k]]

    @staticmethod
    def _merge(old: list[dict], new: list[dict], limit: int) -> list[dict]:
        seen, merged = set(), []
        for chunk in old + new:
            if chunk["id"] not in seen:
                seen.add(chunk["id"])
                merged.append(chunk)
        return merged[:limit]

    # --- main entry point ------------------------------------------------
    def ask(self, question: str, top_k: int | None = None, mode: str = "hybrid") -> dict:
        top_k = top_k or TOP_K
        trace = []

        plan = llm.chat_json(PLAN_PROMPT, question)
        queries = clean_queries(plan.get("queries"), 3) or [question]
        trace.append({"step": "plan", "needs_retrieval": plan.get("needs_retrieval", True), "queries": queries})

        if plan.get("needs_retrieval", True) is False:
            answer = llm.chat([{"role": "user", "content": question}])
            return {"answer": answer, "sources": [], "self_eval": None, "attempts": 0, "trace": trace}

        # 1) retrieve and grade, retry with new queries if the context is thin
        contexts: list[dict] = []
        attempts = 0
        while True:
            attempts += 1
            found = self._retrieve(queries, top_k, mode)
            contexts = self._merge(contexts, found, limit=top_k + 3)
            verdict = llm.chat_json(GRADE_PROMPT, f"Question: {question}\n\nContext:\n{format_context(contexts)}")
            sufficient = verdict.get("sufficient", True) is not False
            trace.append(
                {
                    "step": "retrieve",
                    "attempt": attempts,
                    "queries": queries,
                    "chunks": [f"{c['source']}#{c['chunk_index']}" for c in contexts],
                    "sufficient": sufficient,
                    "missing": verdict.get("missing", ""),
                }
            )
            if sufficient or attempts >= MAX_ATTEMPTS:
                break
            rewrite = llm.chat_json(
                REWRITE_PROMPT,
                f"Question: {question}\nMissing: {verdict.get('missing', '')}\nEarlier queries: {queries}",
            )
            new_queries = clean_queries(rewrite.get("queries"), 2)
            if not new_queries:
                break
            queries = new_queries

        # 2) answer, then check the answer against the context and rewrite if needed
        context_text = format_context(contexts)
        feedback = ""
        answer, self_eval = "", None
        for round_number in range(1 + MAX_REGENERATIONS):
            user = f"Context:\n{context_text}\n\nQuestion: {question}"
            if feedback:
                user += f"\n\nYour previous answer contained claims the context does not support: {feedback}\nLeave those out."
            answer = llm.chat([{"role": "system", "content": ANSWER_PROMPT}, {"role": "user", "content": user}])

            check = llm.chat_json(CHECK_PROMPT, f"Context:\n{context_text}\n\nAnswer:\n{answer}")
            try:
                score = max(0.0, min(1.0, float(check.get("score", 1.0))))
            except (TypeError, ValueError):
                score = 1.0
            unsupported = [c for c in check.get("unsupported_claims", []) if isinstance(c, str)]
            self_eval = {"score": score, "unsupported_claims": unsupported, "passed": score >= FAITHFULNESS_THRESHOLD}
            trace.append({"step": "self_check", "round": round_number + 1, **self_eval})
            if self_eval["passed"]:
                break
            feedback = "; ".join(unsupported) or "some statements were not backed by the context"

        sources = [
            {"n": i, "source": c["source"], "chunk_id": c["id"], "text": c["text"], "found_by": c["found_by"]}
            for i, c in enumerate(contexts, start=1)
        ]
        return {"answer": answer, "sources": sources, "self_eval": self_eval, "attempts": attempts, "trace": trace}
