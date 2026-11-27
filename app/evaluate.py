"""Score the agent with RAGAS on the questions in data/eval_set.json.

Examples:
    python -m app.evaluate                 # hybrid retrieval
    python -m app.evaluate --mode all      # compare dense, bm25 and hybrid
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from app import embeddings
from app.agent import RagAgent
from app.config import BASE_DIR, LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from app.retriever import HybridRetriever

EVAL_FILE = BASE_DIR / "data" / "eval_set.json"
RESULTS_DIR = BASE_DIR / "results"


class LocalEmbeddings(Embeddings):
    """Lets RAGAS reuse the same sentence-transformers model as the retriever."""

    def embed_documents(self, texts):
        return embeddings.embed(texts)

    def embed_query(self, text):
        return embeddings.embed([text])[0]


def run_mode(agent: RagAgent, questions: list[dict], mode: str, judge_llm, judge_embeddings) -> pd.DataFrame:
    rows, self_scores = [], []
    for item in questions:
        result = agent.ask(item["question"], mode=mode)
        rows.append(
            {
                "user_input": item["question"],
                "retrieved_contexts": [s["text"] for s in result["sources"]] or [""],
                "response": result["answer"],
                "reference": item["ground_truth"],
            }
        )
        self_scores.append(result["self_eval"]["score"] if result["self_eval"] else None)

    scores = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    ).to_pandas()
    scores["agent_self_check"] = self_scores
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25", "all"])
    args = parser.parse_args()

    questions = json.loads(Path(EVAL_FILE).read_text(encoding="utf-8"))
    agent = RagAgent(HybridRetriever())
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY or "not-needed", base_url=OPENAI_BASE_URL, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())

    modes = ["dense", "bm25", "hybrid"] if args.mode == "all" else [args.mode]
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {}
    for mode in modes:
        print(f"\nEvaluating mode: {mode}")
        table = run_mode(agent, questions, mode, judge_llm, judge_embeddings)
        table.to_csv(RESULTS_DIR / f"eval_{mode}.csv", index=False)
        summary[mode] = table.select_dtypes("number").mean().round(3)

    print("\nAverage scores (0 to 1, higher is better):")
    print(pd.DataFrame(summary).to_string())
    print(f"\nPer-question results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
