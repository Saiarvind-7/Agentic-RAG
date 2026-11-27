"""Small helpers that don't need any model: tokenizing and rank fusion."""
import re


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Used for BM25 on both the chunks and the query."""
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(rankings, k=60, weights=None):
    """Merge several ranked lists of ids into one.

    Each list is ordered best first. An id gets weight / (k + rank) from every list it
    appears in, so something ranked well by both retrievers beats something ranked
    first by only one. Returns [(id, score), ...] sorted best first.
    """
    weights = weights or [1.0] * len(rankings)
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
