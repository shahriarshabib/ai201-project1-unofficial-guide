"""Stretch feature — Hybrid search (BM25 keyword + semantic dense retrieval).

Semantic search (all-MiniLM-L6-v2) matches *meaning* but can miss a rare,
decisive keyword — that is exactly the project's documented failure case:
"which intro professor do reviews warn beginners about?" never retrieves
Hani Karam's CIS1057 review, even though it literally says
"Not recommended for beginners", because the dense vector is dominated by
sentiment rather than the word "beginners".

BM25 is the opposite: it scores exact term overlap (and weights rare terms
like "beginners" heavily) but has no notion of synonyms. Combining the two
gets the best of both.

Fusion method
-------------
For a query we score all N chunks with each retriever, min-max normalise each
score vector to [0, 1] across the candidate set, then take a weighted sum:

    combined = alpha * semantic_norm + (1 - alpha) * bm25_norm        (alpha=0.5)

semantic_norm comes from cosine similarity (1 - cosine_distance); bm25_norm
from BM25Okapi. alpha=1.0 is pure semantic, alpha=0.0 is pure BM25.

Run:
    python hybrid.py        # compares semantic-only / BM25-only / hybrid
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

from index import get_model

CHUNKS_PATH = Path("data/chunks.json")
DEFAULT_ALPHA = 0.5


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _minmax(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridIndex:
    """In-memory hybrid index over the chunks (small corpus, ~60 chunks)."""

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.embed_texts = [c.get("embed_text", c["text"]) for c in self.chunks]
        # Dense side: precompute normalised embeddings once.
        self.model = get_model()
        self.embeddings = self.model.encode(
            self.embed_texts, normalize_embeddings=True, show_progress_bar=False
        )
        # Sparse side: BM25 over the same lean text.
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.embed_texts])

    def _semantic_sims(self, query: str) -> list[float]:
        q = self.model.encode([query], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        # embeddings are normalised, so dot product == cosine similarity.
        return [float(sum(a * b for a, b in zip(q, e))) for e in self.embeddings]

    def search(self, query: str, k: int = 5, alpha: float = DEFAULT_ALPHA) -> list[dict]:
        sem = self._semantic_sims(query)            # raw cosine similarity [-1, 1]
        bm = list(self.bm25.get_scores(_tokenize(query)))
        sem_n, bm_n = _minmax(sem), _minmax(bm)
        scored = []
        for i, c in enumerate(self.chunks):
            combined = alpha * sem_n[i] + (1 - alpha) * bm_n[i]
            scored.append({
                "text": c["text"],
                "metadata": c["metadata"],
                "semantic": sem_n[i],
                "bm25": bm_n[i],
                "combined": combined,
                "distance": 1.0 - sem[i],          # raw cosine distance (for the grounding gate)
            })
        scored.sort(key=lambda r: r["combined"], reverse=True)
        return scored[:k]


_HYBRID: HybridIndex | None = None


def hybrid_search(query: str, k: int = 5, alpha: float = DEFAULT_ALPHA) -> list[dict]:
    """Module-level convenience wrapper with a cached index."""
    global _HYBRID
    if _HYBRID is None:
        _HYBRID = HybridIndex()
    return _HYBRID.search(query, k=k, alpha=alpha)


def _top(index: HybridIndex, query: str, alpha: float, k: int = 3) -> list[dict]:
    return index.search(query, k=k, alpha=alpha)


def _label(r: dict) -> str:
    m = r["metadata"]
    return f"{m['professor']}/{m['course']}"


def compare(index: HybridIndex, query: str) -> None:
    print(f"\nQUERY: {query}")
    print(f"{'method':<16}{'top result':<34}{'sem':>6}{'bm25':>7}{'comb':>7}")
    print("-" * 70)
    for name, alpha in [("semantic-only", 1.0), ("BM25-only", 0.0), ("hybrid α=.5", 0.5)]:
        top = _top(index, query, alpha)[0]
        print(f"{name:<16}{_label(top):<34}"
              f"{top['semantic']:>6.2f}{top['bm25']:>7.2f}{top['combined']:>7.2f}")
    # Show the hybrid top-3 for context.
    print("hybrid top-3:")
    for r in _top(index, query, 0.5):
        print(f"   {_label(r):<34} sem={r['semantic']:.2f} bm25={r['bm25']:.2f} comb={r['combined']:.2f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    index = HybridIndex()
    for q in [
        "Which intro-level CS professor do reviews warn beginners about?",  # the failure case
        "How heavy is the workload in Christopher Pascucci's courses?",
        "What do students say about Data Structures with David Dobor?",
    ]:
        compare(index, q)


if __name__ == "__main__":
    main()
