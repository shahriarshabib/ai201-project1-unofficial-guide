"""Milestone 4 — Embedding and retrieval.

Embeds the chunks produced by ingest.py with all-MiniLM-L6-v2 and stores them
in a persistent ChromaDB collection, then exposes retrieve() for similarity
search. Run directly to (re)build the index and test it against the
evaluation queries:

    python index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.json")
DB_PATH = "chroma_db"
COLLECTION = "temple_cs_reviews"
MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded lazily so importing retrieve() in Milestone 5 doesn't pay the cost
# until it's actually used.
_model: SentenceTransformer | None = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    """Return the existing Chroma collection (cosine space)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _collection


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_index(chunks_path: Path = CHUNKS_PATH):
    """Embed every chunk and (re)load it into a fresh ChromaDB collection."""
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    # Rebuild from scratch so re-running is idempotent.
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks]                       # stored for display/LLM
    embed_texts = [c.get("embed_text", c["text"]) for c in chunks]  # vectorised
    metadatas = [c["metadata"] for c in chunks]
    # Stable id = source file + position within that file's chunk list.
    ids, per_file_pos = [], {}
    for m in metadatas:
        f = m["source_file"]
        pos = per_file_pos.get(f, 0)
        per_file_pos[f] = pos + 1
        ids.append(f"{f}#{pos}")
        m["position"] = pos  # store chunk position in document (required metadata)

    print(f"Embedding {len(texts)} chunks with {MODEL_NAME} ...")
    embeddings = get_model().encode(
        embed_texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Stored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION}' at ./{DB_PATH}")
    global _collection
    _collection = collection
    return collection


# --------------------------------------------------------------------------- #
# Retrieve
# --------------------------------------------------------------------------- #
def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return the top-k most relevant chunks for a query.

    Each result is {text, metadata, distance}; distance is cosine distance
    (0 = identical direction, lower is more relevant).
    """
    collection = get_collection()
    q_emb = get_model().encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    out = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({"text": text, "metadata": meta, "distance": dist})
    return out


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #
TEST_QUERIES = [
    "Which Temple CS professor do students most recommend, and why?",
    "How heavy is the workload in Christopher Pascucci's web development courses?",
    "What do students say about taking Data Structures with David Dobor?",
    "Which intro CS professor should a beginner avoid?",
    "Which dorm is closest to the computer science building?",  # out-of-corpus probe
]


def _test_retrieval(k: int = 5) -> None:
    print("\n" + "=" * 74)
    print(f"RETRIEVAL TEST  (top-k={k}, cosine distance — lower is better)")
    print("=" * 74)
    for q in TEST_QUERIES:
        print(f"\nQ: {q}")
        for i, r in enumerate(retrieve(q, k=k), 1):
            m = r["metadata"]
            print(f"  {i}. dist={r['distance']:.3f}  [{m['professor']} | "
                  f"{m['course']} | {m['chunk_type']}]")
            print(f"     {r['text'][:140]}{'…' if len(r['text']) > 140 else ''}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    build_index()
    _test_retrieval(k=5)


if __name__ == "__main__":
    main()
