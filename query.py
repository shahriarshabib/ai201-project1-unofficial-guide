"""Milestone 5 — Grounded generation.

Wires retrieval (index.retrieve) to Groq's llama-3.3-70b-versatile and returns
an answer grounded strictly in the retrieved student reviews, plus a
programmatically-built source list.

    python query.py            # interactive CLI
    python query.py "your question here"

Grounding is enforced in three layers:
  1. A relevance gate: if the best retrieved chunk is farther than
     DISTANCE_THRESHOLD, we decline *before* calling the LLM (the
     out-of-corpus "dorm" probe sits at ~0.67, real answers at ~0.3-0.5).
  2. A strict system prompt: answer ONLY from the provided reviews; if they
     don't cover it, say so; never use outside knowledge.
  3. Sources are appended programmatically from the chunks actually used —
     not left to the model to invent.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from index import retrieve

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
DISTANCE_THRESHOLD = 0.60   # cosine distance above which a chunk is "not relevant"
NO_INFO = "I don't have enough information on that in the student reviews I have."

SYSTEM_PROMPT = (
    "You are The Unofficial Guide to Temple University CS professors. You "
    "answer ONLY using the student reviews provided in the CONTEXT below.\n"
    "Rules:\n"
    "1. Use only facts found in the CONTEXT. Do NOT use any outside or prior "
    "knowledge about these professors, courses, or Temple University.\n"
    "2. If the CONTEXT does not contain enough information to answer, reply "
    f"exactly: \"{NO_INFO}\"\n"
    "3. Attribute claims to the professor they are about (e.g. 'Students say "
    "David Dobor...'). When reviews disagree, say so rather than picking a "
    "side.\n"
    "4. Be concise (2-5 sentences). Do not invent ratings, courses, quotes, or "
    "professor names that are not in the CONTEXT."
)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "free key from https://console.groq.com"
            )
        _client = Groq(api_key=key)
    return _client


def _format_context(chunks: list[dict]) -> str:
    """Build the numbered CONTEXT block the model reads."""
    lines = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        lines.append(f"[{i}] (source: {m['source_file']}) {c['text']}")
    return "\n\n".join(lines)


def _sources_from(chunks: list[dict], answer: str) -> list[str]:
    """De-duplicated source list, restricted to professors the answer names.

    Retrieval may pull chunks from several professors who share a course code
    (e.g. CIS2033). To keep attribution honest, we only cite a source if the
    generated answer actually mentions that professor (full name or surname).
    Falls back to all relevant chunks if no name matches.
    """
    def named(prof: str) -> bool:
        return prof.lower() in answer.lower() or prof.split()[-1].lower() in answer.lower()

    seen, sources = set(), []
    for c in chunks:
        m = c["metadata"]
        if m["source_file"] in seen or not named(m["professor"]):
            continue
        seen.add(m["source_file"])
        url = f" — {m['source_url']}" if m.get("source_url") else ""
        sources.append(f"{m['source_file']} ({m['professor']}){url}")

    if sources:  # at least one professor was named
        return sources
    # Fallback: answer named no professor — list all relevant sources.
    for c in chunks:
        m = c["metadata"]
        if m["source_file"] in seen:
            continue
        seen.add(m["source_file"])
        url = f" — {m['source_url']}" if m.get("source_url") else ""
        sources.append(f"{m['source_file']} ({m['professor']}){url}")
    return sources


def ask(question: str, k: int = 5, threshold: float = DISTANCE_THRESHOLD) -> dict:
    """Return {answer, sources, used_chunks} for a question, grounded in reviews."""
    retrieved = retrieve(question, k=k)

    # Layer 1 — relevance gate. Keep only chunks close enough to be on-topic.
    relevant = [c for c in retrieved if c["distance"] <= threshold]
    if not relevant:
        best = min((c["distance"] for c in retrieved), default=1.0)
        return {
            "answer": NO_INFO,
            "sources": [],
            "used_chunks": [],
            "note": f"No chunk within threshold (best distance {best:.3f}).",
        }

    # Layer 2 — grounded generation.
    context = _format_context(relevant)
    user_msg = (
        f"CONTEXT (student reviews):\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above."
    )
    resp = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    if not answer:  # rare: model returns an empty completion -> treat as no answer
        answer = NO_INFO

    # Layer 3 — source attribution guaranteed programmatically.
    sources = [] if answer.strip() == NO_INFO else _sources_from(relevant, answer)
    return {"answer": answer, "sources": sources, "used_chunks": relevant}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_result(question: str, result: dict) -> None:
    print(f"\nQ: {question}")
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Retrieved from:")
        for s in result["sources"]:
            print(f"  • {s}")
    elif result.get("note"):
        print(f"({result['note']})")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) > 1:  # one-shot mode
        q = " ".join(sys.argv[1:])
        _print_result(q, ask(q))
        return

    print("The Unofficial Guide — Temple CS professors. "
          "Ask a question (blank line or Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        _print_result(q, ask(q))


if __name__ == "__main__":
    main()
