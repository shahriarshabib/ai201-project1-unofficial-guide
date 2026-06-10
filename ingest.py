"""Milestone 3 — Document ingestion and chunking.

Loads the Rate My Professors source files from data/raw/, cleans them, and
splits them into chunks following the strategy in planning.md:

    one student review  -> one chunk   (semantic unit, 0 overlap)
    one file's aggregate -> one summary chunk

Each chunk is prefixed with the professor name + course code so it is
self-describing without overlap. Run directly to inspect the output:

    python ingest.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

RAW_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/chunks.json")

# A review block starts with a line like:
#   [CIS1051 | Oct 29, 2025 | Quality 5.0 | Difficulty 3.0 | Grade A+ | Online]
REVIEW_HEADER_RE = re.compile(r"^\[(?P<body>.+)\]\s*$")
COURSE_RE = re.compile(r"CIS\d{3,4}")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Strip HTML tags/entities and normalise whitespace.

    Our collected files are already plain text, so this is mostly defensive:
    it keeps the pipeline correct if a messier source (scraped HTML, a forum
    export) is added later. It must never delete substantive content.
    """
    text = html.unescape(text)            # &amp; -> &, &#39; -> '
    text = re.sub(r"<[^>]+>", "", text)   # drop any stray HTML tags
    text = text.replace(" ", " ")
    # Collapse runs of spaces/tabs/nbsp but preserve line breaks (structure matters).
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_header(lines: list[str]) -> tuple[dict, list[str]]:
    """Read the leading `Key: value` block (up to the first blank line)."""
    header: dict[str, str] = {}
    i = 0
    for i, line in enumerate(lines):
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().lower()] = value.strip()
    return header, lines[i + 1 :]


def _parse_stats(lines: list[str]) -> dict:
    """Read the aggregate-stats block (Overall rating, Would take again, ...)."""
    stats: dict[str, str] = {}
    for line in lines:
        if line.strip().lower().startswith("student reviews"):
            break
        if ":" in line:
            key, value = line.split(":", 1)
            stats[key.strip().lower()] = value.strip()
    return stats


def _parse_review_meta(body: str) -> dict:
    """Parse the contents of a `[...]` review-header line into fields."""
    parts = [p.strip() for p in body.split("|")]
    meta: dict[str, str] = {"raw": body}
    if parts:
        course_match = COURSE_RE.search(parts[0])
        meta["course"] = course_match.group(0) if course_match else parts[0]
    if len(parts) > 1:
        meta["date"] = parts[1]
    for part in parts[2:]:
        low = part.lower()
        if low.startswith("quality"):
            meta["quality"] = part.split()[-1]
        elif low.startswith("difficulty"):
            meta["difficulty"] = part.split()[-1]
        elif low.startswith("grade"):
            meta["grade"] = part.replace("Grade", "").strip()
        else:
            meta.setdefault("flags", []).append(part)  # Online, Mandatory attendance, ...
    return meta


def parse_file(path: Path) -> dict:
    """Parse one source file into {header, stats, reviews:[{meta, text}]}."""
    raw = path.read_text(encoding="utf-8")
    lines = clean_text(raw).splitlines()

    header, rest = _parse_header(lines)
    stats = _parse_stats(rest)

    reviews: list[dict] = []
    current_meta: dict | None = None
    current_text: list[str] = []

    def flush() -> None:
        if current_meta is not None:
            text = " ".join(current_text).strip()
            if text:  # skip empty review bodies
                reviews.append({"meta": current_meta, "text": text})

    for line in rest:
        m = REVIEW_HEADER_RE.match(line)
        if m:
            flush()
            current_meta = _parse_review_meta(m.group("body"))
            current_text = []
        elif current_meta is not None and line.strip():
            current_text.append(line.strip())
    flush()

    return {"file": path.name, "header": header, "stats": stats, "reviews": reviews}


def load_documents(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Load and parse every .txt file in the raw directory."""
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir.resolve()}")
    return [parse_file(p) for p in files]


# --------------------------------------------------------------------------- #
# Chunking  (planning.md: one review = one chunk, 0 overlap, prof+course prefix)
# --------------------------------------------------------------------------- #
def chunk_document(doc: dict) -> list[dict]:
    """Turn one parsed document into review chunks + one summary chunk."""
    professor = doc["header"].get("professor", "Unknown professor")
    source_url = doc["header"].get("url", "")
    overall = doc["stats"].get("overall rating", "")
    wta = doc["stats"].get("would take again", "")
    chunks: list[dict] = []

    courses_seen: list[str] = []
    for rev in doc["reviews"]:
        meta = rev["meta"]
        course = meta.get("course", "")
        if course and course not in courses_seen:
            courses_seen.append(course)

        # Self-describing prefix replaces overlap's context-preservation job.
        ctx = f"{professor} ({course}, RateMyProfessors review"
        if meta.get("date"):
            ctx += f", {meta['date']}"
        rating_bits = []
        if meta.get("quality"):
            rating_bits.append(f"Quality {meta['quality']}/5")
        if meta.get("difficulty"):
            rating_bits.append(f"Difficulty {meta['difficulty']}/5")
        if rating_bits:
            ctx += " — " + ", ".join(rating_bits)
        ctx += "): "
        text = ctx + rev["text"]

        chunks.append(
            {
                "text": text,
                "metadata": {
                    "professor": professor,
                    "course": course,
                    "source_url": source_url,
                    "source_file": doc["file"],
                    "date": meta.get("date", ""),
                    "quality": meta.get("quality", ""),
                    "difficulty": meta.get("difficulty", ""),
                    "grade": meta.get("grade", ""),
                    "overall_rating": overall,
                    "would_take_again": wta,
                    "chunk_type": "review",
                },
            }
        )

    # One aggregate summary chunk per file.
    courses = "/".join(courses_seen)
    summary_text = (
        f"{professor} — overall RateMyProfessors summary"
        f"{f' ({courses})' if courses else ''}: "
        f"overall rating {overall or 'n/a'}, "
        f"would take again {wta or 'n/a'}, "
        f"difficulty {doc['stats'].get('level of difficulty', 'n/a')}. "
        f"Rating distribution: {doc['stats'].get('rating distribution', 'n/a')}."
    )
    chunks.append(
        {
            "text": summary_text,
            "metadata": {
                "professor": professor,
                "course": courses,
                "source_url": source_url,
                "source_file": doc["file"],
                "date": "",
                "quality": "",
                "difficulty": "",
                "grade": "",
                "overall_rating": overall,
                "would_take_again": wta,
                "chunk_type": "summary",
            },
        }
    )
    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    # Defensive: drop any empty chunk the splitter might have produced.
    return [c for c in chunks if c["text"].strip()]


# --------------------------------------------------------------------------- #
# Inspection / CLI
# --------------------------------------------------------------------------- #
def _inspect(chunks: list[dict]) -> None:
    print(f"\nLoaded {len({c['metadata']['source_file'] for c in chunks})} documents")
    print(f"Produced {len(chunks)} chunks "
          f"({sum(c['metadata']['chunk_type'] == 'review' for c in chunks)} reviews + "
          f"{sum(c['metadata']['chunk_type'] == 'summary' for c in chunks)} summaries)\n")

    lengths = [len(c["text"]) for c in chunks]
    print(f"Chunk length (chars): min={min(lengths)} "
          f"avg={sum(lengths) // len(lengths)} max={max(lengths)}")
    empties = sum(1 for c in chunks if not c["text"].strip())
    print(f"Empty chunks: {empties}\n")

    # 5 representative chunks, evenly spaced across the corpus.
    print("=" * 70)
    print("5 REPRESENTATIVE CHUNKS")
    print("=" * 70)
    step = max(1, len(chunks) // 5)
    for idx in range(0, len(chunks), step)[:5]:
        c = chunks[idx]
        m = c["metadata"]
        print(f"\n[chunk {idx}] type={m['chunk_type']} "
              f"file={m['source_file']} course={m['course']}")
        print(f"  {c['text']}")


def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so em-dashes etc. print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    docs = load_documents()
    chunks = build_chunks(docs)
    _inspect(chunks)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nSaved {len(chunks)} chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
