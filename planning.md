# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

**CS Professors and Major at Temple University**

Student knowledge about computer science professors, course difficulty, and major-related survival advice at Temple University. This knowledge is valuable and hard to find officially because it represents authentic student experiences, specific professor teaching styles, course difficulty levels, and practical tips — information that official department websites and syllabi cannot convey. Students currently must search through Reddit threads and Rate My Professors individually, with inconsistent and often fragmented information.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

Each source is one Rate My Professors faculty page, collected verbatim (aggregate stats + individual student reviews). Together they span intro CS, data structures, discrete math, systems, web development, and graduate courses, across the full rating spectrum (1.9 to 4.5 overall), so they cover "who's good," "who to avoid," course difficulty, and workload from many angles.

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Rate My Professors | Andrew Rosen — CIS1051/2168, 4.5★, the most-recommended prof | data/raw/rmp_rosen.txt — https://www.ratemyprofessors.com/professor/2173472 |
| 2 | Rate My Professors | Slobodan Vucetic — CIS3715/5526 data science, 4.0★, heavy labs | data/raw/rmp_vucetic.txt — https://www.ratemyprofessors.com/professor/1080367 |
| 3 | Rate My Professors | Christopher Pascucci — CIS3309/3342 web dev, 3.5★, very high workload | data/raw/rmp_pascucci.txt — https://www.ratemyprofessors.com/professor/995042 |
| 4 | Rate My Professors | Karl Morris — CIS3515 mobile/Android, 3.6★ | data/raw/rmp_morris.txt — https://www.ratemyprofessors.com/professor/1922744 |
| 5 | Rate My Professors | Justin Yuan Shi — CIS3207 systems, 3.2★, "disorganized" | data/raw/rmp_shi.txt — https://www.ratemyprofessors.com/professor/1031508 |
| 6 | Rate My Professors | David Dobor — CIS2168 data structures, 3.3★, divisive | data/raw/rmp_dobor.txt — https://www.ratemyprofessors.com/professor/2206996 |
| 7 | Rate My Professors | Nancy Polychronopoulou — CIS2109, 2.2★ | data/raw/rmp_polychronopoulou.txt — https://www.ratemyprofessors.com/professor/2684253 |
| 8 | Rate My Professors | Hani Karam — CIS1057 intro C / CIS3100, 2.2★ | data/raw/rmp_karam.txt — https://www.ratemyprofessors.com/professor/2672366 |
| 9 | Rate My Professors | Pei Wang — CIS2033/5511 algorithms, 2.4★ | data/raw/rmp_wang.txt — https://www.ratemyprofessors.com/professor/991172 |
| 10 | Rate My Professors | Richard Beigel — CIS1166 discrete / CIS2033, 1.9★, quiz-heavy | data/raw/rmp_beigel.txt — https://www.ratemyprofessors.com/professor/810741 |

> **Collection note (honesty):** The original plan also listed Reddit r/Temple threads and internal student-advice guides. During automated collection, **Reddit blocked the fetcher entirely** (its robots policy denies the crawler) and **Coursicle rate-limited (HTTP 429)** every request. Rather than fabricate Reddit/forum content, the corpus is built from 10 real, verbatim RMP faculty pages. Trade-off: less source-*type* variety (one platform), but each page is genuinely distinct and the courses/ratings span the domain well. Reddit threads can be added later by manual copy-paste if richer "survival advice" prose is needed.

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** One student review per chunk — a *semantic unit*, not a fixed length. Each review runs ~1–3 sentences (~40–120 tokens / ~150–450 characters). A separate "professor summary" chunk per file holds the aggregate stats (overall rating, would-take-again %, difficulty, rating distribution). A 512-token hard cap acts as a safety splitter in case any single review is unexpectedly long. Result: ~6 chunks per file (5 reviews + 1 summary) × 10 files ≈ **55–60 chunks total**.

**Overlap:** 0 tokens (none).

**Reasoning:** Every document here is a *collection of short, independent reviews* separated by a `[CIS#### | date | Quality | Difficulty]` header line — not continuous prose. The natural semantic unit is a single review, so I split on those header lines rather than on a fixed character count.

- **Why not fixed-size 300–400-token chunks?** Each review is ~40–120 tokens, so a 300–400-token window would pack 3–5 *different students' opinions* — often contradictory ("one of the best" next to "RUNNNNN don't take this class") — into one chunk. Retrieval would then return a blob where the model can't tell whose opinion is whose, and similarity scores get muddied by averaging opposing sentiment.
- **Why 0 overlap?** Overlap exists to rescue a fact whose meaning flows *across* a boundary in continuous prose. Reviews don't flow into each other — review N+1 doesn't continue review N's thought — so overlap would just bleed one student's words into a neighbor's chunk, creating false attribution. The job overlap normally does (preserving context) is instead handled structurally: **each chunk is prefixed with the professor name + course code(s)**, so a chunk like "RUNNNNN don't take this class" still carries "Nancy Polychronopoulou — CIS2109" and is self-describing on its own.
- **How I'd know it's wrong:** *Too large* → answers about one professor leak quotes about another, or "best professor" returns a chunk mixing 4.5★ and 1.9★ profs. *Too small* (e.g., splitting a review mid-sentence) → retrieval returns a dangling fragment ("but he curves the class") with no subject. One-review-per-chunk avoids both.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional, runs locally on CPU, free, and its 256-token input limit comfortably fits my ~40–120-token review chunks. It's a strong default for short English text and needs no API key, which keeps the whole retrieval side offline.

**Top-k:** 5. Most of my test questions are *professor-specific* ("what do students say about Dobor?"), and each professor has 5 review chunks + 1 summary chunk. k=5 surfaces several reviews for the target professor so the model sees the spread of opinion (not just one outlier), while staying small enough that off-topic chunks rarely sneak in.
- *Too few (k=1–2):* a single review can be an outlier — pulling only the 1★ "RUNNNNN" review for a professor who's actually mixed would give a misleading answer.
- *Too many (k=15):* for a question about one professor, the extra chunks are necessarily *other* professors, diluting context and inviting the model to wander off-topic.

**Why semantic search works here:** the query "is this class a lot of work?" never contains the word "homework," yet MiniLM maps both into nearby vectors because they co-occur in similar contexts in its training data. That's the point of embeddings over keyword search — it matches *meaning*, so "heavy workload," "lots of homework," and "took 4–6 hours" all retrieve for a workload question even with zero shared words.

**Implementation refinement (Milestone 4):** I separate the *embedding* text from the *stored/display* text. Each chunk stores the rich, attribution-friendly string (`Professor (course, RateMyProfessors review, date — Quality X/5, Difficulty Y/5): …`) for display and for the LLM, but I **embed a lean version** — `Professor (course): review text` — that drops the date and the Quality/Difficulty numbers. Reason: those fields are near-identical boilerplate across all 50 review chunks, and on a short review they dominate the token budget and wash out the distinctive opinion. Removing them from the embedded string measurably lowered distances on every test query (e.g. Q1 0.349 → 0.302). The out-of-corpus probe ("which dorm is closest?") stays high at ~0.67, which sets a natural relevance floor for the grounding step in Milestone 5.

**Production tradeoff reflection:** If cost weren't a constraint, I'd weigh: **(1) domain accuracy** — a larger model like `text-embedding-3-large` or Cohere `embed-english-v3.0` better distinguishes professor names and CIS course jargon, reducing the risk that two profs with similar review text collapse together; **(2) context length** — irrelevant here (chunks are tiny) but it would matter if I added long Reddit guides; **(3) latency** — MiniLM is ~milliseconds locally vs. a network round-trip per query for an API model; **(4) multilingual** — not needed for an English-only Temple corpus. Given my data is short, opinionated, English review text, MiniLM is the right pick; I'd only upgrade if evaluation showed retrieval confusing similar professors, and I'd pair the upgrade with a re-ranking pass before spending on a bigger embedder.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

Each question maps to a specific source file so a grader can check the system's answer against the ground-truth reviews.

| # | Question | Expected answer (checkable against) |
|---|----------|-----------------|
| 1 | Which Temple CS professor do students most recommend, and why? | **Andrew Rosen** — highest rating in the corpus (4.5★, 92% would take again). Praised as understanding, fair, funny ("cracks himself up"), wants everyone to succeed; grading easier than expected with extra credit (CIS1051/CIS2168). *(rmp_rosen.txt)* |
| 2 | How heavy is the workload in Christopher Pascucci's web-dev courses (CIS3309/3342)? | **Very heavy.** Difficulty rated 4–5/5 on nearly every review; "most amount of work I had ever done at Temple"; time management called more critical than mastery; office-hours attendance important. Mixed payoff — some say they learned the most and got hired off the project work. *(rmp_pascucci.txt)* |
| 3 | What do students say about taking Data Structures (CIS2168) with David Dobor? | **Divisive** (3.3★, 55% would take again). Strict about *how* you phrase answers, lecture/test-heavy, lots of homework, "overwhelming"; but fans call him passionate, funny, a favorite — "his teaching style is not for everyone." *(rmp_dobor.txt)* |
| 4 | Is Richard Beigel's class hard, and how is the grade determined? | Low overall (1.9★) and seen as a weak lecturer ("couldn't teach a fish how to swim," "scrolls through 50-page google docs"), but the grade is **quiz-driven**: 100+ weekly quizzes with unlimited retakes and a power-mean formula, so you can "play the game" to pass; attendance mandatory. *(rmp_beigel.txt)* |
| 5 | Which intro-level CS professor do reviews warn beginners about? | **Hani Karam** for CIS1057 — a review states plainly "Not recommended for beginners"; tough grader, lectures described as uninteresting / "watch videos," "extremely unempathetic." (One dissenting review calls him fair but difficult.) *(rmp_karam.txt)* |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Genuinely contradictory reviews for the same professor.** This is intrinsic to the data, not noise: Dobor is "one of my favorite professors ever" *and* "I wouldn't choose this class again" in the same file; Beigel has 10 Awesome and 49 Awful ratings. If retrieval happens to pull only one polarity, the answer misrepresents the consensus. *Mitigation:* attach `quality`, `difficulty`, and the per-professor `overall_rating`/`would_take_again` as chunk metadata; include the aggregate "summary" chunk so the model can frame individual quotes against the overall score; set top-k=5 so multiple opinions surface together; instruct the LLM to acknowledge disagreement rather than pick a side.

2. **Cross-professor leakage on broad queries.** A vague query like "is CS hard at Temple?" has no single subject, so semantic search may return chunks from several professors and the model could blur them into one false claim ("the professor is disorganized") without saying *which* professor. Several profs also share course codes (CIS2033 appears for Dobor, Wang, and Beigel), inviting confusion. *Mitigation:* prefix every chunk with its professor name + course code so attribution travels with the text; require the generation prompt to name the professor for each claim; surface source filenames/URLs in the answer.

3. **Missing-coverage questions answered from thin air (hallucination).** The corpus is RMP-only and professor-centric. Questions it can't answer — "which dorm is near the CS building?", "what's the average CS starting salary?", or any professor not among the 10 collected — risk the LLM inventing a plausible response. *Mitigation:* a strict grounding system prompt that forbids using outside knowledge and instructs the model to reply "the reviews I have don't cover that" when retrieved chunks are off-topic; optionally drop chunks below a similarity threshold before generation.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  1. DOCUMENT INGESTION                          [ Python stdlib ]       │
│  Read all data/raw/rmp_*.txt files; parse the header block (professor,  │
│  course, URL, aggregate stats) and the list of individual reviews.      │
└───────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  2. CHUNKING                          [ custom delimiter splitter ]     │
│  Split on the `[CIS#### | date | ...]` review header → ONE review =     │
│  ONE chunk. 0 overlap. Prefix each chunk with "Professor — course".     │
│  Emit 1 summary chunk/file from the aggregate stats. (~55–60 chunks)    │
└───────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  3. EMBEDDING + VECTOR STORE   [ all-MiniLM-L6-v2 + ChromaDB (local) ]  │
│  Embed each chunk (384-dim) with sentence-transformers; persist to a    │
│  Chroma collection with metadata: professor, course_codes, source_url,  │
│  date, quality, difficulty, overall_rating, chunk_type(review|summary). │
└───────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  4. RETRIEVAL                  [ ChromaDB similarity search ]           │
│  Embed the user query with the SAME model; cosine-similarity search;    │
│  return top-k = 5 chunks (text + metadata).                             │
└───────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  5. GENERATION                 [ Groq API — llama-3.3-70b-versatile ]   │
│  Grounded system prompt: answer ONLY from the retrieved reviews, name   │
│  the professor for each claim, cite source, say "the reviews don't      │
│  cover that" if unsupported. Surfaced via CLI (Milestone 5; Streamlit   │
│  optional). Output: grounded answer + source attribution.               │
└───────────────────────────────────────────────────────────────────────┘
```

> Note: the diagram reflects the stack pinned in `requirements.txt` — `sentence-transformers`, `chromadb`, `groq`, `python-dotenv`. No LangChain (chunking is a small custom function); embeddings/vector store run locally, only generation calls an API (Groq, free key).

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

I'm using **Claude (Claude Code)** as the primary coding assistant. For each milestone I give it the relevant planning.md section plus one real sample file so the generated code matches my actual data format, then verify against a concrete check.

**Milestone 3 — Ingestion and chunking**
- *Tool:* Claude.
- *Input I give it:* the **Documents** section + the full text of one file (e.g. `data/raw/rmp_dobor.txt`) so it sees the exact `Source:/Professor:/URL:` header and `[CIS#### | date | Quality | Difficulty]` review format, plus my **Chunking Strategy** section.
- *What I expect it to produce:* `load_documents(dir)` that parses each file into `{header_metadata, [reviews]}`, and `chunk_reviews(doc)` that emits one chunk per review (prefixed with "Professor — course") plus one summary chunk, returning `{text, metadata}` dicts. Explicitly **not** a fixed-size splitter.
- *How I verify:* total chunk count is ~55–60; print 5 random chunks and confirm none mixes two professors and none is a mid-sentence fragment.

**Milestone 4 — Embedding and retrieval**
- *Tool:* Claude.
- *Input I give it:* my **Retrieval Approach** section + the chunk dict schema from M3.
- *What I expect it to produce:* `build_index(chunks)` that embeds with `all-MiniLM-L6-v2` and upserts into a persistent ChromaDB collection with my metadata fields; `retrieve(query, k=5)` that embeds the query with the same model and returns the top-5 chunks with text + metadata + distances.
- *How I verify:* run my 5 evaluation questions through `retrieve()` and confirm the top chunks are the expected professor's reviews (e.g. Q3 returns Dobor/CIS2168 chunks, not Beigel's).

**Milestone 5 — Generation and interface**
- *Tool:* Claude.
- *Input I give it:* my **Anticipated Challenges** mitigations (grounding rules) + the retrieval output shape, and the note that `requirements.txt` pins `groq`.
- *What I expect it to produce:* `answer(query)` that calls `retrieve()`, formats the chunks into a context block, sends them with a strict grounding system prompt to a Groq chat model (`llama-3.3-70b-versatile`), and returns the answer with per-claim professor names + source citations; plus a simple CLI loop (`while` reading stdin).
- *How I verify:* run all 5 eval questions and score them in the README's Evaluation Report; deliberately ask an out-of-corpus question ("which dorm is closest to campus?") and confirm it answers "the reviews I have don't cover that" instead of hallucinating.
