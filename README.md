# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) question-answering system over real student reviews of
Temple University Computer Science professors. Ask about who's good, course
difficulty, workload, or grading, and get an answer grounded **only** in the
collected reviews — with the source documents cited.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows
pip install -r requirements.txt
cp .env.example .env                                   # then add your free Groq key

python ingest.py     # 1. load + clean + chunk  -> data/chunks.json
python index.py      # 2. embed + store in ChromaDB (+ retrieval self-test)
python app.py        # 3. Gradio UI at http://localhost:7860
# or: python query.py            (interactive CLI)
#     python query.py "your question"
```

Pipeline: **Ingestion → Chunking → Embedding + ChromaDB → Retrieval → Groq generation.**
See [planning.md](planning.md) for the full spec and architecture diagram.

---

## Domain

**CS professors and the CS major at Temple University.** The system answers questions
about professor teaching styles, course difficulty, workload, and grading from the
perspective of students who took the classes.

This knowledge is valuable and hard to find through official channels because course
catalogs and syllabi describe *what* a course covers, not *what it's like to take it* —
whether the professor can actually teach, how heavy the workload really is, how grading
works in practice, or whether a class is right for a beginner. That information lives in
scattered student reviews that you'd otherwise have to read one professor at a time.

---

## Document Sources

10 source documents, each a Rate My Professors faculty page collected verbatim
(aggregate stats + individual student reviews). They span intro CS through graduate
courses and the full rating spectrum (1.9★–4.5★), so the corpus covers "who's good,"
"who to avoid," difficulty, and workload from many angles.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Andrew Rosen (CIS1051/2168, 4.5★) | Rate My Professors | [data/raw/rmp_rosen.txt](data/raw/rmp_rosen.txt) — https://www.ratemyprofessors.com/professor/2173472 |
| 2 | Slobodan Vucetic (CIS3715/5526, 4.0★) | Rate My Professors | [data/raw/rmp_vucetic.txt](data/raw/rmp_vucetic.txt) — https://www.ratemyprofessors.com/professor/1080367 |
| 3 | Christopher Pascucci (CIS3309/3342, 3.5★) | Rate My Professors | [data/raw/rmp_pascucci.txt](data/raw/rmp_pascucci.txt) — https://www.ratemyprofessors.com/professor/995042 |
| 4 | Karl Morris (CIS3515, 3.6★) | Rate My Professors | [data/raw/rmp_morris.txt](data/raw/rmp_morris.txt) — https://www.ratemyprofessors.com/professor/1922744 |
| 5 | Justin Yuan Shi (CIS3207, 3.2★) | Rate My Professors | [data/raw/rmp_shi.txt](data/raw/rmp_shi.txt) — https://www.ratemyprofessors.com/professor/1031508 |
| 6 | David Dobor (CIS2168, 3.3★) | Rate My Professors | [data/raw/rmp_dobor.txt](data/raw/rmp_dobor.txt) — https://www.ratemyprofessors.com/professor/2206996 |
| 7 | Nancy Polychronopoulou (CIS2109, 2.2★) | Rate My Professors | [data/raw/rmp_polychronopoulou.txt](data/raw/rmp_polychronopoulou.txt) — https://www.ratemyprofessors.com/professor/2684253 |
| 8 | Hani Karam (CIS1057/3100, 2.2★) | Rate My Professors | [data/raw/rmp_karam.txt](data/raw/rmp_karam.txt) — https://www.ratemyprofessors.com/professor/2672366 |
| 9 | Pei Wang (CIS2033/5511, 2.4★) | Rate My Professors | [data/raw/rmp_wang.txt](data/raw/rmp_wang.txt) — https://www.ratemyprofessors.com/professor/991172 |
| 10 | Richard Beigel (CIS1166/2033, 1.9★) | Rate My Professors | [data/raw/rmp_beigel.txt](data/raw/rmp_beigel.txt) — https://www.ratemyprofessors.com/professor/810741 |

> **Honesty note on source variety.** The original plan also included Reddit r/Temple
> threads and student-advice guides. During collection, Reddit blocked the automated
> fetcher entirely (its robots policy denies the crawler) and Coursicle rate-limited
> (HTTP 429) every request. Rather than fabricate forum content, the corpus is built
> from 10 real, verbatim RMP pages. The trade-off is less source-*type* variety (one
> platform), which shows up in the failure case below.

---

## Chunking Strategy

**Chunk size:** One student review per chunk — a *semantic unit*, not a fixed length.
Reviews run ~1–3 sentences (~40–120 tokens / 150–450 characters). Each file also gets one
"summary" chunk holding its aggregate stats (overall rating, would-take-again %, difficulty,
rating distribution).

**Overlap:** 0. Reviews are independent — review N+1 does not continue review N's thought —
so overlap would only bleed one student's words into a neighbor's chunk and create false
attribution. The job overlap normally does (preserving context across a boundary) is instead
handled structurally: **each chunk is prefixed with the professor name + course code**, so a
chunk like *"RUNNNNN don't take this class"* still carries *"Nancy Polychronopoulou — CIS2109"*
and is self-describing.

**Why these choices fit the documents:** Every file is a *list of short, independent
reviews* delimited by a `[CIS#### | date | Quality | Difficulty]` header line — not
continuous prose. So I split on those header lines rather than a fixed character count.
A 300–400-token fixed window (a common default) would pack 3–5 *different students'
opinions* — often contradictory — into one chunk, and retrieval would return a blob where
the model can't tell whose opinion is whose.

**Preprocessing:** `clean_text()` unescapes HTML entities, strips any HTML tags, and
collapses whitespace/nbsp while preserving line breaks. (The collected files are already
plain text, so this is mostly defensive for future messier sources.)

**Final chunk count:** **60 chunks** (50 reviews + 10 summaries), verified self-contained,
0 empty, 144–336 characters each.

---

## Embedding Model

**Model used:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional, runs locally on
CPU, free, no API key, and its 256-token input limit comfortably fits the tiny review chunks.
A strong default for short English text.

> **Implementation detail that mattered:** the system *stores* the rich, attribution-friendly
> chunk text but *embeds* a leaner `Professor (course): review text` string that drops the date
> and Quality/Difficulty numbers. Those fields are near-identical boilerplate across all 50
> reviews and, on a short review, dominate the token budget and wash out the distinctive
> opinion. Dropping them from the embedded string lowered cosine distance on every test query
> (e.g. Q1 0.349 → 0.302).

**Production tradeoff reflection:** If cost weren't a constraint I'd weigh **(1) domain
accuracy** — a larger model like `text-embedding-3-large` or Cohere `embed-english-v3.0`
better distinguishes professor names and CIS course jargon, reducing the chance that two
professors with similar review text collapse together (the root of my failure case);
**(2) context length** — irrelevant here (chunks are tiny) but it would matter if I added long
Reddit guides; **(3) latency** — MiniLM is milliseconds locally vs. a network round-trip per
query for an API model; **(4) multilingual** — not needed for an English-only Temple corpus.
Given short, opinionated, English review text, MiniLM is the right pick. I'd only upgrade if
evaluation showed retrieval confusing similar professors, and I'd pair the upgrade with a
re-ranking pass.

---

## Grounded Generation

**LLM:** Groq `llama-3.3-70b-versatile` (free tier), `temperature=0`.

Grounding is enforced in **three layers**, not left to the model's good intentions:

1. **Relevance gate (pre-LLM).** Retrieval returns top-5 chunks with cosine distances. If
   the *closest* chunk is farther than **0.60**, the system returns a fixed
   *"I don't have enough information on that…"* message **without calling the LLM at all**.
   This threshold comes from observation: real answers retrieve at 0.30–0.49, while
   out-of-corpus probes ("closest dorm", "capital of France") sit at 0.68–0.89.

2. **System prompt grounding instruction (verbatim):**
   > *"You answer ONLY using the student reviews provided in the CONTEXT below. Use only facts
   > found in the CONTEXT. Do NOT use any outside or prior knowledge about these professors,
   > courses, or Temple University. If the CONTEXT does not contain enough information to
   > answer, reply exactly: 'I don't have enough information on that in the student reviews I
   > have.' Attribute claims to the professor they are about. When reviews disagree, say so
   > rather than picking a side. Do not invent ratings, courses, quotes, or professor names
   > that are not in the CONTEXT."*

3. **Source attribution (programmatic).** Sources are **not** left to the model to write. After
   generation, `query.py` builds the source list from the chunks that were actually used,
   then restricts it to the professors the answer actually names (full name or surname). This
   prevents over-citing: retrieval often pulls chunks from several professors who share a
   course code (e.g. CIS2033), but only the ones reflected in the answer are cited.

**How source attribution is surfaced:** each answer is followed by a "Retrieved from" list of
`filename (Professor) — URL`, e.g. `rmp_dobor.txt (David Dobor) — https://…`.

---

## Evaluation Report

All five questions were run through the live system (`query.py`). Responses are summarized;
full text is reproducible by running the questions.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which Temple CS professor do students most recommend, and why? | Andrew Rosen — highest rated (4.5★, 92% would take again); fair, funny, wants students to succeed | Named **Andrew Rosen** ("one of the best professors at Temple", great teaching) — correct — **but also added Christopher Pascucci** based on a "learned the most" review | Relevant (Rosen top 2) but mixed in a positive Pascucci chunk | **Partially accurate** |
| 2 | How heavy is the workload in Christopher Pascucci's web-dev courses (CIS3309/3342)? | Very heavy; difficulty 4–5/5; "most work I'd ever done"; time management critical | "Very challenging… 'most amount of work I had ever done at Temple'… difficulty 5.0/5… time management more critical than mastery" | Relevant (all 5 chunks Pascucci) | **Accurate** |
| 3 | What do students say about taking Data Structures (CIS2168) with David Dobor? | Divisive — strict/overwhelming for some, "favorite professor ever" for others | "Difficult class, very strict about small things, lecture/test heavy, lots of homework; one student wouldn't choose it again" — **captured only the negative half** | Partially relevant (top chunk Dobor; positive Dobor chunk fell just outside top-5) | **Partially accurate** |
| 4 | Is Richard Beigel's class hard, and how is the grade determined? | Low overall (1.9★), weak lecturer, quiz-driven grade (100+ quizzes, unlimited retakes, power-mean) | "Difficulty ~3.7/5; grading rewards 'structure mastery'; 100+ quizzes with unlimited retakes" | Relevant (3 of 5 chunks Beigel) | **Accurate** |
| 5 | Which intro-level CS professor do reviews warn beginners about? | Hani Karam (CIS1057) — review states "Not recommended for beginners" | Named **Justin Yuan Shi** (CIS3207) as "one of the most disorganized," while noting "it's not explicitly stated that these classes are intro-level" | **Off-target** (Karam/CIS1057 never retrieved) | **Inaccurate** — see Failure Case |

**Summary:** 2 accurate, 2 partially accurate, 1 inaccurate. The two "partial" results share a
cause with the failure case: semantic retrieval surfaces the *loudest* relevant chunks, which
can crowd out a quieter-but-correct review (Q3's positive Dobor reviews) or pull in an
adjacent positive review (Q1's Pascucci).

---

## Failure Case Analysis

**Question that failed:** *"Which intro-level CS professor do reviews warn beginners about?"*
(Q5). The corpus does contain the answer: Hani Karam's CIS1057 review literally says
*"Not recommended for beginners."*

**What the system returned:** A confident answer naming **Justin Yuan Shi** as the professor to
warn beginners about, citing his CIS3207/CIS3238 reviews ("one of the most disorganized",
"can't teach"). Karam's file was never retrieved (not even in the top 10).

**Root cause (embedding / retrieval stage):** Two compounding effects, both in the embedding step.
1. **The embedding model weights sentiment over the literal constraint.** all-MiniLM-L6-v2 maps
   "warn beginners about / professor to avoid" close to emphatic negative reviews
   ("most disorganized", "can't teach") regardless of course level. It has no knowledge that
   CIS1057 *is* an intro course, so the word "intro-level" exerts little pull toward Karam.
2. **The correct review is a 4-word fragment.** Karam's relevant review is just
   *"Not recommended for beginners. Tags: Tough grader"* — very little text to embed. Its signal
   is weak compared to Shi's longer, more vivid negative reviews, so it loses the similarity race.
   Because Shi's chunk passes the 0.60 relevance gate (distance 0.45), the system does **not**
   decline — it grounds confidently in the wrong source.

**What I would change to fix it:**
- **Enrich short chunks with course-level metadata** so "intro-level" can match: add a derived
  tag like `level: introductory` (from the course number, CIS10xx/CIS1057) into the embedded
  text. This gives the constraint something to latch onto.
- **Add a metadata pre-filter** for queries that name a course level or number, narrowing the
  candidate set before semantic ranking (hybrid retrieval).
- **A stronger embedding model** (see tradeoff section) that better separates "negative review"
  from "intro-level negative review." I deliberately did *not* hand-tune the retriever to force
  this one query, since that would overfit to the eval set and hide the real limitation.

---

## Spec Reflection

**One way the spec helped during implementation:** Deciding the chunking strategy *before*
writing code — "one review = one chunk, 0 overlap, prefix with professor + course" — gave the
ingestion code a precise target and a built-in test. Because the spec named the exact failure
modes of the alternatives (a fixed-size window would merge contradictory opinions; overlap
would cause false attribution), the verification step was obvious: print chunks and confirm
each is one self-contained review attributed to one professor. The implementation matched the
spec on the first pass and the 60-chunk count landed exactly where the spec predicted.

**One way the implementation diverged from the spec, and why:** The spec said to *"prefix each
chunk with professor + course"* and treat that prefixed string as the chunk. In practice I split
it into **two representations**: a rich prefixed string that is *stored* (for display and for the
LLM) and a *leaner* string that is actually *embedded* (dropping the date and Quality/Difficulty
boilerplate). I discovered during Milestone 4 retrieval testing that the verbose prefix — being
nearly identical across all 50 reviews — diluted the semantic signal of short reviews and inflated
distances. Separating "what we embed" from "what we store" fixed it and lowered distances on every
query. I updated planning.md to record this change.

---

## AI Usage

**Instance 1 — Chunking + the embed/store split (Milestones 3–4).**
- *What I gave the AI:* my Chunking Strategy section plus the full text of one source file
  (`rmp_dobor.txt`) so it could see the exact header and `[CIS#### | … ]` review format.
- *What it produced:* a parser and a chunker that emitted one chunk per review with a rich
  `Professor (course, date — Quality/Difficulty): text` prefix, used as **both** the stored and
  embedded text.
- *What I changed / directed differently:* after testing retrieval, the shortest reviews (e.g.
  Karam's *"Not recommended for beginners"*) retrieved poorly because the identical verbose prefix
  dominated the embedding. I directed the AI to add a separate lean `embed_text` field
  (`Professor (course): text`) for vectorization while keeping the rich text for display, and
  verified that distances dropped on all five queries.

**Instance 2 — Grounded generation + honest source attribution (Milestone 5).**
- *What I gave the AI:* my grounding requirements from the Anticipated Challenges section (answer
  only from context, decline when unsupported, cite sources) and the retrieval output shape.
- *What it produced:* an `ask()` function calling Groq with a grounding system prompt, and a source
  list built from **all** retrieved chunks.
- *What I changed / overrode:* the initial source list over-cited — for a Dobor question it listed
  `rmp_rosen.txt` too, because a Rosen chunk had been retrieved (several professors share CIS2033).
  I directed the AI to restrict the programmatic source list to professors the answer actually
  *names*, and I added a pre-LLM relevance gate (decline if the best distance > 0.60) so
  out-of-corpus questions are refused rather than answered from a loosely-related chunk.

---

## Demo Video

*(Add your 3–5 minute recording link here.)* The demo should show: at least 3 queries with
visible source citations; one query where retrieval + generation both work well (e.g. Q2,
Pascucci workload); one where the system fails, narrated (Q5 — names Shi instead of Karam, the
failure case above); and a walkthrough of this evaluation report.
