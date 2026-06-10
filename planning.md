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

**Chunk size:** 300-400 tokens (~1000-1200 characters)

**Overlap:** 50 tokens (~200 characters)

**Reasoning:** Your corpus is a mix of short RateMyProfessors reviews (often 1-3 paragraphs) and longer Reddit threads and student guides. 300-400 tokens provides enough context to capture a complete thought (e.g., one professor's teaching style or a specific course tip) while avoiding excessive redundancy. Reddit threads benefit from some overlap to preserve context across sentence boundaries, especially for opinion statements that reference earlier discussion.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-MiniLM-L6-v2 (via sentence-transformers)

**Top-k:** 4

**Production tradeoff reflection:** For real deployment, I'd prioritize domain-specific accuracy over speed/cost. A fine-tuned model like instructor-large or Cohere's embed-english-v3.0 would improve retrieval quality by better understanding professor names and course-specific jargon (e.g., "weeder courses"). The 4-chunk default balances: getting multiple student perspectives on the same topic while staying under latency limits. If cost allowed, retrieval could increase to 6-8 chunks and implement re-ranking to surface the most relevant reviews first.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Which CS professors at Temple are considered the best by students? | Names of professors frequently recommended in Reddit discussions and RMP reviews with reasons (engaging teaching, fair grading, helpful) |
| 2 | What should I know before taking Intro to CS? | Common tips about workload, prerequisites, study strategies, and professor-specific advice from student guides |
| 3 | How difficult is the Data Structures course? | Student reports on difficulty level, time commitment, grading practices, and specific professor comparisons |
| 4 | What is the overall CS major workload like at Temple? | General difficulty/workload assessment compared to other majors, time commitment expectations, and course progression advice |
| 5 | What are students saying about [specific professor name]'s teaching style? | Consolidated feedback from RMP and Reddit on grading, clarity, approachability, and course difficulty |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Outdated or conflicting information across sources.** RMP reviews span multiple years and professors may teach differently now. Student advice might reference old prerequisite structures. Multiple chunks retrieved may contradict each other (e.g., "Professor X is easy" vs. "Professor X grades harshly"), making summarization difficult. Mitigation: Include date/source metadata in chunks; prioritize recent reviews.

2. **Off-topic retrieval for vague queries.** Queries like "best professor" or "hard class" are ambiguous — the system might retrieve reviews about difficulty when the user wants grading style, or retrieve unrelated courses with similar names. Reddit discussions often go off-topic. Mitigation: Chunk document sections clearly with headers; use top-k=4 to include multiple perspectives and filter noisy results.

3. **Chunks that split key information.** A single professor's review might span 200+ tokens; chunking at 300 tokens could split one review across boundaries, making it unclear which opinions belong to which professor. Mitigation: Preserve document boundaries when possible; ensure overlap captures professor names and course titles.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  DOCUMENT INGESTION                                                 │
│  Load .txt files from data/raw/ directory                          │
│  (Reddit threads, RMP reviews, student advice guides)              │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHUNKING                                                           │
│  LangChain RecursiveCharacterTextSplitter                           │
│  chunk_size=300-400 tokens | chunk_overlap=50 tokens               │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EMBEDDING + VECTOR STORE                                           │
│  Embedding: sentence-transformers/all-MiniLM-L6-v2                 │
│  Vector Store: FAISS (in-memory) or Pinecone (cloud)               │
│  Store: chunk_id, text, metadata (source, date)                    │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL                                                          │
│  Input: User query (e.g., "best CS professors at Temple")           │
│  Embed query with same model                                        │
│  Similarity search: retrieve top-k=4 chunks                         │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GENERATION (Optional)                                              │
│  Pass retrieved chunks + query to LLM (Claude, GPT-4)               │
│  Prompt: Synthesize student perspectives into concise answer        │
│  Output: Natural language response with source attribution          │
└─────────────────────────────────────────────────────────────────────┘
```
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

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

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
