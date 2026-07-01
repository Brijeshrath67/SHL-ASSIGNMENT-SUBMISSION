# SHL Assessment Advisor — Approach Document

## 1. Architecture & Design Choices

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Client / Evaluator                     │
└──────────────┬──────────────────────────────┬────────────┘
               │ GET /health                  │ POST /chat
               ▼                              ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI (Uvicorn)                        │
│  • CORS middleware (allow all origins)                    │
│  • Pydantic request/response validation                   │
│  • 30s timeout per request                                │
│  • Static HTML UI served at GET /                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Agent Layer                            │
│                                                           │
│  1. Turn cap check (max 16 messages)                     │
│  2. Off-topic / legal refusal check                      │
│  3. Intent classification                                  │
│     ├─ farewell → end conversation                       │
│     ├─ clarify  → ask for more info                      │
│     ├─ recommend → search catalog + respond              │
│     ├─ refine   → merge existing + new recs              │
│     └─ compare  → side-by-side product details           │
│  4. Response generation                                    │
│     ├─ Template (always available)                       │
│     └─ LLM via Groq (if LLM_API_KEY set)                │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  CatalogRetriever                         │
│                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐   │
│  │ Embeddings   │    │  FAISS Index │    │ Keyword   │   │
│  │ (384-dim)    │───▶│  (FlatIP)    │    │ Fallback  │   │
│  └─────────────┘    └──────┬───────┘    └───────────┘   │
│                            │                              │
│                            ▼                              │
│  ┌──────────────────────────────────────────────────┐    │
│  │          120 SHL Catalog Items                    │    │
│  │  (enriched with _search_text for multi-field     │    │
│  │   matching across name, desc, keys, levels,      │    │
│  │   languages)                                      │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Core Components

**FastAPI HTTP Layer** (`app/main.py`):
- Three routes: `GET /` (glassmorphism chat UI), `GET /health` (returns `{"status": "ok"}`), `POST /chat` (main conversation endpoint).
- CORS middleware configured to allow all origins for cross-domain evaluation.
- Request validation via Pydantic `ChatRequest` / `ChatResponse` models with strict type enforcement.
- Server runs behind Uvicorn ASGI server, port configurable via `$PORT` env var (default 7860 for HF Spaces, 8320 for local).

**Agent Layer** (`app/agent.py`):
- **Stateless design**: conversation state lives entirely in the request body's `messages` array. No session store, no database, no server-side state. Turn count derived from `ceil(len(messages)/2)`, capped at 8 turns (16 messages).
- **Intent classifier**: rule-based using regex patterns and keyword sets. Five intents plus two refusal modes. Pure-greeting detection for single-word "hi"/"hello" inputs. Short-answer detection (<3 words) routes to clarification. Pattern matching for comparison (2+ named products with "vs/versus/difference/between"), refinement ("add/include/also/drop/remove/replace"), and farewell ("thanks/bye/confirmed/locking it in/that's good").
- **Dual-path response generation**: template-based responses (always available, zero dependencies) and LLM-enhanced responses via Groq API (optional, enabled by setting `LLM_API_KEY`). LLM path uses a structured prompt with `##REPLY`/`##RECOMMENDATIONS`/`##END` markers for reliable parsing. On LLM failure (timeout, rate limit, parsing error), falls back to templates transparently.
- **Legal/off-topic refusal**: two-layer detection. General off-topic keywords (weather, sports, cooking, etc.) are ignored if hiring context terms are also present. Legal patterns use multi-word regex (e.g., "legally required", "is this legal", "satisfy compliance requirement") to avoid false positives on standalone words like "compliance".

**Retrieval Layer** (`app/retrieval.py`):
- Hybrid search: semantic embedding similarity (primary) + keyword term overlap (secondary), with results deduplicated by entity ID.
- Embedding model cached on disk after first download (~15s cold start), subsequent restarts are near-instant.

## 2. Retrieval Setup

Hybrid retrieval pipeline combining semantic embedding search with keyword fallback.

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — free, lightweight (~80MB), no API key needed. Produces normalized embeddings for cosine similarity via FAISS IndexFlatIP.
- **Index**: built at startup over 120 SHL catalog items. Each item enriched with a `_search_text` field combining name, description, test type keys, job levels, and languages for multi-field matching.
- **Query construction**: all user messages from the conversation are concatenated to form the search query, enabling multi-turn context awareness (e.g., "English" then "US" triggers accent-specific search).
- **Keyword fallback**: if embedding init fails (no internet), a term-overlap scorer ranks items by query term matches in name and description.
- **Location-aware retrieval**: a `LOCATION_QUERY_MAP` maps "us", "uk", "india" etc. to accent-specific search terms for contact-center assessments.

## 3. Prompt Design & Agent Behavior

The LLM system prompt follows a structured format:

- **Role definition**: "You are an expert SHL assessment consultant helping hiring managers."
- **Available products**: formatted list of retrieved catalog items (name, description, test type, levels, duration) — never the full 120-item catalog, to minimize token usage and hallucination risk.
- **Behavior rules**: clarify if vague, recommend 1-10 products, compare when asked, refine on add/drop/replace requests, never hallucinate products, politely refuse off-topic and legal questions.
- **Response format**: structured markers `##REPLY` / `##RECOMMENDATIONS` / `##END` for reliable parsing back into the ChatResponse schema.
- **Off-topic detection**: combines keyword matching (weather, sports, politics, etc.) with hiring-context terms. Legal questions caught by regex patterns like "legally required", "is this legal", "satisfy compliance requirement" — tuned to avoid false positives on words like "compliance" in hiring contexts.

## 4. Evaluation Method

Ten sample conversation traces (C1-C10) are replayed turn-by-turn against the live server. Every response is validated:

- **Schema checks**: `reply` must be string, `recommendations` list of `{name, url, test_type}` with max 10 items, `end_of_conversation` boolean.
- **Hallucination check**: every recommendation URL must contain "shl.com" (non-empty, real SHL links).
- **Behavior probes**: implicitly tests off-topic refusal (weather, legal), clarification flow (short answers), refinement (add/drop), comparison (2+ products), and termination (farewell patterns).
- **Turn cap**: conversations limited to 8 user turns (16 messages). Extra turns receive a termination response.
- **Automation**: `run_tests.py` replays all traces with a single command and exits non-zero on any schema violation.

## 5. What Did Not Work

- **Vector-only retrieval**: pure semantic search missed exact product names in some queries. Added keyword overlap scoring as a complement, then deduplicated by entity ID.
- **LLM-only response generation (no catalog context)**: the LLM hallucinated product names and details. Fixed by restricting the system prompt to only the retrieved catalog subset.
- **Simple "compliance" in legal keywords**: flagged plant-operator safety queries as off-topic. Replaced single-word matching with multi-word regex patterns requiring explicit legal phrasing.
- **Full catalog in LLM context**: exceeded token limits and diluted relevance. Now only top-15 retrieved items are included in the prompt.

## 6. Measuring Improvement

Improvement was measured quantitatively and qualitatively:

- **Schema compliance**: initial runs had 3 trace failures. After fixes, all 10 traces pass with 0 schema errors and 0 hallucination warnings.
- **Recommendation count**: 145 total recommendations across 38 user turns (avg 3.8/turn). Distribution shows 5-8 recs when recommending, 0 when clarifying or refusing.
- **Behavior correctness**: all 10 traces terminate with `end_of_conversation=true` at the final expected turn. Legal questions refused (C7 T3 returns 0 recs with refusal). Comparison prompts return exactly 2 products (C5, C6).
- **Response quality**: LLM-enhanced responses (when `LLM_API_KEY` set) produce more natural, context-aware replies with better product justifications vs template-generated responses.
- **Latency**: template responses <200ms. LLM responses ~2-5s via Groq API. Cold start ~15-20s for embedding model download on first deployment.

---

**API URL**: https://mithulrath-shl-assessment-advisor.hf.space  
**Source**: https://github.com/Brijeshrath67/SHL-ASSIGNMENT-SUBMISSION
