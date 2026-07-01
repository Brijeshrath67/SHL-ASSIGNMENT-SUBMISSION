## Approach: SHL Assessment Recommender Agent

### 1. Architecture Overview

```
User → POST /chat → FastAPI → Agent → CatalogRetriever → FAISS + Embedding
                            ↓                     ↓
                        Intent Classifier    SentenceTransformer
                            ↓                     ↓
                     Template / LLM Groq    Catalog (120 items)
                            ↓
                  ChatResponse (reply + recommendations)
```

Two-layer architecture:
- **FastAPI server** (`app/main.py`): Stateless HTTP endpoints — `GET /health` and `POST /chat`. Each request carries full message history; no server-side session storage.
- **Agent core** (`app/agent.py`): Intent classification, retrieval, and response generation (template-based primary path, Groq LLM for enhancement).

### 2. Key Design Decisions

**Free embedding + vector search over paid LLM for core retrieval.**
We use `sentence-transformers/all-MiniLM-L6-v2` + FAISS IndexFlatIP to embed all 120 catalog items at startup. The search query is a concatenation of all user messages (for multi-turn context). A keyword-search fallback activates if embedding init fails. This avoids per-query LLM cost and keeps cold-start overhead to a one-time ~15s model download.

**LLM as optional enhancement, not the backbone.**
The Groq-hosted Llama 3.3 70B is called only when `LLM_API_KEY` is set; template-based responses handle the common path. The LLM receives only the retrieved catalog subset (not the full catalog) to avoid hallucinations and reduce token usage. On LLM failure, it gracefully falls back to templates.

**Intent classification is rule-based, not LLM-based.**
Five intents (clarify, recommend, refine, compare, farewell) are detected via regex patterns and keyword matching. This is faster, cheaper, and more predictable for the constrained domain. Off-topic/legal refusal uses separate pattern sets with careful phrasing to avoid false positives (e.g., "compliance" in a hiring context must not trigger refusal).

**Conversation state is entirely in the request body.**
No session store, no state management. Each `/chat` call receives all prior messages. Turn count is derived from `⌈len(messages)/2⌉`, capped at 8 turns (16 messages). A 30-second timeout applies per Gateway spec.

### 3. Catalog

120 individual SHL assessment products from the fallback catalog, filtered to exclude pre-packaged "Solution" bundles (except telephone/phone simulations). Each item has: name, URL, description, test type (keys), job levels, duration, and languages. The catalog loader normalizes field names (`link`→`url`, `keys`→`test_type`) for schema compliance.

### 4. Agent Behavior

| Intent | Trigger | Behavior |
|--------|---------|----------|
| Clarify | Short answers (<3 words), pure greetings, single locations/languages | Asks for missing info (role, skills, level). For contact center + language → asks about accent/dialect. For "US"/"UK" → triggers location-aware search. |
| Recommend | Role/skill/level mentioned | Builds query from all user messages, retrieves top 15 via FAISS, returns top 5-8 with descriptions. |
| Refine | "Add X", "Drop X", "Replace X with Y" | Merges previous recommendations with new search results, avoids duplicates. |
| Compare | 2+ named products with "vs/compare/difference" | Returns both products with side-by-side details (template) or structured comparison (LLM). |
| Farewell | "Thanks", "Confirmed", "That's good", "Locking it in" | Ends conversation (`end_of_conversation: true`), 0 recommendations. |
| Refuse | Off-topic keywords, legal patterns | Returns refusal message, 0 recommendations, conversation continues (not ended). |

### 5. Schema

```json
{
  "reply": "string",
  "recommendations": [{"name": "str", "url": "str", "test_type": ["str"]}],
  "end_of_conversation": bool
}
```

- `recommendations` is empty when gathering info or refusing, 1-10 when committing.
- All URLs are real SHL catalog URLs from the loaded data.
- `end_of_conversation` goes `true` only on farewell intent.

### 6. Evaluation Results

All 10 traces (C1-C10) replayed against the running server:

| Trace | Turns | Recs | Schema Issues | Notes |
|-------|-------|------|---------------|-------|
| C1 | 4 | 18 | 0 | Senior leadership → OPQ + reports |
| C2 | 3 | 10 | 0 | Rust engineer → technical assessments |
| C3 | 5 | 21 | 0 | Contact center → location-aware search |
| C4 | 3 | 10 | 0 | Graduate financial → refine with SJT |
| C5 | 3 | 7 | 0 | Sales audit → comparison OPQ vs MQ |
| C6 | 3 | 7 | 0 | Plant operators → safety assessments |
| C7 | 4 | 10 | 0 | Bilingual healthcare → legal question refused |
| C8 | 3 | 10 | 0 | Admin assistants → Excel/Word screening |
| C9 | 7 | 47 | 0 | Engineer JD → multi-turn refine |
| C10 | 3 | 10 | 0 | Graduate battery → replace OPQ |

**Total: 38 user turns, 145 recommendations, 0 schema errors, 0 hallucinations.**

### 7. Deployment

The server is deployed as a public API endpoint on Render. Configuration via `render.yaml`:
- Python runtime, `uvicorn` as WSGI server
- `LLM_API_KEY` set as a secret environment variable (Groq API key)
- Embedding model cached at build time for fast cold starts
