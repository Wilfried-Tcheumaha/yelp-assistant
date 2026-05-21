# Yelp Assistant (Agentic RAG with LangGraph + Superlinked + Qdrant)

This project implements a product-inspired Yelp assistant that answers user questions through an **agentic RAG workflow**:
- an **intent router** filters off-topic questions before any retrieval happens,
- a **tool-calling QA agent** decides when (and how) to search the business catalog and the reviews corpus,
- two retrieval tools — a **hybrid structured search** over Yelp business records (semantic + numeric + hard filters) and a **review-text search** scoped to a set of business ids — back the agent,
- and the agent generates the final response with **OpenAI**, citing the businesses it actually used.

The whole graph is orchestrated with **LangGraph** and persisted across turns with a **Postgres** checkpointer, enabling **multi-turn conversations**. A **LangSmith** feedback endpoint records thumbs/comment feedback against the trace id of each answer.

At runtime, the assistant is exposed as a small **FastAPI** service. The project also ships a **Streamlit chat UI** for interactive usage, and the two retrieval tools are additionally packaged as **MCP servers** (HTTP) for reuse by external agents.

Yelp dataset: https://business.yelp.com/data/resources/open-dataset/

## What works (implemented)

### Agentic workflow (LangGraph)
The graph is defined in `api/src/api/agents/graph.py` and has three nodes plus conditional routing:

```
START
  └─> intent_router_node
        ├─ question_relevant=False ─> END
        └─ question_relevant=True  ─> agent_node
                                       ├─ final_answer=True or iteration>2 ─> END
                                       ├─ tool_calls present ─> tool_node ─> agent_node (loop)
                                       └─ no tool calls ─> END
```

- **`intent_router_node`** (`agents/agents.py`): a small `gpt-4.1-mini` call (via `instructor`) returning `IntentRouterResponse(question_relevant: bool, answer: str)`. Off-topic questions short-circuit straight to `END` with a polite refusal, so we never spend tokens on retrieval/generation for irrelevant queries. The router run also captures the LangSmith **trace id** that the rest of the pipeline will be grouped under, and propagates it on the state for downstream feedback.
- **`agent_node`** (`agents/agents.py`): the main QA agent. It reads the available tool descriptions from state, runs `gpt-4.1-mini` with `instructor` against the `AgentResponse` schema, and emits any combination of `tool_calls`, an `answer`, structured `references`, and a `final_answer` boolean.
- **`tool_node`**: a LangGraph `ToolNode` wired with the two retrieval tools — `get_formatted_context` (businesses) and `get_formatted_reviews_context` (reviews scoped to business ids).
- **`tool_router`**: stops the loop when `final_answer=True`, when `iteration > 2` (safety cap), or when there are no pending tool calls.

State is a Pydantic `State` model with reducer-merged `messages` and `references` (`Annotated[..., add]`), plus `iteration`, `final_answer`, `available_tools`, and `trace_id`.

### Multi-turn conversations (Postgres checkpointer)
- Every request carries a `thread_id` (generated client-side, e.g. by Streamlit per session).
- `agent_execution` opens a `PostgresSaver.from_conn_string(...)` and compiles the graph with that checkpointer, so each turn resumes the prior graph state for the same `thread_id`.
- The Postgres service is part of `docker-compose.yml` (`langgraph_user` / `langgraph_db`). Its data is **volume-mounted** at `./postgres_data` but **not tracked in git** (see `.gitignore`).

### Structured + semantic business retrieval (Superlinked + Qdrant)
The retrieval tool `get_formatted_context(query: str, top_k: int = 5) -> str` lives in `api/src/api/agents/tools.py`:
- A Superlinked **`Business` schema** defines the available business fields (name/address/location, stars/review_count, amenities flags, category tags, opening hours).
- A Superlinked **index** combines:
  - `TextSimilaritySpace` using `sentence-transformers/all-MiniLM-L6-v2` for category semantic matching,
  - `NumberSpace` for `review_count` and `stars`,
  - and stores all business metadata in Qdrant.
- A Superlinked **natural-language query interface** uses OpenAI to convert the user query into structured query parameters (city, rating ranges, open/closed constraints, amenity flags, time-of-day open/close filters, etc.).
- Retrieval runs through a Superlinked `RestExecutor` with a `RestSource` over the `Business` schema, a `RestQuery` targeting `business_search`, and a `QdrantVectorDatabase` pointing at `http://qdrant:6333`.
- The Qdrant app is **lazy-initialized** (`get_qdrant_app`) so we don't open a Qdrant connection at import time.
- The tool returns the top-k results as a compact, formatted string of `id / name / rating / review_count / state / city / similarity_score` lines — cheap to feed back into the agent's next step.

### Review-text retrieval (Qdrant + RRF fusion)
The retrieval tool `get_formatted_reviews_context(query: str, business_ids: list[str], k: int = 15) -> str`:
- Embeds the query with OpenAI `text-embedding-3-small`.
- Issues a Qdrant **`query_points`** call against `yelp-reviews-collection-00` with:
  - a `Prefetch` filtered by `business_id ∈ business_ids` (limited to 20), and
  - a `FusionQuery(fusion="rrf")` final stage that returns the top-`k` review chunks.
- This means the agent typically calls `get_formatted_context` first to narrow down candidate businesses, then `get_formatted_reviews_context` to pull on-topic review excerpts for those exact ids.

### Response generation (OpenAI + Instructor, structured)
- The agent uses [`instructor`](https://github.com/jxnl/instructor) to get a **structured** completion typed as `AgentResponse`:
  - `answer: str` — free-text response,
  - `references: list[RAGUsedContext]` — typed list of business ids + short descriptions actually used,
  - `final_answer: bool` — whether the agent is done,
  - `tool_calls: list[ToolCall]` — any tools the agent wants to invoke next.
- After the graph returns, `yelp_agent_wrapper` re-hydrates each cited reference by querying Qdrant directly via `QdrantClient.scroll` on `__object_id__`, returning a `used_context` list with `name`, `address`, `latitude`, `longitude`, `stars`, `reviews`, `categories`, `attributes`, and `hours`. This is what powers the UI cards + map.
- The wrapper also returns the `trace_id` so the UI can attach feedback to the exact LangSmith run.

### MCP servers (optional reuse path)
Both retrieval tools are *also* packaged as standalone **MCP servers** under `restaurants_mcp_server/` and `reviews_mcp_server/`:
- `restaurants_mcp_server` exposes `get_formatted_context` over `fastmcp` (HTTP transport, port 8001).
- `reviews_mcp_server` exposes `get_formatted_reviews_context` (HTTP transport, port 8002).
- The MCP services are **decoupled from the `api` package**: `restaurants_mcp_server` carries its own forked copy of the Superlinked schema/index/query under `restaurants_mcp_server/src/restaurants_mcp_server/superlinked_app/`; `reviews_mcp_server` only depends on `qdrant-client` + `openai` and is intentionally tiny.
- The in-process API does not call the MCP services today — it still imports the tools directly from `api.agents.tools`. The MCP services exist so external agents (Claude Desktop, another LangGraph app, etc.) can call the same retrieval logic over MCP.

### Streamlit UI (`chatbot_ui/`)
- A persistent `session_id` (UUID) is created per browser session and sent as `thread_id` on every request, enabling multi-turn memory through the Postgres checkpointer.
- Chat column on the left; **right-side column shows a pydeck map** with numbered red pins for every suggested restaurant that has valid coordinates.
- Sidebar renders a **Yelp-style business card** per suggestion:
  - orange star chips for the rating, review count,
  - live **Open / Closed** status computed from `hours` (handles overnight ranges and next-open time),
  - category tags,
  - clickable address that deep-links to `https://www.yelp.com/search?find_desc=<name>&find_loc=<address>`.
- **Thumbs feedback** under each assistant turn (`st.feedback("thumbs", ...)` driven by an `on_change` callback). Negative feedback opens an optional comment box. The UI POSTs to `/feedback/` with the `trace_id` captured from the last `/rag/` response.
- UI rendering helpers live in `chatbot_ui/src/chatbot_ui/utils/` (`business_card.py`, `restaurants_map.py`).

### Observability & feedback (LangSmith)
- The API uses [LangSmith](https://smith.langchain.com/) via the `langsmith` SDK (`@traceable` on the intent router, the retrieval steps, embedding, and the review search).
- Each `POST /rag/` request can produce a trace tree like:
  - `agent_execution` → `intent_router_node` → `agent_node` → `retriever_top_n` → `format_retrieved_context` → `agent_node` → `retrieve_reviews_data` → `embed_query` → `process_reviews_context` → `agent_node` (final)
- The router records the trace id on the graph state, the API surfaces it in the `RAGResponse`, and the UI uses it to attach thumbs/comment feedback via `POST /feedback/` → `langsmith.Client.create_feedback`.
- Enable tracing by setting the standard LangSmith environment variables (see `env.example`):
  - `LANGSMITH_TRACING=true`
  - `LANGSMITH_API_KEY` (from your LangSmith account)
  - `LANGSMITH_PROJECT` (project name in LangSmith)
  - `LANGSMITH_ENDPOINT` (optional; defaults to the public LangSmith API)
- Docker Compose loads `.env` into each service, so the same variables apply in containers.

## API

### Endpoints
- `POST /rag/` — run the agentic RAG workflow for one user turn.
- `POST /feedback/` — record thumbs (`feedback_score`: 0 or 1) and/or free-text feedback against a previous answer's `trace_id`.

### `POST /rag/`

Request:
```json
{
  "query": "Find Italian restaurants open at 7pm with outdoor seating in Paris",
  "thread_id": "session-uuid-string"
}
```

`thread_id` identifies the conversation. Reusing it across requests gives you a multi-turn conversation (state restored from the Postgres checkpointer); a fresh `thread_id` starts a new conversation.

Response:
```json
{
  "request_id": "uuid-string",
  "trace_id": "langsmith-run-uuid",
  "answer": "assistant response text",
  "used_context": [
    {
      "id": "business_id",
      "description": "short description of the restaurant",
      "name": "Joe's Pizza",
      "address": "123 Main St, Paris",
      "latitude": 48.8566,
      "longitude": 2.3522,
      "stars": 4.5,
      "reviews": 312,
      "categories": ["Pizza", "Italian"],
      "attributes": { "OutdoorSeating": true },
      "hours": { "Monday": "11:0-22:0" }
    }
  ]
}
```

### `POST /feedback/`

Request:
```json
{
  "trace_id": "langsmith-run-uuid",
  "feedback_score": 1,
  "feedback_text": "optional free-text comment",
  "feedback_source_type": "api",
  "thread_id": "session-uuid-string"
}
```

`feedback_score` is `1` (thumbs up), `0` (thumbs down), or `null`. `feedback_text` may be empty. The endpoint forwards both signals to LangSmith as separate feedback records (`key="thumbs"` and `key="comment"`) on the given run, and flushes the LangSmith client before returning.

Response:
```json
{
  "request_id": "uuid-string",
  "status": "Feedback sent successfully"
}
```

Notes:
- The retrieval tools default to `top_k=5` (businesses) and `k=15` (reviews); the agent can request different values per call.
- The graph caps the agent loop at `iteration > 2` to avoid runaway tool use.
- `used_context` only includes the businesses the LLM actually cited (via the `instructor`-typed `references` field), re-hydrated with full Qdrant payloads.
- Off-topic questions are answered directly by the intent router and return an empty `used_context`.

## Docker / local run

This repo includes `docker-compose.yml` with:
- `qdrant`: Qdrant vector database (`./qdrant_storage` volume)
- `postgres`: Postgres 16 for the LangGraph checkpointer (`./postgres_data` volume, gitignored)
- `api`: the FastAPI service — port `8000`
- `streamlit-app`: the chat UI service — port `8501`
- `restaurants_mcp_server`: MCP server exposing the businesses tool — port `8001`
- `reviews_mcp_server`: MCP server exposing the reviews tool — port `8002`

1. Create your `.env` from `env.example` (OpenAI key, optional LangSmith vars, etc.).
2. Start services:
   - `make run-docker-compose`
3. Open:
   - UI: `http://localhost:8501`
   - API: `http://localhost:8000` (Swagger at `/docs`)
   - Restaurants MCP: `http://localhost:8001`
   - Reviews MCP: `http://localhost:8002`
4. Optional direct API call:
   - `POST http://localhost:8000/rag/` with `{"query": "...", "thread_id": "..."}`

### Model download/cache
- Superlinked downloads `sentence-transformers/all-MiniLM-L6-v2` on first container startup (and then reuses the cached files).
- Docker images set the cache to writable locations (e.g. under `/tmp`) for non-root execution.

### CPU-only PyTorch
- Superlinked transitively requires `torch` + `torchvision`. To keep Linux images small (and avoid pulling the multi-GB CUDA stack into containers that don't need it), the root `pyproject.toml` pins both packages to PyTorch's CPU index via `[tool.uv.sources]` + `[[tool.uv.index]]`:
  ```toml
  [tool.uv.sources]
  torch        = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
  torchvision  = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url  = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```
- On Linux the lockfile resolves `torch==X.Y.Z+cpu` and `torchvision==X.Y.Z+cpu`. macOS still gets the regular PyPI builds. If you add `torchaudio` later, give it the same `[tool.uv.sources]` entry — torchvision/torchaudio register C++ ops at import time against a specific torch ABI, and a mismatch crashes with `RuntimeError: operator torchvision::nms does not exist`.

## Repository layout (selected)

```
api/src/api/
  agents/
    graph.py                # LangGraph workflow + State + agent_execution + yelp_agent_wrapper
    agents.py               # intent_router_node, agent_node, structured response models
    tools.py                # get_formatted_context + get_formatted_reviews_context + tracing
    prompts/
      intent_router_agent.yaml
      qa_agent.yaml
    superlinked_app/        # Business schema, index, NL query definition
    utils/                  # tool descriptions, prompt management, formatting helpers
    qdrant_url.py
  api/
    endpoints.py            # POST /rag/, POST /feedback/
    middleware.py           # RequestIDMiddleware
    models.py               # RAGRequest/Response, FeedbackRequest/Response
    processors/
      submit_feedback.py    # LangSmith feedback writer (thumbs + comment)
  app.py                    # FastAPI app, CORS, router registration
  core/config.py
  patches/instructor_compat.py

chatbot_ui/src/chatbot_ui/
  app.py                    # Streamlit chat + map + sidebar suggestions + thumbs feedback
  utils/                    # business_card.py, restaurants_map.py, css

restaurants_mcp_server/src/restaurants_mcp_server/
  main.py                   # FastMCP service exposing get_formatted_context
  utils.py                  # Superlinked retrieval (forked from api/agents/tools.py)
  superlinked_app/          # forked Business schema/index/query (no api/ dependency)
  qdrant_url.py
  instructor_compat.py
  core/config.py

reviews_mcp_server/src/reviews_mcp_server/
  main.py                   # FastMCP service exposing get_formatted_reviews_context
  utils.py                  # Qdrant RRF query + OpenAI embedding (no Superlinked needed)
  qdrant_url.py
  core/config.py

notebooks/
  09-Query-Rewriting.ipynb
  10-Router.ipynb
  11-Single-turn-agent.ipynb
  12-Multiturn-Agent.ipynb
  13-Multiple-Tools.ipynb
  14-Human-Feedback.ipynb
  15-mcp.ipynb

docker-compose.yml          # qdrant + postgres + api + streamlit-app + restaurants_mcp + reviews_mcp
```

## Dataset files (notebooks input)

The notebooks expect the Yelp Open Dataset JSON files to be placed under `data/raw/` (relative to the notebook folder).

Common raw inputs used in `notebooks/01-explore-yelp-data.ipynb`:
- `data/raw/yelp_academic_dataset_business.json`
- `data/raw/yelp_academic_dataset_review.json`
- `data/raw/yelp_academic_dataset_checkin.json`
- `data/raw/yelp_academic_dataset_tip.json`
- `data/raw/yelp_academic_dataset_user.json`

The RAG pipeline notebooks use a preprocessed restaurant sample with hours, e.g.:
- `data/raw/yelp_academic_dataset_business_restaurants_with_hours_sample_1000.json`

To run the serving API, you need to have the Qdrant collections populated (created/ingested from the notebooks). Two collections are used:
- `yelp-businesses-collection-00` — written by Superlinked (named vectors per Space + business payload).
- `yelp-reviews-collection-00` — flat vectors + `business_id` / `text` payload, used by the reviews tool's RRF query.

## Roadmap (Next)
- Photo embeddings / visual retrieval
- Real-time website search
- Review-text sentiment retrieval
- Explicit "citations" attached to reviews/photos
- Recommendations
- Turn the solution into a Voice agent
- Deployment
- Wire the API to call its retrieval tools through the MCP servers (instead of in-process imports), so the MCP transport is exercised by the assistant itself.
