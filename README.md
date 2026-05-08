# Yelp Assistant (Agentic RAG with LangGraph + Superlinked + Qdrant)

This project implements a product-inspired Yelp assistant that answers user questions through an **agentic RAG workflow**:
- an **intent router** filters off-topic questions before any retrieval happens,
- a **tool-calling QA agent** decides when (and how) to search the business catalog,
- the retrieval tool runs a **hybrid structured search** over Yelp business records (semantic similarity + numeric similarity + hard filters) backed by **Superlinked** + **Qdrant**,
- and the agent generates the final response with **OpenAI**, citing the businesses it actually used.

The whole graph is orchestrated with **LangGraph** and persisted across turns with a **Postgres** checkpointer, enabling **multi-turn conversations**.

At runtime, the assistant is exposed as a small **FastAPI** service.
The project also includes a **Streamlit chat UI** for interactive usage.

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

- **`intent_router_node`** (`agents/agents.py`): a small `gpt-4.1-mini` call (via `instructor`) that returns `IntentRouterResponse(question_relevant: bool, answer: str)`. Off-topic questions short-circuit straight to `END` with a polite refusal, so we never spend tokens on retrieval/generation for irrelevant queries.
- **`agent_node`** (`agents/agents.py`): the main QA agent. It reads the available tool descriptions from state, runs `gpt-4.1-mini` with `instructor` against the `AgentResponse` schema, and emits any combination of `tool_calls`, an `answer`, structured `references`, and a `final_answer` boolean.
- **`tool_node`**: a LangGraph `ToolNode` wired with the `get_formatted_context` tool, which is the structured retriever (see below).
- **`tool_router`**: stops the loop when `final_answer=True`, when `iteration > 2` (safety cap), or when there are no pending tool calls.

State is a Pydantic `State` model with reducer-merged `messages` and `references` (`Annotated[..., add]`), plus `iteration`, `final_answer`, and `available_tools`.

### Multi-turn conversations (Postgres checkpointer)
- Every request carries a `thread_id` (generated client-side, e.g. by Streamlit per session).
- `agent_execution` opens a `PostgresSaver.from_conn_string(...)` and compiles the graph with that checkpointer, so each turn resumes the prior graph state for the same `thread_id`.
- The Postgres service is part of `docker-compose.yml` (`langgraph_user` / `langgraph_db`), and its data lives under `./postgres_data`.

### Structured + semantic retrieval (Superlinked + Qdrant)
The retrieval tool (`api/src/api/agents/tools.py`) is exposed to the agent as `get_formatted_context(query: str, top_k: int = 5) -> str`:
- A Superlinked **schema** (`Business`) defines the available business fields (name/address/location, stars/review_count, amenities flags, category tags, opening hours).
- A Superlinked **index** combines:
  - `TextSimilaritySpace` using `sentence-transformers/all-MiniLM-L6-v2` for category semantic matching,
  - `NumberSpace` for `review_count` and `stars`,
  - and returns business metadata stored in Qdrant.
- A Superlinked **natural-language query interface** uses OpenAI to convert the (possibly rewritten) user query into structured query parameters (city, rating ranges, open/closed constraints, amenity flags, time-of-day open/close filters, etc.).
- Retrieval runs through a Superlinked `RestExecutor` with:
  - `RestSource` over the `Business` schema,
  - `RestQuery` targeting `business_search`,
  - and `QdrantVectorDatabase` pointing at `http://qdrant:6333`.
- The Qdrant app is **lazy-initialized** (`get_qdrant_app`) so we don't open a Qdrant connection at import time.
- The tool returns the top-k results as a compact, formatted string of `id / name / rating / review_count / state / city / similarity_score` lines — designed to be cheap to feed back into the agent's next step.

### Response generation (OpenAI + Instructor, structured)
- The agent uses [`instructor`](https://github.com/jxnl/instructor) to get a **structured** completion typed as `AgentResponse`:
  - `answer: str` — free-text response,
  - `references: list[RAGUsedContext]` — typed list of business ids + short descriptions actually used,
  - `final_answer: bool` — whether the agent is done,
  - `tool_calls: list[ToolCall]` — any tools the agent wants to invoke next.
- After the graph returns, `yelp_agent_wrapper` re-hydrates each cited reference by querying Qdrant directly via `QdrantClient.scroll` on `__object_id__`, returning a `used_context` list with `name`, `address`, `latitude`, `longitude`, `stars`, `reviews`, `categories`, `attributes`, and `hours`. This is what powers the UI cards + map.

### Streamlit UI (`chatbot_ui/`)
- A persistent `session_id` (UUID) is created per browser session and sent as `thread_id` on every request, enabling multi-turn memory through the Postgres checkpointer.
- Chat column on the left; **right-side column shows a pydeck map** with numbered red pins for every suggested restaurant that has valid coordinates.
- Sidebar renders a **Yelp-style business card** per suggestion:
  - orange star chips for the rating, review count,
  - live **Open / Closed** status computed from `hours` (handles overnight ranges and next-open time),
  - category tags,
  - clickable address that deep-links to `https://www.yelp.com/search?find_desc=<name>&find_loc=<address>`.
- UI rendering helpers live in `chatbot_ui/src/chatbot_ui/utils/` (`business_card.py`, `restaurants_map.py`).

### Observability (LangSmith)
- The API uses [LangSmith](https://smith.langchain.com/) via the `langsmith` SDK (`@traceable` on the intent router and the retrieval steps).
- Each `POST /rag/` request can produce a trace tree such as:
  - `agent_execution` → `intent_router_node` → `agent_node` → `retriever_top_n` → `format_retrieved_context` → `agent_node` (final)
- Enable tracing by setting the standard LangSmith environment variables (see `env.example`):
  - `LANGSMITH_TRACING=true`
  - `LANGSMITH_API_KEY` (from your LangSmith account)
  - `LANGSMITH_PROJECT` (project name in LangSmith)
  - `LANGSMITH_ENDPOINT` (optional; defaults to the public LangSmith API)
- Docker Compose loads `.env` into the `api` service, so the same variables apply in containers.

## API

### Endpoint
- `POST /rag/`

### Request body
```json
{
  "query": "Find Italian restaurants open at 7pm with outdoor seating in Paris",
  "thread_id": "session-uuid-string"
}
```

`thread_id` identifies the conversation. Reusing the same `thread_id` across requests gives you a multi-turn conversation (state restored from the Postgres checkpointer); using a new `thread_id` starts a fresh conversation.

### Response body
```json
{
  "request_id": "uuid-string",
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

Notes:
- The retrieval tool defaults to `top_k=5` but the agent can request a different `top_k` per call.
- The graph caps the agent loop at `iteration > 2` to avoid runaway tool use.
- `used_context` only includes the businesses the LLM actually cited (via the `instructor`-typed `references` field), re-hydrated with full Qdrant payloads.
- Off-topic questions are answered directly by the intent router and return an empty `used_context`.

## Docker / local run

This repo includes `docker-compose.yml` with:
- `qdrant`: Qdrant vector database (`./qdrant_storage` volume)
- `postgres`: Postgres 16 for the LangGraph checkpointer (`./postgres_data` volume)
- `api`: the FastAPI service
- `streamlit-app`: the chat UI service

1. Create your `.env` from `env.example` (OpenAI key, optional LangSmith vars, etc.).
2. Start services:
   - `make run-docker-compose`
3. Open:
   - UI: `http://localhost:8501`
   - API: `http://localhost:8000`
4. Optional direct API call:
   - `POST http://localhost:8000/rag/` with `{"query": "...", "thread_id": "..."}`

### Model download/cache
- Superlinked downloads `sentence-transformers/all-MiniLM-L6-v2` on first container startup (and then reuses the cached files).
- The Docker image sets the cache to writable locations (e.g. under `/tmp`) for non-root execution.

## Repository layout (selected)

```
api/src/api/
  agents/
    graph.py              # LangGraph workflow + State + agent_execution + yelp_agent_wrapper
    agents.py             # intent_router_node, agent_node, structured response models
    tools.py              # get_formatted_context (Superlinked + Qdrant retrieval tool)
    prompts/
      intent_router_agent.yaml
      qa_agent.yaml
    superlinked_app/      # Business schema, index, NL query definition
    utils/                # tool descriptions, prompt management, formatting helpers
  api/
    endpoints.py          # POST /rag/
    models.py             # RAGRequest, RAGResponse, RAGUsedContext
chatbot_ui/src/chatbot_ui/
  app.py                  # Streamlit chat + map + sidebar suggestions
  utils/                  # business_card.py, restaurants_map.py, css
notebooks/
  09-Query-Rewriting.ipynb
  10-Router.ipynb
  11-Single-turn-agent.ipynb
  12-Multiturn-Agent.ipynb
docker-compose.yml        # qdrant + api + streamlit-app + postgres
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

To run the serving API, you need to have the Qdrant collections populated (created/ingested from the notebooks).

## Roadmap (Next)
- Photo embeddings / visual retrieval
- Real-time website search
- Review-text sentiment retrieval
- Explicit "citations" attached to reviews/photos
- Recommendations
- Turn the solution into a Voice agent
- Deployment
