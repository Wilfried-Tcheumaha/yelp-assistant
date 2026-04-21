# Yelp Assistant (Structured RAG with Superlinked + Qdrant)

This project implements a product-inspired Yelp assistant that answers user questions by:
- orchestrating a **hybrid structured search** over Yelp business records (semantic similarity + numeric similarity + hard filters),
- retrieving grounded business fields from **Qdrant**,
- and generating a final response with **OpenAI** using the retrieved records as the input.

At runtime, the assistant is exposed as a small **FastAPI** service.
The project also includes a **Streamlit chat UI** for interactive usage.

Yelp dataset: https://business.yelp.com/data/resources/open-dataset/

## What works (implemented)

### Structured + semantic retrieval (Superlinked)
- A Superlinked **schema** (`Business`) defines the available business fields (name/address/location, stars/review_count, amenities flags, category tags, and opening hours).
- A Superlinked **index** combines:
  - `TextSimilaritySpace` using `sentence-transformers/all-MiniLM-L6-v2` for category semantic matching,
  - `NumberSpace` for `review_count` and `stars`,
  - and returns business metadata stored in Qdrant.
- A Superlinked **natural-language interface** uses OpenAI to convert the user question into structured query parameters (e.g. city, rating ranges, open/closed constraints, amenity flags, and time-of-day open/close filters).

### Qdrant-backed context
- Retrieval uses a Superlinked `RestExecutor` with:
  - `RestSource` over the `Business` schema,
  - `RestQuery` targeting `business_search`,
  - and `QdrantVectorDatabase` pointing at `http://qdrant:6333`.
- The API transforms the Qdrant result payload into a list of business dictionaries (parses JSON strings for `attributes` and `hours`).

### Response generation (OpenAI + Instructor)
- The assistant builds a prompt from the retrieved businesses and calls OpenAI via [`instructor`](https://github.com/jxnl/instructor) to get a **structured** response:
  - `instructor.from_openai(openai.OpenAI()).chat.completions.create_with_completion(model="gpt-4.1-mini", response_model=RAGGenerationResponse, ...)`
  - `RAGGenerationResponse` returns both the free-text `answer` and a typed list of `references` (business id + short description) actually used in the answer.
- After generation, `rag_pipeline_wrapper` re-hydrates each reference by fetching the full Qdrant payload (via `__object_id__`) and returns a `used_context` list containing `name`, `address`, `latitude`, `longitude`, `stars`, `reviews`, `categories`, `attributes`, and `hours`. This is what powers the UI cards + map.

### Streamlit UI (`chatbot_ui/`)
- Chat column on the left; **right-side column shows a pydeck map** with numbered red pins for every suggested restaurant that has valid coordinates.
- Sidebar renders a **Yelp-style business card** per suggestion:
  - orange star chips for the rating, review count,
  - live **Open / Closed** status computed from `hours` (handles overnight ranges and next-open time),
  - category tags,
  - clickable address that deep-links to `https://www.yelp.com/search?find_desc=<name>&find_loc=<address>`.
- UI rendering helpers live in `chatbot_ui/src/chatbot_ui/utils/` (`business_card.py`, `restaurants_map.py`).

### Observability (LangSmith)
- The API uses [LangSmith](https://smith.langchain.com/) via the `langsmith` SDK (`@traceable` on the RAG steps in `api/src/api/agents/retrieval_generation.py`).
- Each `POST /rag/` request can produce a trace tree such as:
  - `rag_pipeline` → `Retrieve_context` → `_result_to_restaurants` → `build_prompt` → `generate_answer`
- When OpenAI returns usage on the completion, the **`generate_answer`** run records **`usage_metadata`** on the LangSmith run (`input_tokens`, `output_tokens`, `total_tokens`).
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
}
```

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
- Retrieval currently uses a fixed `k=5` limit (see `Retrieve_context` in `api/src/api/agents/retrieval_generation.py`).
- `used_context` only includes the businesses the LLM actually cited (via the `instructor`-typed `references` field), re-hydrated with full Qdrant payloads.

## Docker / local run

This repo includes `docker-compose.yml` with:
- `qdrant`: Qdrant vector database
- `api`: the FastAPI service
- `streamlit-app`: the chat UI service

1. Create your `.env` from `env.example` (OpenAI key, optional LangSmith vars, etc.).
2. Start services:
   - `make run-docker-compose:`
3. Open:
   - UI: `http://localhost:8501`
   - API: `http://localhost:8000`
4. Optional direct API call:
   - `POST http://localhost:8000/rag/`

### Model download/cache
- Superlinked downloads `sentence-transformers/all-MiniLM-L6-v2` on first container startup (and then reuses the cached files).
- The Docker image sets the cache to writable locations (e.g. under `/tmp`) for non-root execution.


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

To run the serving API, need to have the Qdrant collections populated (created/ingested from the notebooks).

## Roadmap (Next)
- Photo embeddings / visual retrieval
- real-time website search
- review-text sentiment retrieval
- explicit "citations" attached to reviews/photos
- Multiturn conversations
- Recommendations
- Turn the solution into Voice agent
- Deployment
