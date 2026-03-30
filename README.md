# Yelp Assistant (Structured RAG with Superlinked + Qdrant)

This project implements a product-inspired Yelp assistant that answers user questions by:
- orchestrating a **hybrid structured search** over Yelp business records (semantic similarity + numeric similarity + hard filters),
- retrieving grounded business fields from **Qdrant**,
- and generating a final response with **OpenAI** using the retrieved records as the input.

At runtime, the assistant is exposed as a small **FastAPI** service.

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

### Response generation (OpenAI)
- The assistant builds a prompt from the retrieved businesses and calls:
  - `openai.chat.completions.create(model="gpt-5-nano", ...)`
- The model is instructed to answer based on the provided retrieved business records.

## API

### Endpoint
- `POST /rag/`

### Request body
```json
{
  "query": "Find Italian restaurants open at 7pm with outdoor seating in Paris",
  "top_k": 5
}
```

### Response body
```json
{
  "request_id": "uuid-string",
  "answer": "assistant response text"
}
```

Notes:
- `top_k` exists in the request model, but the current retrieval implementation uses a fixed limit (see `api/src/api/agents/retrieval_generation.py`).
- The response is grounded in retrieved **business fields**; the current code does not attach explicit citations.

## Docker / local run

This repo includes `docker-compose.yml` with:
- `qdrant`: Qdrant vector database
- `api`: the FastAPI service

1. Create your `.env` (see “Configuration” below).
2. Start services:
   - `make run-docker-compose:`
3. Call:
   - `POST http://localhost:8000/rag/`

### Model download/cache
- Superlinked downloads `sentence-transformers/all-MiniLM-L6-v2` on first container startup (and then reuses the cached files).
- The Docker image sets the cache to writable locations (e.g. under `/tmp`) for non-root execution.



## Roadmap (Next)
- Photo embeddings / visual retrieval
- real-time website search
- review-text sentiment retrieval
- explicit “citations” attached to reviews/photos 
- Multiturn conversations
- Recommendations
- Turn the solution into Voice agent
- Deployment
