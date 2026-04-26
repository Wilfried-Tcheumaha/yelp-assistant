import api.patches.instructor_compat  # noqa: F401 — Superlinked NLQ vs instructor 1.14+ API

from superlinked import framework as sl
import json
import os
from typing import Any
import openai
from api.agents.qdrant_url import resolve_qdrant_url
from api.agents.superlinked_app.index import business_index, business
from api.agents.superlinked_app.query import query
from api.agents.superlinked_app.utils.utils import *
from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field
import instructor
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from api.agents.utils.prompt_management import prompt_template_config


class RAGUsedContext (BaseModel):
    id: str = Field(description="The id of the restaurants used to answer the question")
    description: str = Field(description="A short description of the business")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: list[RAGUsedContext]=Field(description="The context used to answer the question")

qdrant_vdb = sl.QdrantVectorDatabase(
    url=resolve_qdrant_url(),
    # Superlinked's QdrantVectorDatabase currently requires an api_key arg.
    # For local Qdrant this is typically unused, so we default to empty.
    api_key=os.getenv("QDRANT_API_KEY", ""),
)
parser = sl.DataFrameParser(business)

source_qdrant = sl.RestSource(
    business,
    parser=parser,
)

# RestExecutor needs sl.RestQuery (path for /api/v1/search/<query_path> by default).
business_rest_query = sl.RestQuery(
    rest_descriptor=sl.RestDescriptor(query_path="business_search"),
    query_descriptor=query,
)

executor_qdrant = sl.RestExecutor(
    sources=[source_qdrant],
    indices=[business_index],
    vector_database=qdrant_vdb,
    queries=[business_rest_query],
)

_qdrant_app = None


def get_qdrant_app():
    """Lazy init: avoids Qdrant TCP on import (e.g. eval scripts importing rag_pipeline)."""
    global _qdrant_app
    if _qdrant_app is None:
        _qdrant_app = executor_qdrant.run()
    return _qdrant_app

@traceable(
    name="Retrieve_context",
    run_type="embedding",
    metadata={"ls_nlq_provider": "openai", "ls_nlq_model": "gpt-4o-mini","ls_nlq_embedding_provider": "huggingface ", "ls_nlq_embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}
)
def Retrieve_context(question, qdrant_app, k=5):
    qdrant_results = qdrant_app.query(
    query,
    natural_query=question,
    limit=k,
)

    format_minute_columns_to_hhmm(sl.PandasConverter.to_pandas(qdrant_results))
    return qdrant_results

@traceable(
    name="_result_to_restaurants",
    run_type="prompt"
)
def _result_to_restaurants(result) -> list[dict[str, Any]]:
    df_columns = ["business_id", "name", "address", "city", "state", "postal_code", "latitude", "longitude", "stars", "review_count", "is_open", "categories", "attributes", "hours"]
    df = sl.PandasConverter.to_pandas(result).rename(columns={"id": "business_id"})
    # Derive `is_open` robustly.
    # Some payloads may contain only `is_open_i` (0/1) instead of the boolean `is_open`.
    if "is_open" not in df.columns:
        if "is_open_i" in df.columns:
            df["is_open"] = df["is_open_i"].astype(int)
        else:
            df["is_open"] = 0

    df = df.assign(
        categories=df.get("category_tags", df.get("categories_text")),
        is_open=df["is_open"].astype(int),
    )

    # Parse attributes/hours when present.
    for c in ("attributes", "hours"):
        if c in df.columns:
            df[c] = df[c].map(
                lambda v: json.loads(v) if isinstance(v, str) and v.strip()
                else ({} if v in ("", None) else v)
            )
        else:
            df[c] = {}
    return df.reindex(columns=df_columns).to_dict(orient="records")

@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, question):

    template = prompt_template_config('api/agents/prompts/retrieval_generation.yaml', 'retrieval_generation')
    rendered_prompt= template.render(preprocessed_context=preprocessed_context, question=question)
    return rendered_prompt

@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model": "gpt-4.1-mini"}
)
def generate_answer(prompt):
    client = instructor.from_openai(openai.OpenAI())
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role":"system", "content": prompt}],
        temperature=0,
        response_model=RAGGenerationResponse
    )
    current_run = get_current_run_tree()
    u = getattr(raw_response, "usage", None)
    if current_run is not None and u is not None:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None) or 0,
            "output_tokens": getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None) or 0,
            "total_tokens": getattr(u, "total_tokens", None) or 0,
        }
    return response

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, qdrant_app=None):
    app = qdrant_app if qdrant_app is not None else get_qdrant_app()
    context = Retrieve_context(question, app)
    preprocessed_context=_result_to_restaurants(context)
    prompt=build_prompt(preprocessed_context, question)
    answer=generate_answer(prompt)

    return {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": [e.id for e in context.entries],
        "retrieved_restaurant_names": [e.fields.get("name") for e in context.entries],
        "similarity_score": [e.metadata.score for e in context.entries],
    }

def rag_pipeline_wrapper(question, top_k=5):
   

    app = get_qdrant_app()
    result = rag_pipeline(question, app)

    # Superlinked stores the Yelp id under `__object_id__` (the actual Qdrant point id
    # is a derived UUID), so we filter on the payload field rather than retrieve(ids=...).
    raw_client = QdrantClient(
        url=resolve_qdrant_url(),
        api_key=os.getenv("QDRANT_API_KEY", ""),
    )
    collection = os.getenv("QDRANT_COLLECTION", "yelp-businesses-collection-00")

    def _maybe_json(v):
        if isinstance(v, str) and v.strip():
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    used_context = []
    for item in result.get("references", []):
        points, _ = raw_client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="__object_id__", match=MatchValue(value=item.id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        payload = points[0].payload if points else {}

        used_context.append({
            "id": item.id,
            "description": item.description,
            "name": payload.get("__schema_field__Business_name"),
            "address": payload.get("__schema_field__Business_address"),
            "latitude": payload.get("__schema_field__Business_latitude"),
            "longitude": payload.get("__schema_field__Business_longitude"),
            "stars": payload.get("__schema_field__Business_stars"),
            "reviews": payload.get("__schema_field__Business_review_count"),
            "categories": payload.get("__schema_field__Business_category_tags"),
            "attributes": _maybe_json(payload.get("__schema_field__Business_attributes")),
            "hours": _maybe_json(payload.get("__schema_field__Business_hours")),
        })

    return {
        "answer": result["answer"],
        "used_context": used_context,
    }