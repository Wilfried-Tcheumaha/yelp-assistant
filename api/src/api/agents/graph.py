import api.patches.instructor_compat  # noqa: F401 — Superlinked NLQ vs instructor 1.14+ API

from superlinked import framework as sl
import json
import os
from typing import Any
from api.agents.qdrant_url import resolve_qdrant_url
from api.agents.superlinked_app.index import business_index, business
from api.agents.superlinked_app.query import query
from api.agents.superlinked_app.utils.utils import *
from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from typing import Annotated, List, Dict
from pydantic import BaseModel, Field
from api.agents.agents import ToolCall, RAGUsedContext
from langgraph.graph import StateGraph, START, END
from api.agents.tools import (
    get_formatted_context,
    get_formatted_reviews_context,
    get_photos_for_businesses,
    get_top_review_for_businesses,
)
import concurrent.futures
import contextvars
from api.agents.utils.utils import get_tool_descriptions
from api.agents.agents import agent_node, intent_router_node
from operator import add
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []
    trace_id: str = ""

def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def intent_router_conditional_edges(state: State):

    if state.question_relevant:
        return "agent_node"
    else:
        return "end"

 ### Workflow
workflow = StateGraph(State)

tools = [get_formatted_context, get_formatted_reviews_context]
tool_node = ToolNode(tools)
tool_descriptions = get_tool_descriptions(tools)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END
    }
)

workflow.add_edge("tool_node", "agent_node")

# graph = workflow.compile()

### Agent Execution

def rag_agent_stream_wrapper(question, thread_id: str):
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

    def _string_for_sse(message:str):
        return f"data: {message}\n\n"

    def process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _is_node_end(chunk):
            return chunk[0] == "updates"

        def _tool_to_text(tool_call):
            if tool_call.name == "get_formatted_context":
                return f"Looking for items: {tool_call.arguments.get('query', '')}."
            elif tool_call.name == "get_formatted_reviews_context":
                return f"Fetching user reviews..."
            else:
                return f"Unknown tool call: {tool_call.name}"

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return ("Analysing the question...")
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return ("Planning...")
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).tool_calls])
                return (message)
        else:   
            return False

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "available_tools": tool_descriptions,
        "iteration": 0
    }

    config = {"configurable": {"thread_id": thread_id}}


    with PostgresSaver.from_conn_string(
"postgresql://langgraph_user:postgres_password@postgres:5432/langgraph_db") as checkpointer:
        graph=workflow.compile(checkpointer=checkpointer)
        for chunk in graph.stream(initial_state, config, stream_mode=["debug","values"]):
            processed_chunk = process_graph_event(chunk)
            if processed_chunk:
                yield _string_for_sse(processed_chunk)
            
            if chunk[0] == "values":
                result = chunk[1]
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
            "photos": [],
            "top_review": "",
        })

    # Soft enrichment: best photos + most-relevant review per cited business.
    # Run in parallel so the user-visible delay is max(photos, reviews), not their sum.
    if used_context:
        yield _string_for_sse("Pulling photos and reviews...")
        biz_ids = [entry["id"] for entry in used_context]
        photos_by_business: dict = {}
        review_by_business: dict = {}
        # Snapshot the caller's contextvars so each worker thread inherits
        # LangSmith's current-run pointer — otherwise the @traceable retriever
        # spans below run in fresh threads with empty context and either
        # orphan or get dropped from the trace tree.
        ctx_photos = contextvars.copy_context()
        ctx_reviews = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_photos = ex.submit(
                ctx_photos.run,
                lambda: get_photos_for_businesses(
                    query=question,
                    business_ids=biz_ids,
                    photos_per_business=3,
                ),
            )
            f_reviews = ex.submit(
                ctx_reviews.run,
                lambda: get_top_review_for_businesses(
                    query=question,
                    business_ids=biz_ids,
                ),
            )
            try:
                photos_by_business = f_photos.result() or {}
            except Exception:
                photos_by_business = {}
            try:
                review_by_business = f_reviews.result() or {}
            except Exception:
                review_by_business = {}

        for entry in used_context:
            entry["photos"] = photos_by_business.get(entry["id"], [])
            entry["top_review"] = review_by_business.get(entry["id"], "")

    yield _string_for_sse(json.dumps({
        "type": "final_answer",
        "data": {
            "answer": result["answer"],
            "used_context": used_context,
            "trace_id": result.get("trace_id", "")
        }
    }))
