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
from api.agents.tools import get_formatted_context,get_formatted_reviews_context
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
def agent_execution(question: str, thread_id: str) -> dict:
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "available_tools": tool_descriptions,
        "iteration": 0
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with PostgresSaver.from_conn_string(
    "postgresql://langgraph_user:postgres_password@postgres:5432/langgraph_db") as checkpointer:
        graph=workflow.compile(checkpointer=checkpointer)
        result = graph.invoke(initial_state, config)
    return result
    

def yelp_agent_wrapper(question, thread_id: str):
    """Wrapper for the Yelp agent execution"""
    result = agent_execution(question, thread_id)

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
        "trace_id": result.get("trace_id", "")
    }