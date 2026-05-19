import api.patches.instructor_compat  # noqa: F401 — Superlinked NLQ vs instructor 1.14+ API

from superlinked import framework as sl
import os
from api.agents.qdrant_url import resolve_qdrant_url
from api.agents.superlinked_app.index import business_index, business
from api.agents.superlinked_app.query import query
from api.agents.superlinked_app.utils.utils import *
from langsmith import traceable, get_current_run_tree
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, MatchAny, FusionQuery, Prefetch


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
    """Lazy init: avoids opening a Qdrant connection at module import time."""
    global _qdrant_app
    if _qdrant_app is None:
        _qdrant_app = executor_qdrant.run()
    return _qdrant_app


@traceable(
    name="retriever_top_n",
    run_type="retriever",
    )
def retrieve_context(question, k=5):
    qdrant_results = get_qdrant_app().query(
        query,
        natural_query=question,
        limit=k,
    )

    format_minute_columns_to_hhmm(sl.PandasConverter.to_pandas(qdrant_results))

    return {
        "retrived_restaurant_ids":[e.id for e in qdrant_results.entries],
        "retrived_restaurants_names":[e.fields.get("name") for e in qdrant_results.entries],
        "retrived_restaurants_ratings":[e.fields.get("stars") for e in qdrant_results.entries],
        "retrived_restaurants_reviews_count":[e.fields.get("review_count") for e in qdrant_results.entries],
        "retrived_states":[e.fields.get("state") for e in qdrant_results.entries],
        "retrived_cities":[e.fields.get("city") for e in qdrant_results.entries],
        "similarity_scores":[e.metadata.score for e in qdrant_results.entries],
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context=""
    for id, name, rating, review_count, state, city, similarity_score in zip(context["retrived_restaurant_ids"], context["retrived_restaurants_names"], context["retrived_restaurants_ratings"], context["retrived_restaurants_reviews_count"], context["retrived_states"], context["retrived_cities"], context["similarity_scores"]):
        formatted_context += f"-ID: {id}, Name: {name}, Rating: {rating}, Review Count: {review_count}, State: {state}, City: {city}, Similarity Score: {similarity_score}\n"

    return formatted_context


def get_formatted_context(query:str, top_k:int=5)->str:
    """Get the top k context, each representing a restaurant for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_context(query, top_k)
    formatted_context = process_context(context)

    return formatted_context

    #### Reviews Tool
@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider":"openai","ls_model":"text-embedding-3-small"}

)
def get_review_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding

@traceable(
    name="retrieve_reviews_data",
    run_type="retriever"
)
def retrieve_reviews_data(query, business_ids, k=5):

    query_embedding = get_review_embeddings(query)
    qdrant_client=QdrantClient(
        url=resolve_qdrant_url(),
        api_key=os.getenv("QDRANT_API_KEY", ""),
    )

    results = qdrant_client.query_points(
        collection_name="yelp-reviews-collection-00",
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="business_id",
                            match=MatchAny(
                                any=business_ids
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )
    retreved_context_ids=[]
    retrieved_context=[]
    similarity_scores=[]
    for result in results.points:
        retreved_context_ids.append(result.payload["business_id"])
        retrieved_context.append(result.payload["text"])
        similarity_scores.append(result.score)

    return {
        "retreved_context_ids":retreved_context_ids,
        "retrieved_context":retrieved_context,
        "similarity_scores":similarity_scores,
    }

@traceable(
    name="process_reviews_context",
    run_type="prompt"
)
def process_reviews_context(context):
    formatted_reviews_context=""
    for id, chunk_context in zip(context["retreved_context_ids"], context["retrieved_context"]):
        formatted_reviews_context += f"-ID: {id}, review: {chunk_context}\n"
    return formatted_reviews_context

def get_formatted_reviews_context(query:str, business_ids:list[str], k:int=15)->str:
    """Get the top k reviews context for a given query and business ids.
    
    Args:
        query: The query to get the top k reviews context for
        business_ids: The list of business ids to get the reviews context for
        k: The number of reviews context to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k reviews context with IDs and reviews prepending each chunk, each representing a review for a given query and business ids.
    """
    context = retrieve_reviews_data(query, business_ids, k)
    formatted_reviews_context = process_reviews_context(context)
    return formatted_reviews_context