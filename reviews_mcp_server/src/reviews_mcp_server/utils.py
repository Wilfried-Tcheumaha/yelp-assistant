import api.patches.instructor_compat  # noqa: F401 — Superlinked NLQ vs instructor 1.14+ API

from superlinked import framework as sl
import os
from api.agents.qdrant_url import resolve_qdrant_url
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, MatchAny, FusionQuery, Prefetch


    #### Reviews Tool

def get_review_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


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