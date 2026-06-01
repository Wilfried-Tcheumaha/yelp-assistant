import api.patches.instructor_compat  # noqa: F401 — Superlinked NLQ vs instructor 1.14+ API

import concurrent.futures
import contextvars
import logging
import os
from typing import Iterable

from superlinked import framework as sl
from api.agents.qdrant_url import resolve_qdrant_url
from api.agents.superlinked_app.index import business_index, business
from api.agents.superlinked_app.query import query
from api.agents.superlinked_app.utils.utils import *
from langsmith import traceable, get_current_run_tree
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, MatchAny, FusionQuery, Prefetch

logger = logging.getLogger(__name__)


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


@traceable(name="retrieve_top_review", run_type="retriever")
def get_top_review_for_businesses(
    query: str,
    business_ids: Iterable[str],
) -> dict[str, str]:
    """Single most-relevant review per business, ranked by RRF against the user's question.

    Reuses ``retrieve_reviews_data`` (same prefetch + RRF flow as ``get_formatted_reviews_context``)
    but groups results by ``business_id`` so each cited business gets one quote for the card.

    Returns ``{business_id: review_text}``. Soft-fails to ``{}`` so the caller can attach
    no review without raising.
    """
    business_ids = [bid for bid in business_ids if bid]
    if not business_ids:
        return {}

    # Pull a few reviews per business so we still have a result if the top candidate
    # belongs to only one of them (RRF ranks globally, not per-business).
    k = max(len(business_ids) * 4, 8)
    try:
        ctx = retrieve_reviews_data(query, business_ids, k=k)
    except Exception:
        logger.exception("Top-review retrieval failed — skipping.")
        return {}

    top_by_business: dict[str, str] = {}
    for bid, review in zip(ctx["retreved_context_ids"], ctx["retrieved_context"]):
        if bid in top_by_business:
            continue
        if not review:
            continue
        top_by_business[bid] = review
    return top_by_business


    #### Photos Tool

# Hybrid photo retrieval: rank with both the *caption text* and the *image* of
# each photo, then fuse the two rankings with Reciprocal Rank Fusion. The
# caption side catches literal terms ("heaters", "truffle fries") while the
# CLIP image side catches visual concepts the caption never mentions (a heat
# lamp visible in the frame). This mirrors the pattern Yelp's photo search
# uses, and reuses the RRF flow the reviews tool already exercises.
PHOTOS_COLLECTION = os.getenv("QDRANT_PHOTOS_COLLECTION", "yelp-photos-collection-00")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")

# Named vectors inside the photos collection — must match notebooks/17-...
PHOTOS_VECTOR_TEXT = "caption_text"   # 1536-d, OpenAI text-embedding-3-small
PHOTOS_VECTOR_IMAGE = "image_clip"    # 512-d, CLIP-ViT-B-32

# Public path the API serves photo files at (matches the StaticFiles mount in app.py).
PHOTO_URL_PREFIX = "/photos"

_clip_model = None


def _get_clip_model():
    """Lazily load CLIP. Re-raises so the caller can decide how to fall back."""
    global _clip_model
    if _clip_model is None:
        # Imported here so importing this module doesn't pull torch into
        # processes (e.g. unit tests) that won't actually need CLIP.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading CLIP model %s ...", CLIP_MODEL_NAME)
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME)
        logger.info("CLIP model loaded.")
    return _clip_model


@traceable(
    name="embed_photo_clip",
    run_type="embedding",
    metadata={"ls_provider": "sentence-transformers", "ls_model": CLIP_MODEL_NAME},
)
def embed_photo_clip(query: str):
    """L2-normalised 512-d CLIP text embedding — same space as the indexed photo images."""
    model = _get_clip_model()
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]


@traceable(
    name="embed_photo_caption",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model": "text-embedding-3-small"},
)
def embed_photo_caption(query: str):
    """OpenAI text-embedding-3-small — same encoder used at index time over the captions."""
    return get_review_embeddings(query)


@traceable(name="retrieve_photos", run_type="retriever")
def get_photos_for_businesses(
    query: str,
    business_ids: Iterable[str],
    top_k: int = 30,
    photos_per_business: int = 3,
) -> dict[str, list[dict]]:
    """Hybrid (caption-text + image) photo retrieval, scoped to a business set, fused with RRF.

    Returns ``{ business_id: [ {photo_id, caption, label, score, url}, ... ] }``.

    On any failure (missing collection, CLIP load error, OpenAI/Qdrant error), returns ``{}``
    so the caller can attach an empty ``photos`` list per entry without raising.
    """
    business_ids = [bid for bid in business_ids if bid]
    if not business_ids:
        return {}

    # Encode both query embeddings in parallel: total wall-clock is max(text, clip)
    # instead of text+clip. Each task is its own LangSmith span via @traceable.
    #
    # contextvars.copy_context() snapshots the calling thread's context (which
    # holds LangSmith's current-run pointer set by @traceable) so the worker
    # threads see this function as their parent run. Without this, embed_*
    # spans run with empty contextvars in the worker thread and either become
    # orphan top-level traces or get dropped entirely.
    try:
        ctx_text = contextvars.copy_context()
        ctx_clip = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_text = ex.submit(ctx_text.run, embed_photo_caption, query)
            f_clip = ex.submit(ctx_clip.run, embed_photo_clip, query)
            text_qvec = f_text.result()
            clip_qvec = f_clip.result()
    except Exception:
        logger.exception("Photo query embedding failed — skipping photo enrichment.")
        return {}

    qdrant_client = QdrantClient(
        url=resolve_qdrant_url(),
        api_key=os.getenv("QDRANT_API_KEY", ""),
    )

    try:
        if not qdrant_client.collection_exists(PHOTOS_COLLECTION):
            logger.info(
                "Photos collection %r does not exist — skipping photo enrichment.",
                PHOTOS_COLLECTION,
            )
            return {}
    except Exception:
        logger.exception("Could not check photos collection — skipping.")
        return {}

    biz_filter = Filter(must=[
        FieldCondition(key="business_id", match=MatchAny(any=business_ids)),
    ])
    prefetch_limit = max(20, photos_per_business * len(business_ids) * 4)
    final_limit = max(top_k, photos_per_business * len(business_ids))

    try:
        response = qdrant_client.query_points(
            collection_name=PHOTOS_COLLECTION,
            prefetch=[
                Prefetch(
                    query=text_qvec,
                    using=PHOTOS_VECTOR_TEXT,
                    filter=biz_filter,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=clip_qvec.tolist(),
                    using=PHOTOS_VECTOR_IMAGE,
                    filter=biz_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion="rrf"),
            limit=final_limit,
            with_payload=True,
        )
    except Exception:
        logger.exception("Photo retrieval query failed — skipping.")
        return {}

    by_business: dict[str, list[dict]] = {bid: [] for bid in business_ids}
    for point in response.points:
        payload = point.payload or {}
        bid = payload.get("business_id")
        if bid not in by_business:
            continue
        if len(by_business[bid]) >= photos_per_business:
            continue
        photo_id = payload.get("photo_id")
        if not photo_id:
            continue
        by_business[bid].append({
            "photo_id": photo_id,
            "caption": payload.get("caption") or "",
            "label": payload.get("label") or "",
            "score": float(point.score),
            "url": f"{PHOTO_URL_PREFIX}/{photo_id}.jpg",
        })

    return by_business