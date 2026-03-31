from api.core.config import config
from qdrant_client import QdrantClient
from superlinked import framework as sl
import json
import os
from typing import Any
import openai
from api.agents.superlinked_app.index import business_index, business
from api.agents.superlinked_app.query import query
from api.agents.superlinked_app.utils.utils import *

qdrant_vdb = sl.QdrantVectorDatabase(
    url="http://qdrant:6333",
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
qdrant_app = executor_qdrant.run()

def Retrieve_context(question, qdrant_app, k=5):
    qdrant_results = qdrant_app.query(
    query,
    natural_query=question,
    limit=k,
)
    format_minute_columns_to_hhmm(sl.PandasConverter.to_pandas(qdrant_results))
    return qdrant_results


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

def build_prompt(preprocessed_context, question):
    prompt=f"""
    You are a yelp shopping assistant that can answer question about the restaurants.
    You will be given a question and a list of context.
    Instructions:
    - You need to answer questions based on the provided context only
    - Never use the word context and rfer to it as the available businesses or amenities
    - respond naturally and provide as much details as possible to the user request 
    - Refrain from using filter sush as is_open =True ...Rather say open today

    Context:
    {preprocessed_context}

    Question:
    {question}
    """

    return prompt

def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role":"system", "content": prompt}],
        reasoning_effort="medium"
    )
    return response.choices[0].message.content

def rag_pipeline(question):
    context=Retrieve_context(question, qdrant_app)
    preprocessed_context=_result_to_restaurants(context)
    prompt=build_prompt(preprocessed_context, question)
    answer=generate_answer(prompt)
    return answer