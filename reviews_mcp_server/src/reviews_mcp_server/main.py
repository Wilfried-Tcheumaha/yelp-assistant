from fastmcp import FastMCP
from typing import List
from reviews_mcp_server.utils import retrieve_reviews_data,process_reviews_context
from reviews_mcp_server.core.config import config

mcp = FastMCP("reviews-mcp-server")

@mcp.tool
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


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)

