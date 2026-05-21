from fastmcp import FastMCP
from typing import List
from restaurants_mcp_server.utils import retrieve_context,process_context
from restaurants_mcp_server.core.config import config

mcp = FastMCP("restaurants-mcp-server")

@mcp.tool
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


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)

