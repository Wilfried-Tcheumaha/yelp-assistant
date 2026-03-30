from pydantic import BaseModel, Field

class RAGRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG pipeline")
    top_k: int

class RAGResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    request_id: str = Field(..., description="The request ID")