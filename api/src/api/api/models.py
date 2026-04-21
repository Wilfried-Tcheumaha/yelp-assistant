from pydantic import BaseModel, Field

class RAGRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG pipeline")

class RAGUsedContext(BaseModel):
    id: str = Field(..., description="The id of the restaurant used to answer the question")
    description: str = Field(..., description="The description of the restaurant")
    name: str = Field(..., description="The name of the restaurant")
    address: str = Field(..., description="The address of the restaurant")
    latitude: float | None = Field(None, description="Latitude")
    longitude: float | None = Field(None, description="Longitude")
    stars: float | None = Field(None, description="Average rating (1-5)")
    reviews: int | None = Field(None, description="Total number of reviews")
    categories: list[str] = Field(..., description="The categories of the restaurant")
    attributes: dict = Field(..., description="The attributes of the restaurant")
    hours: dict = Field(..., description="The hours of the restaurant")

class RAGResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    request_id: str = Field(..., description="The request ID")
    used_context: list[RAGUsedContext] = Field(..., description="The context used to answer the question")