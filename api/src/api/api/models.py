from pydantic import BaseModel, Field
from typing import Union

class RAGRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG pipeline")
    thread_id: str = Field(..., description="The thread ID of the conversation")

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
    answer: str = Field(description="The answer to the question")
    request_id: str = Field(description="The request ID")
    used_context: list[RAGUsedContext] = Field(description="The context used to answer the question")
    trace_id: str = Field(description="The trace ID")

class FeedbackRequest(BaseModel):
    trace_id: str = Field(description="The trace ID")
    feedback_score: Union[int, None] = Field(description="1 if the feedback is positive, 0 if the feedback is negative")
    feedback_text: str = Field(description="The feedback text")
    feedback_source_type: str = Field(description="The feedback source type. Human or API")
    thread_id: str = Field(description="The thread ID of the conversation")

class FeedbackResponse(BaseModel):
    status: str = Field(description="The status of the feedback")
    request_id: str = Field(description="The request ID")