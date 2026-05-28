from fastapi import APIRouter, Request
import logging
from fastapi.responses import StreamingResponse
from api.agents.graph import rag_agent_stream_wrapper
from api.api.models import RAGRequest, FeedbackRequest, FeedbackResponse
from api.api.processors.submit_feedback import submit_feedback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
)->StreamingResponse:
    return StreamingResponse(rag_agent_stream_wrapper(payload.query,payload.thread_id), media_type="text/event-stream")

@feedback_router.post("/")
def send_feedback(
    request: Request,
    payload: FeedbackRequest
)->FeedbackResponse:
    submit_feedback(payload.trace_id, payload.feedback_score, payload.feedback_text, payload.feedback_source_type)
    return FeedbackResponse(
        status="Feedback sent successfully",
        request_id=request.state.request_id
    )
app = APIRouter()
app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"])

