from fastapi import APIRouter, FastAPI, Request
import logging

from api.agents.graph import yelp_agent_wrapper
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
)->RAGResponse:
    result = yelp_agent_wrapper(payload.query,payload.thread_id)
    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_context=[RAGUsedContext(**item) for item in result["used_context"]]
    )
app = FastAPI()
app.include_router(rag_router, prefix="/rag")
