from fastapi import APIRouter, FastAPI, Request
import logging

from api.agents.retrieval_generation import rag_pipeline_wrapper
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
    result = rag_pipeline_wrapper(payload.query)
    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_context=[RAGUsedContext(**item) for item in result["used_context"]]
    )
app = FastAPI()
app.include_router(rag_router, prefix="/rag")
