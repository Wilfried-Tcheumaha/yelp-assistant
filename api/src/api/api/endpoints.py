from fastapi import FastAPI, Request,APIRouter
from api.core.config import config
from api.api.models import RAGRequest, RAGResponse
import logging
from api.agents.retrieval_generation import rag_pipeline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
)->RAGResponse:
    result=rag_pipeline(payload.query)
    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"]
    )
app = FastAPI()
app.include_router(rag_router, prefix="/rag")
