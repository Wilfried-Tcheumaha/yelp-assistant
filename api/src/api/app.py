import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.api.middleware import RequestIDMiddleware
from api.api.endpoints import rag_router, feedback_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm heavy models at startup so the first user request isn't slow.

    CLIP weights are ~150MB and load takes several seconds (especially under
    x86 emulation on Apple Silicon). Loading them at startup means the slow
    cost is paid once per container life, not on the first /rag/ call.
    Failure here is non-fatal: the lazy loader inside tools.py will retry
    on first use and `get_photos_for_businesses` falls back to "no photos".
    """
    try:
        from api.agents.tools import _get_clip_model
        _get_clip_model()
        logger.info("CLIP pre-loaded at startup.")
    except Exception:
        logger.exception("CLIP pre-load failed; falling back to lazy load on first request.")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"])

# Yelp restaurant photos. The bind-mount in docker-compose maps the local
# `data/raw/photos/` into `/app/photos`. We skip the mount silently if the
# directory is missing so the API still boots in environments that don't
# ship photos.
PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/app/photos")
if os.path.isdir(PHOTOS_DIR):
    app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")
    logger.info("Mounted /photos -> %s", PHOTOS_DIR)
else:
    logger.warning(
        "Photos directory %r not found; /photos will return 404. "
        "Bind-mount data/raw/photos in docker-compose to enable.",
        PHOTOS_DIR,
    )
