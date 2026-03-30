from fastapi import FastAPI
from api.api.middleware import RequestIDMiddleware
from fastapi.middleware.cors import CORSMiddleware
from api.api.endpoints import rag_router

app = FastAPI()
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rag_router, prefix="/rag")
