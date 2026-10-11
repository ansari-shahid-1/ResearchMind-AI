"""FastAPI entry point.

    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import registry
from app.api.routes import router
from app.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    registry.init()
    yield


app = FastAPI(
    title="ResearchMind AI",
    description="Ask questions across a set of research papers, with page-level citations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"name": "ResearchMind AI", "docs": "/docs", "health": "/health"}
