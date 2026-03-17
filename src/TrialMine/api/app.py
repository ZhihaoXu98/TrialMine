"""FastAPI application factory.

Creates and configures the FastAPI app:
- CORS middleware (Streamlit runs on a different port)
- Mounts API routes
- Connects to Elasticsearch on startup
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from TrialMine.api.routes import router
from TrialMine.retrieval.bm25 import ElasticsearchIndex

logger = logging.getLogger(__name__)

ES_URL = "http://localhost:9200"
INDEX_NAME = "trials"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to Elasticsearch. Shutdown: close."""
    logger.info("Connecting to Elasticsearch at %s ...", ES_URL)
    app.state.es_index = ElasticsearchIndex(es_url=ES_URL, index_name=INDEX_NAME)
    logger.info("Elasticsearch connected.")
    yield
    app.state.es_index.es.close()
    logger.info("Elasticsearch connection closed.")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="TrialMine",
        description="ML-powered clinical trial search engine for oncology",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """Entry point for `trialmine-serve`."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run("TrialMine.api.app:app", host="0.0.0.0", port=8000, reload=True)
