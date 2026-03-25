"""FastAPI application factory.

Creates and configures the FastAPI app:
- CORS middleware (Streamlit runs on a different port)
- Mounts API routes
- Connects to Elasticsearch, loads FAISS index + embedder on startup
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from TrialMine.api.routes import router
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

logger = logging.getLogger(__name__)

ES_URL = "http://localhost:9200"
INDEX_NAME = "trials"
FAISS_INDEX_PATH = "data/trial_embeddings.faiss"
FAISS_MAPPING_PATH = "data/trial_embeddings.json"
EMBEDDER_MODEL = "michiyasunaga/BioLinkBERT-base"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to Elasticsearch, load FAISS + embedder. Shutdown: close."""
    # Elasticsearch
    logger.info("Connecting to Elasticsearch at %s ...", ES_URL)
    app.state.es_index = ElasticsearchIndex(es_url=ES_URL, index_name=INDEX_NAME)
    logger.info("Elasticsearch connected.")

    # FAISS index
    faiss_path = Path(FAISS_INDEX_PATH)
    if faiss_path.exists():
        logger.info("Loading FAISS index from %s ...", FAISS_INDEX_PATH)
        app.state.faiss_index = FAISSIndex()
        app.state.faiss_index.load(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
        logger.info("FAISS index loaded (%d vectors).", app.state.faiss_index.index.ntotal)
    else:
        logger.warning("FAISS index not found at %s — semantic search disabled.", FAISS_INDEX_PATH)
        app.state.faiss_index = None

    # Embedder
    if app.state.faiss_index is not None:
        logger.info("Loading embedding model %s ...", EMBEDDER_MODEL)
        app.state.embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
        logger.info("Embedder loaded.")
    else:
        app.state.embedder = None

    # Hybrid retriever
    if app.state.faiss_index is not None and app.state.embedder is not None:
        app.state.hybrid_retriever = HybridRetriever(
            bm25=app.state.es_index,
            semantic=app.state.faiss_index,
            embedder=app.state.embedder,
        )
        logger.info("HybridRetriever initialised.")
    else:
        app.state.hybrid_retriever = None

    yield

    app.state.es_index.es.close()
    logger.info("Elasticsearch connection closed.")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="TrialMine",
        description="ML-powered clinical trial search engine for oncology",
        version="0.2.0",
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
