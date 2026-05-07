"""FastAPI application factory.

Creates and configures the FastAPI app:
- CORS middleware (Streamlit runs on a different port)
- Prometheus instrumentation (counter + duration histogram + /metrics)
- Mounts API routes
- Connects to Elasticsearch, loads FAISS index + embedder on startup
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from TrialMine.api.routes import router
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.monitoring import metrics_app, metrics_middleware
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

# Configure root logging at import time so lifespan / middleware logs show up
# under uvicorn (whose CMD bypasses main()). Idempotent — basicConfig is a
# no-op if the root logger already has handlers (e.g. when main() runs).
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

logger = logging.getLogger(__name__)

# Driven by env so Docker can point at the `elasticsearch` service hostname.
ES_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "trials"
FAISS_INDEX_PATH = "data/trial_embeddings.faiss"
FAISS_MAPPING_PATH = "data/trial_embeddings.json"
EMBEDDER_MODEL = "michiyasunaga/BioLinkBERT-base"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown handler.

    Each resource loads independently and tolerantly — if any one is
    unavailable the API still starts in a degraded mode and routes return
    structured 503s for the affected feature (per the project rule of
    never letting the API return a raw 500).
    """
    # Elasticsearch — tolerant of being down at startup
    logger.info("Connecting to Elasticsearch at %s ...", ES_URL)
    try:
        app.state.es_index = ElasticsearchIndex(
            es_url=ES_URL, index_name=INDEX_NAME
        )
        logger.info("Elasticsearch connected.")
    except Exception as exc:
        logger.warning(
            "Elasticsearch unavailable at startup: %s — bm25 / hybrid / "
            "trial-lookup endpoints will return 503 until restart.",
            exc,
        )
        app.state.es_index = None

    # FAISS index
    faiss_path = Path(FAISS_INDEX_PATH)
    if faiss_path.exists():
        logger.info("Loading FAISS index from %s ...", FAISS_INDEX_PATH)
        try:
            app.state.faiss_index = FAISSIndex()
            app.state.faiss_index.load(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
            logger.info(
                "FAISS index loaded (%d vectors).",
                app.state.faiss_index.index.ntotal,
            )
        except Exception as exc:
            logger.warning("FAISS index load failed: %s", exc)
            app.state.faiss_index = None
    else:
        logger.warning(
            "FAISS index not found at %s — semantic search disabled.",
            FAISS_INDEX_PATH,
        )
        app.state.faiss_index = None

    # Embedder
    if app.state.faiss_index is not None:
        logger.info("Loading embedding model %s ...", EMBEDDER_MODEL)
        try:
            app.state.embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
            logger.info("Embedder loaded.")
        except Exception as exc:
            logger.warning("Embedder load failed: %s", exc)
            app.state.embedder = None
    else:
        app.state.embedder = None

    # Hybrid retriever — needs all three
    if (
        app.state.es_index is not None
        and app.state.faiss_index is not None
        and app.state.embedder is not None
    ):
        app.state.hybrid_retriever = HybridRetriever(
            bm25=app.state.es_index,
            semantic=app.state.faiss_index,
            embedder=app.state.embedder,
        )
        logger.info("HybridRetriever initialised.")
    else:
        app.state.hybrid_retriever = None

    # Agent pipeline — graph compile is cheap (<100 ms); the heavy resources
    # (cross-encoder, LightGBM blender) lazy-load via tools.py singletons.
    try:
        from TrialMine.agents.pipeline import build_pipeline

        app.state.pipeline = build_pipeline()
        logger.info("Agent pipeline compiled.")
    except Exception as exc:
        logger.warning("Agent pipeline build failed: %s", exc)
        app.state.pipeline = None

    # Pre-warm CE + LightGBM blender on the agent's hybrid retriever. Without
    # this, the first /api/v1/search call with use_agent=True pays ~12 s of
    # CE cold-load + first inference inside the 15 s agent budget — works,
    # but tight enough that one slow Anthropic round-trip tips it into a
    # timeout. Run a single throwaway retrieval to JIT-warm the inference
    # path; cost moves from request time into startup time (compose's
    # start_period: 90s already accommodates a longer lifespan).
    #
    # IMPORTANT: warm the SAME singletons the orchestrator will call. The
    # orchestrator uses ``tools._get_hybrid()``, not ``app.state.hybrid_retriever``
    # — they share CE + blender singletons but the retriever objects (and
    # their internal embedder caches) are independent. Warming the wrong
    # one was the bug in the first version of this code.
    if app.state.pipeline is not None:
        try:
            from TrialMine.agents.tools import (
                _get_blender_or_none,
                _get_hybrid,
                _get_reranker_or_none,
            )

            t0 = time.perf_counter()
            hybrid = _get_hybrid()
            reranker = _get_reranker_or_none()
            blender = _get_blender_or_none()
            if reranker is not None and hybrid is not None:
                hybrid.full_pipeline(
                    query="warmup",
                    reranker=reranker,
                    blender=blender,
                    top_k=1,
                    rerank_top_k=5,
                    filters=None,
                )
                logger.info(
                    "Pre-warmed agent retriever + CE + blender "
                    "(blender=%s) in %.0f ms",
                    "loaded" if blender is not None else "missing",
                    (time.perf_counter() - t0) * 1000,
                )
            else:
                logger.warning(
                    "Pre-warm skipped: hybrid=%s, reranker=%s. First "
                    "agent query will pay the cold-load cost.",
                    "loaded" if hybrid is not None else "None",
                    "loaded" if reranker is not None else "None",
                )
        except Exception as exc:
            # Non-fatal: first request just pays the cold-load tax instead.
            logger.warning("Pre-warm failed (non-fatal): %s", exc)

    yield

    if app.state.es_index is not None:
        try:
            app.state.es_index.es.close()
            logger.info("Elasticsearch connection closed.")
        except Exception as exc:
            logger.warning("Error closing Elasticsearch client: %s", exc)


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

    # Prometheus: per-request counter + latency histogram, then /metrics scrape
    # endpoint mounted as a sub-asgi app (Starlette / FastAPI native pattern).
    app.add_middleware(BaseHTTPMiddleware, dispatch=metrics_middleware)
    app.mount("/metrics", metrics_app())

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
