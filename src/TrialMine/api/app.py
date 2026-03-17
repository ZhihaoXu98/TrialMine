"""FastAPI application factory.

Creates and configures the FastAPI app:
- Mounts API routes
- Attaches Prometheus /metrics endpoint
- Registers startup/shutdown lifecycle events (ES client, FAISS index)
- Configures structured JSON logging
"""

import logging

from fastapi import FastAPI

from TrialMine.api.routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application.

    Returns:
        FastAPI application instance ready for uvicorn.
    """
    app = FastAPI(
        title="TrialMine",
        description="ML-powered clinical trial search engine for oncology",
        version="0.1.0",
    )

    app.include_router(router)

    # TODO: mount prometheus_client.make_asgi_app() at /metrics
    # TODO: add startup event to initialise ES client, FAISS index, LLM pipeline
    # TODO: add shutdown event to close ES client
    # TODO: add structured JSON logging middleware

    return app


app = create_app()


def main() -> None:
    """Entry point for `trialmine-serve`."""
    import uvicorn
    uvicorn.run("TrialMine.api.app:app", host="0.0.0.0", port=8000, reload=False)
