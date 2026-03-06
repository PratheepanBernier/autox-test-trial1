import logging
import sys

from fastapi import FastAPI

from backend.src.api.routes import router
from backend.src.core.config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)
    app.include_router(router)

    @app.get("/")
    def health() -> dict[str, str]:
        logger.info("Health check endpoint called")
        return {"status": "running"}

    return app


app = create_app()
