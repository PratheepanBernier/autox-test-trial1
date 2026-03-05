from fastapi import FastAPI
from api.routes import router
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Logistics Document Intelligence Assistant")
app.include_router(router)

@app.get("/")
def health():
    logger.info("Health check endpoint called")
    return {"status": "running"}
