import logging
from dotenv import load_dotenv

from fastapi import FastAPI

from config import MANTIS_VERSION

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mantis")

from api.routes import bootstrap_runtime_identity, router

app = FastAPI(title="Mantis Runtime", version=MANTIS_VERSION)


@app.on_event("startup")
async def startup() -> None:
    bootstrap_runtime_identity()


@app.get("/")
def health():
    return {"status": "ok", "version": MANTIS_VERSION}

app.include_router(router)
