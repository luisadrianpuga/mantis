import logging
from dotenv import load_dotenv

from fastapi import FastAPI

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mantis")

from api.routes import router

app = FastAPI(title="Mantis Runtime", version="0.1.0")

@app.get("/")
def health():
    return {"status": "ok"}

app.include_router(router)
