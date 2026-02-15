#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install uvicorn fastapi chromadb sentence-transformers httpx beautifulsoup4
echo "Install complete."
