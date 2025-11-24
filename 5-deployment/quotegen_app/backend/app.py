import os
import json
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# --- 1. Setup Paths ---
# Get the absolute path to the folder this file is in (backend/)
BACKEND_DIR = Path(__file__).resolve().parent
# Go up one level to find the project root
PROJECT_ROOT = BACKEND_DIR.parent
# Define where other folders are
SRC_DIR = PROJECT_ROOT / "src"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"
STATIC_DIR = BACKEND_DIR / "static"
MODEL_FILE = ARTEFACTS_DIR / "markov_model.json"

# Add src to python path so we can import quotegen
sys.path.append(str(SRC_DIR))

from quotegen.models import MarkovQuoteGenerator
from quotegen.custom_logger import logger

app = FastAPI()
model = MarkovQuoteGenerator()

# --- 2. Load Model ---
@app.on_event("startup")
async def startup_event():
    logger.info(f"Looking for model at: {MODEL_FILE}")
    if not MODEL_FILE.exists():
        logger.error("Model file not found!")
        return

    try:
        with open(MODEL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        model.transition_dict = data.get("transitions", {})
        model.start_words = data.get("starts", [])
        model._START_TOKEN = "_START_"
        model._END_TOKEN = "_END_"
        logger.info(f"Model loaded! {len(model.transition_dict)} transitions.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

# --- 3. Mount Static Files ---
# This allows /static/styles.css to work
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- 4. API Endpoints ---

# Serve your frontend at the root URL
@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")

# The endpoint your frontend calls
@app.get("/generate")
def generate_quotes(
    num_words: int = Query(5, alias="num_words"), 
    temperature: float = 1.0
):
    """
    Generates N quotes. 
    Note: The current Markov model implementation might not support temperature,
    but we accept the parameter to prevent errors from the frontend.
    """
    if not model.transition_dict:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate 'num_words' amount of quotes (using your sample_n method)
    quotes = model.sample_n(n=num_words)
    return quotes

# Keep the old endpoint just in case
@app.get("/quote")
def get_single_quote():
    if not model.transition_dict:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"quote": model.generate_quote()}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)