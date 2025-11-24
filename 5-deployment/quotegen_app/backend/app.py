import os
import json
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
import uvicorn

# --- 1. Setup Paths (Critical for Docker) ---
# This ensures we find files relative to THIS script, not the current working directory
BASE_DIR = Path(__file__).resolve().parent.parent  # Goes up from backend/ -> app/
SRC_DIR = BASE_DIR / "src"
ARTEFACTS_DIR = BASE_DIR / "artefacts"
MODEL_FILE = ARTEFACTS_DIR / "markov_model.json"

# Add src to python path so we can import quotegen
sys.path.append(str(SRC_DIR))

from quotegen.models import MarkovQuoteGenerator
from quotegen.custom_logger import logger

app = FastAPI()
model = MarkovQuoteGenerator()

# --- 2. Load Model on Startup ---
@app.on_event("startup")
async def startup_event():
    logger.info(f"Looking for model at: {MODEL_FILE}")
    
    if not MODEL_FILE.exists():
        logger.error("Model file not found! Did you run the training script?")
        return

    try:
        with open(MODEL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Reconstruct the model from JSON
        model.transition_dict = data.get("transitions", {})
        model.start_words = data.get("starts", [])
        model._START_TOKEN = "_START_"
        model._END_TOKEN = "_END_"
        
        logger.info(f"Model loaded successfully! {len(model.transition_dict)} transitions.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

# --- 3. API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "Quote Generator is running!"}

@app.get("/quote")
def get_quote(max_length: int = 30):
    if not model.transition_dict:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"quote": model.generate_quote(max_length)}

# --- 4. The Fix for Cloud Run (Host & Port) ---
if __name__ == "__main__":
    # Cloud Run gives us the PORT environment variable (default 8080)
    port = int(os.environ.get("PORT", 8080))
    
    # HOST must be 0.0.0.0 to accept connections from outside the container
    uvicorn.run("app:app", host="0.0.0.0", port=port)