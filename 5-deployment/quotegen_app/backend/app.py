import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
# Note: We no longer import torch or tokenizers
from utils import sample_n

logger.add("logs/quotegen_app.log", rotation="5 MB")

frontend_folder = Path("static").resolve()
artefacts = Path("artefacts").resolve()

if not frontend_folder.exists():
    raise FileNotFoundError(f"Cant find the frontend folder at {frontend_folder}")
else:
    logger.info(f"Found {frontend_folder}")

if not artefacts.exists():
    logger.warning(f"Couldnt find artefacts at {artefacts}, trying parent")
    artefacts = Path("../artefacts").resolve()
    if not artefacts.exists():
        msg = f"Cant find the artefacts folder at {artefacts}"
        raise FileNotFoundError(msg)
    else:
        logger.info(f"Found {artefacts}")
else:
    logger.info(f"Found {artefacts}")

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory=str(frontend_folder)), name="static")


# Model loading
def load_model():
    model_file = artefacts / "markov_model.json"
    config_file = artefacts / "config.json"

    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        raise FileNotFoundError("Model not found. Run the training script first.")
        
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        raise FileNotFoundError("Config not found. Run the training script first.")

    with open(model_file, "r", encoding="utf-8") as f:
        model = json.load(f)
        
    with open(config_file, "r") as f:
        config = json.load(f)
        
    logger.info("Markov model and config loaded successfully.")
    return model, config


model, config = load_model()
saved_quotes = []


def new_quotes(n: int, temperature: float):
    # 'temperature' is no longer used by our simple Markov model,
    # but we keep it in the signature to match the frontend request.
    logger.info(f"Generating {n} new quotes (temperature={temperature} - ignored)")
    max_length = config.get("model", {}).get("max_length", 30)
    output_quotes = sample_n(
        n=n,
        model=model,
        max_length=max_length,
    )
    return output_quotes


class Quote(BaseModel):
    quote: str


@app.get("/generate")
async def generate_quotes(num_words: int = 10, temperature: float = 1.0):
    try:
        # num_words from frontend is now num_quotes
        quotes = new_quotes(num_words, temperature)
        return quotes
    except Exception as e:
        logger.error(f"Error during quote generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/starred")
async def get_starred_quotes():
    return saved_quotes


@app.post("/starred")
async def add_starred_quote(quote: Quote):
    if quote.quote not in saved_quotes:
        saved_quotes.append(quote.quote)
    return saved_quotes


@app.post("/unstarred")
async def remove_starred_quote(quote: Quote):
    if quote.quote in saved_quotes:
        saved_quotes.remove(quote.quote)
    return saved_quotes


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def read_index():
    logger.info("serving index.html")
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)