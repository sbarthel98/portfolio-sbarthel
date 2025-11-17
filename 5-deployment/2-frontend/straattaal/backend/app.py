import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from slanggen import models
from tokenizers import Tokenizer
from utils import sample_n

logger.add("logs/app.log", rotation="5 MB")

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
    tokenizerfile = str(artefacts / "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizerfile)
    with (artefacts / "config.json").open("r") as f:
        config = json.load(f)
    model = models.SlangRNN(config["model"])
    modelpath = str(artefacts / "model.pth")
    model.load_state_dict(torch.load(modelpath, weights_only=False))
    return model, tokenizer


model, tokenizer = load_model()
starred_words = []


def new_words(n: int, temperature: float):
    output_words = sample_n(
        n=n,
        model=model,
        tokenizer=tokenizer,
        max_length=20,
        temperature=temperature,
    )
    return output_words


class Word(BaseModel):
    word: str


@app.get("/generate")
async def generate_words(num_words: int = 10, temperature: float = 1.0):
    try:
        words = new_words(num_words, temperature)
        return words
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/starred")
async def get_starred_words():
    return starred_words


@app.post("/starred")
async def add_starred_word(word: Word):
    if word.word not in starred_words:
        starred_words.append(word.word)
    return starred_words


@app.post("/unstarred")
async def remove_starred_word(word: Word):
    if word.word in starred_words:
        starred_words.remove(word.word)
    return starred_words


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
