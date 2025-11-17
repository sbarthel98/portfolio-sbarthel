import json
import os
import tomllib
from pathlib import Path

from quotegen import datatools, models
from quotegen.custom_logger import logger

def main():
    # Get config
    configfile = Path("quotegen.toml").resolve()
    if not configfile.exists():
        logger.error(f"Config file not found at {configfile}")
        return
        
    with configfile.open(mode="rb") as f:
        config = tomllib.load(f)

    # Define paths
    assets_dir = Path(config["data"]["assets_dir"])
    artefacts_dir = Path(config["data"]["artefacts_dir"])
    data_filename = config["data"]["filename"]
    
    # Ensure directories exist
    assets_dir.mkdir(exist_ok=True)
    artefacts_dir.mkdir(exist_ok=True)

    # Load data
    logger.info("Loading and preprocessing data...")
    quotes = datatools.load_data(assets_dir, data_filename)
    
    if not quotes:
        logger.error("No quotes were loaded. Exiting training.")
        return

    # "Train" model (build transition dictionary)
    logger.info("Building model...")
    model = models.MarkovQuoteGenerator()
    model.build_model(quotes)

    # Save artefacts
    model_file = artefacts_dir / "markov_model.json"
    with open(model_file, "w", encoding="utf-8") as f:
        json.dump({
            "transitions": model.transition_dict,
            "starts": model.start_words
        }, f, indent=2)
    logger.info(f"Model saved to {model_file}")

    # Save config to artefacts folder
    config_file = artefacts_dir / "config.json"
    with open(config_file, "w") as f:
        f.write(json.dumps(config, indent=4))
    logger.info(f"Config saved to {config_file}")

if __name__ == "__main__":
    main()