import re
from pathlib import Path
import kagglehub
import pandas as pd
from quotegen.custom_logger import logger

def clean_text(text):
    """Cleans a single quote."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", '', text)  # Keep letters, spaces, and apostrophes
    text = re.sub(r"\s+", ' ', text).strip() # Replace multiple spaces with one
    return text

def load_data(assets_dir: Path, filename: str) -> list[str]:
    """
    Downloads the dataset from Kaggle, processes it, and saves it to the assets_dir.
    Returns a list of cleaned quotes.
    """
    datafile = assets_dir / filename
    
    if not datafile.exists():
        logger.info(f"Dataset not found at {datafile}, downloading from Kaggle...")
        # Download latest version
        try:
            path = kagglehub.dataset_download("manann/quotes-500k")
            logger.info(f"Path to dataset files: {path}")
            
            # The dataset might be in a sub-folder, let's find the csv
            csv_files = list(Path(path).rglob("*.csv"))
            if not csv_files:
                logger.error("No CSV file found in downloaded Kaggle dataset.")
                return []
            
            # Use the first CSV file found
            downloaded_csv = csv_files[0]
            logger.info(f"Reading from {downloaded_csv}")
            
            # Ensure assets directory exists
            assets_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file to our assets directory
            import shutil
            shutil.copy(downloaded_csv, datafile)
            
        except Exception as e:
            logger.error(f"Failed to download or process Kaggle dataset: {e}")
            return []
    
    # Now, load and process the file from our assets_dir
    logger.info(f"Loading and processing data from {datafile}")
    try:
        df = pd.read_csv(datafile)
        
        if "quote" not in df.columns: # Changed "Quote" to "quote"
            logger.error(f"'quote' column not in CSV. Available columns: {df.columns}")
            return []
            
        processed_quotes = (
            df["quote"] # Changed "Quote" to "quote"
            .dropna()
            .apply(clean_text)
            .unique()
        )
        
        # Filter out empty strings that might result from cleaning
        processed_quotes = [q for q in processed_quotes if q]
        
        logger.info(f"Loaded and cleaned {len(processed_quotes)} unique quotes.")
        return processed_quotes
        
    except Exception as e:
        logger.error(f"Failed to read or process CSV file {datafile}: {e}")
        return []