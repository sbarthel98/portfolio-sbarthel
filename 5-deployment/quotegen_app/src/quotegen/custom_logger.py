# logger_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

class CustomLogger:
    def __init__(self, name='main', log_file='logs/main.log', max_bytes=5*1024*1024, backup_count=5):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Check if handlers already exist to prevent adding them multiple times
        if not self.logger.hasHandlers():
            
            # Create a logging format
            formatter = logging.Formatter('%(asctime)s | %(name)s - %(levelname)s - %(message)s')

            # --- Handler 1: Console (CRITICAL for Cloud Run) ---
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # --- Handler 2: File (Only if we can write to it) ---
            try:
                # Use absolute path relative to this file's parent
                log_path = Path(log_file)
                if not log_path.is_absolute():
                    # If relative, make it relative to the project root (2 levels up from here)
                    log_path = Path(__file__).resolve().parents[2] / log_file
                
                logfolder = log_path.parent
                if not logfolder.exists():
                    logfolder.mkdir(parents=True, exist_ok=True)

                file_handler = RotatingFileHandler(str(log_path), maxBytes=max_bytes, backupCount=backup_count)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                # If file logging fails (e.g., Read-Only filesystem in Cloud Run), just warn and continue
                print(f"Warning: Could not set up file logging: {e}")

    def __getattr__(self, attr):
        return getattr(self.logger, attr)

# Instantiate and configure the custom logger
logger = CustomLogger().logger