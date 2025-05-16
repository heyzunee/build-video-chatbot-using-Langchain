import logging
import os
import shutil

from .config import PROJECT_NAME

logger = logging.getLogger(PROJECT_NAME)


def cleanup(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=False)
        logger.info(f"Deleted {folder}.")
    else:
        logger.info(f"{folder} doesn't exist.")
