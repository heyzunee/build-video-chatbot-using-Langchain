import os

import torch

PROJECT_NAME = "video-qa"
DEVICE = f"cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = min(32, os.cpu_count() * 5)
SEGMENT_TIME = 120
