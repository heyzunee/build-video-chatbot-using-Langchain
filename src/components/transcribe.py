import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from faster_whisper import WhisperModel

from ..config import *
from ..utils import cleanup

logger = logging.getLogger(PROJECT_NAME)


class Transcriber:
    def __init__(self):
        self.model = WhisperModel(
            "base", device=DEVICE, compute_type="int8" if DEVICE == "cpu" else "float16"
        )
        self.output_dir = None

    @staticmethod
    def format_timestamp(seconds):
        return str(timedelta(seconds=int(seconds))).rjust(8, "0")

    def transcribe(self, filename):
        logger.info(f"Transcribing: {filename}\n")

        chunk_index = int(filename.split("_")[1].split(".")[0])
        offset = chunk_index * SEGMENT_TIME

        segments, _ = self.model.transcribe(os.path.join(self.output_dir, filename))

        transcript = ""
        for segment in segments:
            start = self.format_timestamp(segment.start + offset)
            end = self.format_timestamp(segment.end + offset)
            transcript += f"[{start} - {end}] {segment.text.strip()}\n"

        return transcript

    def extract_transcript(self, output_dir):
        self.output_dir = output_dir
        files = sorted(
            os.listdir(output_dir), key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(self.transcribe, files))

        transcript = "\n".join(results)

        cleanup(output_dir)
        return transcript
