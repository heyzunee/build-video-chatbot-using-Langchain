import logging
import os
import uuid

import ffmpeg
import yt_dlp

from ..config import PROJECT_NAME
from ..utils import cleanup

logger = logging.getLogger(PROJECT_NAME)


def extract_audio(video_url) -> str:
    try:
        downloads = "downloads"
        os.makedirs(downloads, exist_ok=True)

        uid = str(uuid.uuid4())
        audio_path = os.path.join(downloads, f"{uid}.mp3")

        if video_url.startswith(("http://", "https://", "www.")):
            logger.info("Downloading audio from YouTube...")

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(downloads, f"{uid}.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "noplaylist": True,
                "extractaudio": True,
                "writeinfojson": False,
                "writesubtitles": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_code = ydl.download([video_url])
            if error_code:
                raise Exception("Failed to download audio from YouTube.")

        else:
            logger.info("Extracting audio from local file...")

            if not os.path.isfile(video_url):
                raise FileNotFoundError("Not found video.")
            try:
                (
                    ffmpeg.input(video_url)
                    .output(
                        audio_path, vn=None, acodec="libmp3lame", audio_bitrate="192k"
                    )
                    .run(overwrite_output=True)
                )
            except ffmpeg.Error as e:
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")

            except Exception as e:
                raise Exception(f"Failed to extract audio from local file: {str(e)}")

        logger.info("Segmenting audio...")
        output_dir = "audio"
        os.makedirs(output_dir, exist_ok=True)
        try:
            (
                ffmpeg.input(audio_path)
                .output(
                    os.path.join(output_dir, "chunk_%03d.mp3"),
                    f="segment",
                    segment_time=120,
                    c="copy",
                )
                .run(overwrite_output=True)
            )
        except Exception as e:
            raise Exception(f"Failed to segment audio: {str(e)}")

        cleanup(downloads)
        return output_dir

    except FileNotFoundError as e:
        logger.error(e)
        return None
    except Exception as e:
        logger.error(e)
        return None
