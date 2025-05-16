import logging

import gradio as gr
from langchain.schema import Document
from predict import Predict
from src.components.extract_audio import extract_audio
from src.components.split_document import Splitter
from src.components.transcribe import Transcriber
from src.config import PROJECT_NAME

logger = logging.getLogger(PROJECT_NAME)


class Handler:
    def __init__(self):
        self.predictor = Predict()
        self.transcriber = Transcriber()
        self.splitter = Splitter()

    def upload_video(self, video_file, video_link):
        """Process video and return transcript chunks"""
        if video_file is not None:
            video_url = video_file.name
        elif video_link.strip() != "":
            video_url = video_link
        else:
            return (
                None,
                gr.update(interactive=False),
                "⚠️ Please upload a video or provide a valid link.",
            )

        try:
            # Extract audio from video
            audio_path = extract_audio(video_url)

            # Transcribe audio
            transcript = self.transcriber.extract_transcript(audio_path)

            # Create document and split into chunks
            document = Document(
                page_content=transcript, metadata={"source": "transcript"}
            )
            chunks = self.splitter.chunking([document])

            # Return video chunks and success message
            return (
                chunks,
                gr.update(interactive=True),
                "<div class='status success'>✅ The video has been processed successfully! You can now ask questions about its content.</div>",
            )
        except Exception as e:
            return (
                None,
                gr.update(interactive=False),
                f"<div class='status error'>❌ Video processing error: {str(e)}</div>",
            )

    def submit_message(self, message, history):
        logger.debug(
            "Message received:", message
        )  # {'text': 'how is the weather today?', 'files': []}
        logger.debug(
            "Current history:", history
        )  # [{'role': 'user', 'metadata': None, 'content': 'what is your name', 'options': None}, {'role': 'assistant', 'metadata': None, 'content': "My name is Zara, and I am a skilled and dedicated AI assistant. I am here to help you with any queries or challenges you may have regarding your digital life, whether it's about your online presence, social media, or any other aspect of your online presence.", 'options': None}]

        text = message.get("text", "")

        # files = message.get("files", [])
        # if files and len(files) > 0:
        #     response = f"Received text: {text} with {len(files)} files."
        #     history.append({"role": "user", "content": text})
        #     history.append({"role": "assistant", "content": response})
        #     return None, history

        # Update the history with the user message
        history.append({"role": "user", "content": text})

        try:
            # Get the answer
            response = self.predictor.predict(history)
        except Exception as e:
            logger.error(f"Error in predictor: {str(e)}")

        # Update history with the assistant's response
        history.append({"role": "assistant", "content": response})

        return None, history

    @staticmethod
    def print_like_dislike(x: gr.LikeData):
        print(x.index, x.value, x.liked)
