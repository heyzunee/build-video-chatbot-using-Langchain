import os
import re
import tempfile
import traceback
import time

import streamlit as st
from langchain.schema import Document
from src.components.extract_audio import extract_audio
from src.components.llm_model import LLMModel
from src.components.split_document import Splitter
from src.components.transcribe import Transcriber


def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube URL with enhanced regex"""
    youtube_regex = (
        r"(https?://)?(www\.)?"
        r"(youtube|youtu|youtube-nocookie)\.(com|be)/"
        r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    )
    return re.match(youtube_regex, url) is not None


def reset_chat():
    """Completely reset the chat and processing state"""
    keys_to_keep = ["components_loaded"]  # Preserve cached resources
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]


def display_video(file_path=None, youtube_url=None):
    """Safely display video player with error handling"""
    try:
        if youtube_url and is_valid_youtube_url(youtube_url):
            st.video(youtube_url)
        elif file_path and os.path.exists(file_path):
            with open(file_path, "rb") as file:
                video_bytes = file.read()
            st.video(video_bytes)
    except Exception as e:
        st.error(f"Failed to display video: {str(e)}")


@st.cache_resource(show_spinner=False)
def load_components():
    """Load and cache the main components with robust error handling"""
    try:
        with st.spinner("Loading AI components (this may take a minute)..."):
            transcriber = Transcriber()
            splitter = Splitter()
            llm_model = LLMModel()
            st.session_state.components_loaded = True
            return transcriber, splitter, llm_model
    except Exception as e:
        st.error(f"Component initialization failed: {str(e)}")
        st.error(traceback.format_exc())
        st.stop()


def process_video():
    """Process video with comprehensive error handling and cleanup"""
    if not st.session_state.get("video_path"):
        st.error("No video selected for processing")
        return

    try:
        with st.spinner("Processing video (this may take several minutes)..."):
            # Extract audio
            audio_path = None
            try:
                audio_path = extract_audio(st.session_state.video_path)
                print(audio_path)
                if not audio_path or not os.path.exists(audio_path):
                    raise RuntimeError("Audio extraction failed")
            except Exception as e:
                raise RuntimeError(f"Audio extraction error: {str(e)}")

            # Transcribe audio
            try:
                transcript = st.session_state.transcriber.extract_transcript(audio_path)
                if not transcript or not transcript.strip():
                    raise ValueError("Empty transcript received")
                st.session_state.transcript = transcript
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {str(e)}")

            # Prepare document chunks
            try:
                document = Document(
                    page_content=st.session_state.transcript,
                    metadata={"source": st.session_state.get("video_title", "video")},
                )
                chunks = st.session_state.splitter.chunking([document])
                if not chunks:
                    raise ValueError("No chunks generated from document")
                st.session_state.context = chunks
            except Exception as e:
                raise RuntimeError(f"Document processing failed: {str(e)}")

            # Cleanup temporary files
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass

            st.session_state.video_processed = True
            st.success("Video processing completed successfully!")

    except Exception as e:
        st.error(f"Video processing failed: {str(e)}")
        st.session_state.video_processed = False
        # Cleanup on failure
        if "audio_path" in locals() and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


def initialize_session_state():
    """Initialize session state with all required variables"""
    required_state = {
        "messages": [],
        "video_processed": False,
        "transcript": "",
        "context": None,
        "video_path": None,
        "is_youtube": False,
        "video_title": None,
        "processing_error": None,
    }

    for key, default in required_state.items():
        if key not in st.session_state:
            st.session_state[key] = default


def handle_file_upload():
    """Process uploaded file with proper cleanup"""
    if st.session_state.uploaded_file is None:
        return

    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, st.session_state.uploaded_file.name)

        # Save uploaded file
        with open(video_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())

        # Cleanup any previous video
        if st.session_state.get("video_path") and os.path.exists(st.session_state.video_path):
            try:
                os.remove(st.session_state.video_path)
                os.rmdir(os.path.dirname(st.session_state.video_path))
            except:
                pass

        # Update state
        st.session_state.video_path = video_path
        st.session_state.video_title = st.session_state.uploaded_file.name
        st.session_state.is_youtube = False
        st.session_state.video_processed = False
        st.session_state.messages = []

    except Exception as e:
        st.error(f"File upload failed: {str(e)}")
        st.session_state.uploaded_file = None


def main():
    st.set_page_config(
        page_title="Video Chatbot",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Load components (cached)
    if "transcriber" not in st.session_state:
        try:
            (
                st.session_state.transcriber,
                st.session_state.splitter,
                st.session_state.llm_model,
            ) = load_components()
        except:
            st.error("Failed to initialize AI components. Please try again.")
            st.stop()

    # Sidebar for video input
    with st.sidebar:
        st.header("Video Input")

        input_method = st.radio("Choose input method:", ["Upload Video", "YouTube Link"], index=0)

        if input_method == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=["mp4", "mov", "avi", "mkv"],
                key="uploaded_file",
                on_change=handle_file_upload,
            )
        else:
            youtube_url = st.text_input(
                "Enter YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                key="youtube_url",
            )

            if youtube_url:
                if is_valid_youtube_url(youtube_url):
                    st.session_state.video_path = youtube_url
                    st.session_state.is_youtube = True
                    st.session_state.video_title = "YouTube Video"
                    st.session_state.video_processed = False
                    st.session_state.messages = []
                    st.success("YouTube URL ready for processing!")
                else:
                    st.error("Please enter a valid YouTube URL")

        # Process button
        if st.session_state.get("video_path"):
            if st.button(
                "Process Video",
                disabled=st.session_state.get("video_processed", False),
                key="process_btn",
            ):
                process_video()

        # Display transcript if available
        if st.session_state.get("video_processed") and st.session_state.get("transcript"):
            st.subheader("Transcript Preview")
            with st.expander("View Full Transcript"):
                st.text_area(
                    "Transcript",
                    value=st.session_state.transcript,
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                )

    # Main content area
    st.title("🎥 Video Chatbot")
    st.markdown("Chat with any video by uploading or providing a YouTube link")

    # Display video if processed
    if st.session_state.get("video_processed"):
        if st.session_state.get("is_youtube"):
            display_video(youtube_url=st.session_state.video_path)
        else:
            display_video(file_path=st.session_state.video_path)

        # Chat interface
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("Chat with your video")
        with col2:
            st.button("Clear Chat", on_click=reset_chat, key="reset_chat")

        # Display chat history
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me about the video content", key="user_input"):
            if not st.session_state.get("is_processing", False):
                st.session_state.is_processing = True

                try:
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""

                        if not st.session_state.get("context"):
                            raise ValueError("Missing video context. Please reprocess the video.")

                        qa_chain = st.session_state.llm_model.question_answering()
                        response = qa_chain({"question": prompt, "context": st.session_state.context})

                        answer = response.get("answer") if hasattr(response, "get") else str(response)

                        for word in answer.split():
                            full_response += word + " "
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.05)

                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                finally:
                    st.session_state.is_processing = False
                    st.rerun()


if __name__ == "__main__":
    main()
