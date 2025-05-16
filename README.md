# Build Video Chatbot using Langchain

This is an AI-powered chatbot app built with **Gradio**. The chatbot serves as an assistant, allowing users to ask anything about content insights extracted from videos.


## Features

- Transcribe voice from video/audio using `faster-whisper`
- Embedding model: `infloat/multilingual-e5-small`
- Search knowledge base using `ChromaDB` vector store
- LLM model: `meta-llama/llama-3.1-8b-instruct`
- Simple and intuitive web interface built with `Gradio`



## Tech Stack

- `PyTorch` 1.11.0 with `CUDA 11.3`
- `faster-whisper` for speech recognition
- `transformers` from Hugging Face
- `langchain`, `langchain-community`, `langchain-huggingface`
- `chromadb` for vector search
- `Gradio` for web interface



## Installation with Docker

### 1. Build the Docker image

Make sure you are in the project root directory where the `Dockerfile` is located.

```bash
docker build -t video-chatbot:1.0 .
```

### 2. Run the Docker container with GPU support
```bash
docker run -p 8501:8501 --gpus all video-chatbot:1.0
```



## Access the App

Once the container is running, open your browser and navigate to:

```bash
http://192.168.11.11:8501
```
