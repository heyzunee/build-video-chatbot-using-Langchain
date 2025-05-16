from langchain.text_splitter import RecursiveCharacterTextSplitter


class Splitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunking(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " "]
        )

        chunks = text_splitter.split_documents(documents)
        return chunks
