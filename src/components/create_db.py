import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from ..config import PROJECT_NAME

logger = logging.getLogger(PROJECT_NAME)


class VectorDB:
    def __init__(
        self,
        model_name="./models/infloat/multilingual-e5-small",
        persist_directory="./chroma_db",
    ):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def create_db(self, chunks):
        logger.info("Creating vectorstore...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        logger.info(
            "Number of documents in vectorstore:", self.vectorstore._collection.count()
        )

    def retrieve(self):
        return self.vectorstore.as_retriever()
