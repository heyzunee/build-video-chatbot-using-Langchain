import logging

import torch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from ..config import *
from .create_db import VectorDB

logger = logging.getLogger(PROJECT_NAME)


class LLMModel:
    def __init__(self, model_name="./models/meta-llama/llama-3.1-8b-instruct"):
        self.db = VectorDB()
        self.model_name = model_name
        if DEVICE == "cpu":
            self.quantization_config = None
        else:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.load_model()
        self.query_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=200,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )
        self.llm = HuggingFacePipeline(
            pipeline=self.query_pipeline,
        )

    def load_model(self):
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            quantization_config=self.quantization_config,
            device_map="auto" if DEVICE == "cuda" else None,
        )

    def qa_chain(self):
        logger.info("Setting up QA chain...")
        # Define the prompt template for RetrievalQA
        self.prompt_template = (
            "You are a smartly assistant.\n\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Based on the context above, give a short and concise answer to the question.\n"
            "Do not make up information. End the answer with 'Thanks for asking.'\n"
            "Question: {question}\n"
        )
        qa_prompt = PromptTemplate.from_template(self.prompt_template)

        # Create QA chain with custom prompt template
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.db.retrieve(),
                chain_type_kwargs={"prompt": qa_prompt},
            )
            return qa_chain
        except Exception as e:
            logger.error(f"Failed to set up QA chain: {str(e)}")
