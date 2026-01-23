import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

MODEL_NAME = "BAAI/bge-m3"


class HFHostedEmbeddings(Embeddings):
    """LangChain-compatible embeddings using HF Inference API"""

    def embed_query(self, text: str):
        return hf_client.feature_extraction(text, model=MODEL_NAME)

    def embed_documents(self, texts: list[str]):
        return [hf_client.feature_extraction(t, model=MODEL_NAME) for t in texts]

embeddings = HFHostedEmbeddings()
