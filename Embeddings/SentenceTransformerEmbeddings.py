from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
import torch

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self,
                 model_name: str,
                 stored_path_model: str)-> None:
        """
        Initialize the CustomEmbeddings class with a specified SentenceTransformer model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
            stored_path_model (str): path to where store downloaded SentenceTransformer model.
        """
        print("💻Loading the embeddings model... ")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, cache_folder=stored_path_model, device=self.device)
        print("✅ Loaded the embeddings model")

    def get_model(self):
        return self.model

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the SentenceTransformer model.

        Args:
            documents (List[str]): A list of documents (strings) to be embedded.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """

        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query using the SentenceTransformer model.

        Args:
            query (str): The query string to be embedded.

        Returns:
            List[float]: The embedding of the query as a list of floats.
        """

        return self.model.encode([query])[0].tolist()
