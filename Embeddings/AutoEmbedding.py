from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings.base import Embeddings
from typing import List
import torch

class AutoEmbedding(Embeddings):
    def __init__(self, model_name: str, stored_path_model: str) -> None:
        """
        Initialize the AutoEmbedding class with AutoTokenizer and AutoModelForCausalLM model.

        Args:
            model_name (str): The name of the model.
            stored_path_model (str): Path to store the downloaded model.
        """
        print("💻 Loading the embeddings model... ")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=stored_path_model)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=stored_path_model).to(self.device)
        print("✅ Loaded the embeddings model")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = []
        
        for doc in documents:
            # Tokenize and get hidden states
            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Pool hidden states to get a single embedding
            last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)
            embedding = last_hidden_state.mean(dim=1).squeeze().tolist()
            embeddings.append(embedding)
        
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query using the model's hidden states.

        Args:
            query (str): The query string to be embedded.

        Returns:
            List[float]: The embedding of the query as a list of floats.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # shape: (1, sequence_length, hidden_size)
        embedding = last_hidden_state.mean(dim=1).squeeze().tolist()  # Average pooling
        return embedding
