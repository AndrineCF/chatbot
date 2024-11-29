from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from Database import DBchroma

class Retrieval:
    def __init__(self, 
                 db: DBchroma
                 ) -> None:
        print("Creating a retrieve 🔎")
        print(db.get_vector_store())

        # Set up semantic similarity-based retrievers
        self.similarity_score_threshold = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.75}
        )
        
        self.mmr = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 50}
        )

        # Combine all retrievers in an EnsembleRetriever with weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.mmr,
                self.similarity_score_threshold,
            ],
            weights=[0.5, 0.5]  # Adjust weights as needed
        )
        
        print("Successfully created a retriever ✅")

    def get_retriever(self) -> BaseRetriever:
        return self.ensemble_retriever
