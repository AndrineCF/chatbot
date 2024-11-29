from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing import List
from uuid import uuid4
from Embeddings import SentenceTransformerEmbeddings
import os
import shutil


class DBchroma:
    """
    A class for managing a Chroma-based vector database with custom embeddings.

    This class allows for connecting to a Chroma vector database, adding document chunks,
    and managing the connection to persist or retrieve document embeddings for NLP or
    semantic search tasks.

    Attributes:
    -----------
    vectorStore : VectorStore
        The Chroma vector store used to store and retrieve document embeddings.
    """

    def __init__(self,
                 chroma_path: str,
                 embedding_model: SentenceTransformerEmbeddings,
                 collection_name: str) -> None:
        """
        Initializes a DBchroma instance and connects to the Chroma vector database.

        Parameters:
        -----------
        chroma_path : str
            The file path where the Chroma database will persist data.

        embedding_model : CustomEmbeddings
            A custom embedding model for generating embeddings of the documents.

        collection_name : str
            The name of the collection within the Chroma database to interact with.

        Prints:
        -------
        "Connecting to chroma ☎️" when attempting to establish the connection.
        "Connected to Chroma ✅" when successfully connected to the Chroma database.
        """

        print("Connecting to chroma ☎️")
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        #self.clear_database()
        self.vectorStore = Chroma(
            collection_name=self.collection_name,  # Name of the collection
            embedding_function=self.embedding_model,  # Embedding model to use
            persist_directory=self.chroma_path  # Directory to persist the data
        )

        print("Connected to Chroma ✅")

    def get_collection_name(self):
        return self.collection_name

    def get_vector_store(self):
        return self.vectorStore

    def clear_database(self):
        try:
            # Inform the user that the database clearing process is starting
            print("🧹 Clearing Database...")

            # Check if the specified path exists
            if os.path.exists(self.chroma_path):
                # If it exists, remove the directory and all its contents
                shutil.rmtree(self.chroma_path)

            # Inform the user that the database was cleared successfully
            print("✅ Clearing Database successfully")
        except Exception as e:
            # If an error occurs, print the error message
            print("error: " + str(e))
            # Inform the user that the database clearing process failed
            print("Failed to clear the database ❌")
            

    def add_document_chunks(self, chunks: List[Document] | list[str] | list[list[str] | list[list[str]]]):
        """
        Adds a list of document chunks to the Chroma vector database.

        Each chunk is assigned a unique identifier (UUID) for tracking within the database.

        Parameters:
        -----------
        chunks : List[Document]
            A list of Document objects representing individual document chunks to be stored.
        """
        print("Adding document to database 📨")

        # Iterate over each chunk and add one at a time
        for chunk in chunks:
            # Generate a unique identifier (UUID) for this chunk
            for text_chunk in chunk:  # 'chunk' here is a list, iterate over each text chunk
                uuid = str(uuid4())
                print("Try to add chunk to the database")

                # Pass the string (text_chunk) directly, not a list
                self.vectorStore.add_texts(
                        texts=[text_chunk],  # Pass each chunk as a string, not a list
                        ids=[uuid],          # Use the generated UUID
                        collection_name=self.collection_name,
                        embedding=self.embedding_model,
                        persist_directory=self.chroma_path
                )
                print("Successfully added documents to the database ✅")

    def as_retriever(self, search_type, search_kwargs):
        return self.vectorStore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


                
