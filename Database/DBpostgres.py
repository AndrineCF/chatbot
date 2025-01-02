from langchain_core.documents import Document
from sqlalchemy import create_engine
from langchain_postgres.vectorstores import PGVector
from Embeddings import SentenceTransformerEmbeddings
from uuid import uuid4
from typing import List
import pandas as pd

class DBpostgres:
    def __init__(self,
                 connection: str,
                 embedding_model: SentenceTransformerEmbeddings,
                 collection_name: str
                 ) -> None:
        print("Connecting to postgres database... ☎️")
        self.connection = connection
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Create the database engine
        engine = create_engine(connection)

        self.vector_store = PGVector(
            embeddings=embedding_model,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        print("Connected to postgres database ✅")

    def get_collection_name(self):
        return self.collection_name

    def get_vector_store(self):
        return self.vector_store

    def add_document_chunks(self, chunks: List[Document] | list[list[dict[str, str | dict]]]):
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
                self.vector_store.add_texts(
                        texts=[text_chunk],  # Pass each chunk as a string, not a list
                        ids=[uuid],          # Use the generated UUID
                        collection_name=self.collection_name,
                        embedding=self.embedding_model,
                )
                print("Successfully added documents to the database ✅")

    def add_document_semantic_chunking(self, chunks: list[list[dict[str, str | dict]]]):
        """
        Adds a list of document chunks to the Chroma vector database.

        Each chunk is assigned a unique identifier (UUID) for tracking within the database.

        Parameters:
        -----------
        chunks : List[Document]
            A list of Document objects representing individual document chunks to be stored.
        """
        print("Adding document to database 📨")

        # Iterate over each chunk
        for chunk_list in chunks:
            for chunk_data in chunk_list:
                text_chunk = chunk_data.get('chunk')
                metadata = chunk_data.get('metadata')

                if text_chunk and metadata:
                    # Generate a unique identifier (UUID) for this chunk
                    uuid = str(uuid4())

                    # Pass the text chunk and its metadata directly to the vector store
                    self.vector_store.add_texts(
                        texts=[text_chunk],  # The chunk text
                        ids=[uuid],  # The generated UUID
                        collection_name=self.collection_name,  # Collection to store the chunks
                        embedding=self.embedding_model,  # The embedding model to use
                        metadatas=[metadata]  # Attach metadata to the chunk
                    )
                else:
                    print(f"Skipping invalid chunk: {chunk_data}")

    def as_retriever(self, search_type, search_kwargs):
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
