from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from Embeddings import SentenceTransformerEmbeddings

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

        vector_store = PGVector(
            embeddings=embedding_model,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        print("Connected to postgres database ✅")