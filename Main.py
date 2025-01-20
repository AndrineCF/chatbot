from Embeddings import SentenceTransformerEmbeddings
from Document_reader import read_folder
from DocumentHandler import semantic_chunking, recursiveCharacter
from Database import DBpostgres
from App import App
import argparse
from dotenv import load_dotenv
import os


# GLOBAL VARIABLES
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
DOCUMENT_FOLDER = "data"
CHUNK_SIZE = 510
CHROMA_PATH = "kartverketDB"
COLLECTION_NAME = "produktspesifikasjoner"
MODEL_TYPE = "groq"
MODEL_NAME = "llama-3.1-70b-versatile"

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# Access the environment variables
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
CONNECTION_STRING = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
CUSTOM_CACHE_DIR = os.getenv("CUSTOM_CACHE_DIR")

def setup_backend():
    # setup all the class
    embedding = SentenceTransformerEmbeddings(EMBEDDING_MODEL_NAME, CUSTOM_CACHE_DIR)
    
    db = DBpostgres(CONNECTION_STRING,embedding, COLLECTION_NAME)
    loaders = read_folder(DOCUMENT_FOLDER)

    # The tokenizer is accessible via the model's `tokenizer` attribute
    tokenizer = embedding.get_model().tokenizer
    tokenizer.model_max_length = 1024
    tokenizer.padding="max_length"
    tokenizer.truncation=True
    tokenizer.return_tensors="pt"
    
    # Spilt the documents into chucks and add to the database
    for loader in loaders:
        chunks = semantic_chunking(loader, CHUNK_SIZE, tokenizer)
        db.add_document_semantic_chunking(chunks)

if __name__ == '__main__':
    setup_backend()

    #app = App(CONNECTION_STRING, EMBEDDING_MODEL_NAME, MODEL_NAME, COLLECTION_NAME, CUSTOM_CACHE_DIR)
    #app.initialize()
