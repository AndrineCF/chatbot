from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List






class recursiveCharacter:
    """
    A class for handling document loading and chunking.

    This class manages document loaders, sets custom chunk separators,
    and provides functionality for splitting documents into chunks using
    a recursive character-based text splitter.

    Attributes:
    -----------
    document_loaders : List[PyPDFLoader]
        A list to store document loaders.

    document_loader : PyPDFLoader
        A placeholder for a single document loader.

    markdown_separators : List[str]
        A list of strings representing the different types of markdown
        separators used for splitting text into chunks.
    """

    def __init__(self):
        """
        Initializes the DocumentHandler instance with default values.
        """
        self.document_loaders = []
        self.document_loader = None
        self.markdown_separators = [
            "\n#{1,6} ",  # Header separator
            "```\n",  # Code block separator
            "\n\\*\\*\\*+\n",  # Bold text separator
            "\n---+\n",  # Horizontal rule separator
            "\n___+\n",  # Horizontal rule separator
            "\n\n",  # Paragraph separator
            "\n",  # Newline separator
            " ",  # Space separator
            "",  # Empty string separator for fallback
        ]

    def chunking_recursive_character(self,
                                     document: PyPDFLoader,
                                     chuck_size: int = 510,
                                     chuck_overlap: float = 510/2
                                     ) -> List[Document]:
        """
        Splits a document into smaller chunks using a recursive character-based text splitter.

        Parameters:
        -----------
        chuck_size : int
            The maximum number of characters in each chunk.

        chuck_overlap : float
            The number of characters to overlap between chunks.

        document : List[Document]
            A list of Document objects representing the document to be split.

        Returns:
        --------
        List[Document]
            A list of Document chunks after splitting.

        Prints:
        -------
        "{metadata}" where metadata represents the metadata of the document.
        "Splitting document with Recursive Character Text Splitter 📑" when the splitting begins.
        "Documents successfully split into {n} chunks ✅" where n is the number of chunks.
        """
        loader = document.load()
        print(f"{loader[0].metadata}\n")
        print(f'Splitting document with Recursive Character Text Splitter 📑\n')

        # Initialize the RecursiveCharacterTextSplitter with the specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chuck_size,  # The maximum number of characters in a chunk
            chunk_overlap=chuck_overlap,  # The number of characters to overlap between chunks
            strip_whitespace=True,  # Strip whitespace from the start and end of every document
            length_function=len,  # Function to calculate the length of the document
            separators=self.markdown_separators  # Custom markdown separators for chunking
        )

        # Split the document into chunks
        split_doc = text_splitter.split_documents(loader)

        print(f"Documents successfully split into {len(split_doc)} chunks ✅\n")
        return split_doc

