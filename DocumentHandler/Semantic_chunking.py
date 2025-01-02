from langchain_community.document_loaders import PyPDFLoader
from typing import List, Tuple, Union, Callable, Dict
from semchunk import chunkerify
import transformers
import tokenizers


def semantic_chunking(
        document: PyPDFLoader,  # The document loader, presumably loads a PDF or other document.
        chunk_size: int,  # The maximum number of tokens per chunk.
        tokenizer_or_token_counter: Union[  # Tokenizer or a callable function to count tokens.
            str,
            transformers.PreTrainedTokenizer,
            tokenizers.Tokenizer,
            Callable[[str], int]  # Function to count tokens in a string.
        ]
) -> list[
    dict[str, list[str] | list[list[str]] | dict]]:  # The function will return a list of dictionaries with chunk data.

    # Load documents from the provided document loader.
    loader = document.load()

    # Create a chunker using the provided tokenizer or token counter with the specified chunk size.
    chunker = chunkerify(
        tokenizer_or_token_counter=tokenizer_or_token_counter,  # The tokenizer or token counting function.
        chunk_size=chunk_size,  # Max number of tokens per chunk.
    )

    # Initialize an empty list to store the chunks with their associated metadata.
    chunks_with_metadata = []

    # Process each page in the loaded document.
    for index, item in enumerate(loader):

        # Check if the page content is empty. If so, skip this page.
        if not item.page_content.strip():
            continue

        # Chunk the page content into smaller parts using the chunker.
        page_chunks = chunker(item.page_content)

        # Create a list containing metadata for the current page along with its chunks.
        page_chunks_with_metadata = [
            {"chunk": page_chunks, "metadata": item.metadata}  # Store chunks and page metadata.
        ]

        # Add the current page's chunks with metadata to the final list.
        chunks_with_metadata.extend(page_chunks_with_metadata)

    # Return the final list containing all chunks with their metadata.
    return chunks_with_metadata

