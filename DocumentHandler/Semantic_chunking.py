from langchain_community.document_loaders import PyPDFLoader
from typing import Callable, List, Union

import semchunk
import transformers
import tokenizers

def semantic_chunking(
    document: PyPDFLoader,
    tokenizer_or_token_counter: Union[
        str, transformers.PreTrainedTokenizer, tokenizers.Tokenizer, Callable[[str], int]
    ]
) -> List[List[str]]:
    # Load documents
    loader = document.load()

    print(f"Metadata of the first document: {loader[0].metadata}\n")
    print('Splitting document with semantic text Splitter 📑\n')

    # Initialize the chunker
    chunker = semchunk.chunkerify(tokenizer_or_token_counter, chunk_size=512)

    chunks = []
    for index, item in enumerate(loader):
        # Split page content into chunks and preserve metadata
        page_chunks = chunker(item.page_content, 512)
        print(f"Chunks for page {item.metadata['page'], item.metadata['source']}\n")
        chunks.append(page_chunks)

    return chunks