import os

from langchain_community.document_loaders import PyPDFLoader


def read_folder(folder_path: str) -> list[PyPDFLoader]:
    """
    Loads all PDF files from a specified folder and creates a list of PDF document loaders.

    This function iterates through each file in the specified directory, checks if the file
    is a PDF, and loads it using the PyPDFLoader.

    Parameters:
    -----------
    folder_path : str
        The path to the folder containing PDF files.

    Returns:
    --------
    List[PyPDFLoader]
        A list of PyPDFLoader objects, each representing a loaded PDF document.

    Prints:
    -------
    "Loading the documents 📂" when loading begins.
    "{n} Documents loaded ✅" where n is the number of PDF documents successfully loaded.
    """
    print("Loading the documents 📂")

    # Initialize an empty list to hold document loaders
    documents = []

    # Iterate over all files in the specified directory
    for filename in os.listdir(folder_path):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Create a document loader for the PDF file
            document = PyPDFLoader(file_path)
            # Add the document loader to the list
            documents.append(document)

    # Inform the user how many documents were loaded
    print(f"{len(documents)} Documents loaded ✅")
    return documents