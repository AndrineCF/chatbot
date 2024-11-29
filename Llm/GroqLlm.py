from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
import torch


class GroqLlm:
    """
    A class for initializing and managing a custom language model (LLM) instance.

    This class handles loading a specified LLM model for chat-based applications,
    setting parameters such as temperature and max tokens, and verifying GPU
    availability for model execution.

    Attributes:
    -----------
    chat_model : BaseChatModel
        The initialized chat model, specifically a ChatGroq model, for handling
        chat-based NLP tasks.
    """

    def __init__(self,
                 model_name: str,
                 temperature: float,
                 max_tokens: int
                 ) -> None:
        """
        Initializes the CustomLlm instance by loading the specified chat model.

        Checks for GPU availability to optimize model execution, loads the ChatGroq model
        with user-defined parameters for response temperature, token limit, and retry attempts.

        Parameters:
        -----------
        model_name : str
            The name or identifier of the model to load.

        temperature : int
            Controls the creativity of the model’s responses. Higher values lead to more varied responses.

        max_tokens : int
            The maximum number of tokens the model can generate in a single response.

        max_retries : int
            The maximum number of retry attempts in case of model failure or timeout.

        Prints:
        -------
        "💻Loading the chat model..." when loading begins.
        "✅ Loaded the chat model" upon successful model loading.
        """

        print("💻Loading the chat model... ")

        # Check if GPU is available
        device = 0 if torch.cuda.is_available() else -1

        # Initialize the chat model with provided parameters
        self.chat_model = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("✅ Loaded the chat model")

    def get_chat_model(self) -> BaseChatModel:
        """
        Retrieves the initialized chat model instance.

        Returns:
        --------
        BaseChatModel
            The ChatGroq model instance loaded for handling chat-based language tasks.
        """
        return self.chat_model
