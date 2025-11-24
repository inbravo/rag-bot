from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

from embeddings.OpenAIEmbeddings import OpenAIEmbeddings

# amit.dixit@inbravo
# Embeddings class to select and return the appropriate embedding function based on the specified model name.
# It supports Ollama, OpenAI, and Bedrock embedding models.
class EmbeddingFactory:
    """
    EmbeddingFactory is a factory class responsible for creating embedding functions
    based on the specified model name. It supports multiple embedding models, including
    Ollama, OpenAI, and Bedrock.

    Attributes:
        model_name (str): The name of the embedding model to use.
        api_key (str, optional): The API key required for certain embedding models
                                 (e.g., OpenAI).

    Methods:
        create_embedding_function():
            Creates and returns the appropriate embedding function based on the
            specified model name.


                ValueError: If the model name is unsupported or if the OpenAI API key is not
                            provided when using the OpenAI embedding model.
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
    """
    # Constructor to initialize the EmbeddingFactory with model name and optional API key
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    # Create and return the appropriate embedding function based on the model name
    def create_embedding_function(self):
        """
        Creates and returns the appropriate embedding function based on the specified model name.

        Returns:
            An instance of the embedding class corresponding to the specified model name.

        Raises:
            ValueError: If the model name is unsupported or if the OpenAI API key is not provided
                        when using the OpenAI embedding model.
        """
        # Select and return the appropriate embedding function
        if self.model_name == "ollama":
            return OllamaEmbeddings(model="mxbai-embed-large")
        elif self.model_name == "openai":
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key must be provided for OpenAI embeddings"
                )
            return OpenAIEmbeddings(api_key=self.api_key)
        elif self.model_name == "bedrock":
            return BedrockEmbeddings(
                credentials_profile_name="default", region_name="us-east-1"
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")
