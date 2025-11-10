from embeddings.OpenAIEmbeddings import OpenAIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

# Embeddings class to select and return the appropriate embedding function based on the specified model name.
# It supports Ollama, OpenAI, and Bedrock embedding models.
class EmbeddingFactory:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    def create_embedding_function(self):
        if self.model_name == "ollama":
            return OllamaEmbeddings(model="mxbai-embed-large")
        elif self.model_name == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided for OpenAI embeddings")
            return OpenAIEmbeddings(api_key=self.api_key)
        elif self.model_name == "bedrock":
            return BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")