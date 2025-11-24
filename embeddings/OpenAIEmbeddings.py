from openai import OpenAI


# amit.dixit@inbravo
# Define OpenAIEmbeddings class to encapsulate embedding create methods
# This class will be used by Chroma for embedding documents and queries
# It uses OpenAI's embedding API to generate embeddings for given texts
class OpenAIEmbeddings:
    """
    class that implements two methods to be called from Chroma
    """

    # Constructor to initialize the OpenAIEmbeddings with API key
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    # Method to embed a list of documents
    # Each document is a string of text
    # Returns a list of embeddings corresponding to the input texts
    def embed_documents(self, texts: list[str]):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=text, model="text-embedding-3-small"
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    # Method to embed a single query text
    # Returns the embedding corresponding to the input text
    # Input is a single string of text
    def embed_query(self, text: str):
        response = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
