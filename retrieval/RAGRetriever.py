from langchain_chroma import Chroma
from langchain.schema import Document
from embeddings.EmbeddingFactory import EmbeddingFactory

# RAGRetriever class to handle retrieval-augmented generation
# It interacts with a vector database to fetch relevant documents based on query similarity.
# It uses an embedding model to compute text embeddings for similarity comparison.
# The class provides methods to query the vector database and format the results for downstream use.
# It extracts relevant context and source information from the retrieved documents.
class RAGRetriever:
    """
    A class to handle retrieval-augmented generation (RAG) tasks by querying a vector database
    and formatting the results for downstream use.

    Attributes:
        vector_db_path (str): Path to the vector database directory.
        embedding_function (Callable): Function to generate embeddings for input text.
        db (Chroma): Instance of the Chroma vector database.

    Methods:
        __init__(vector_db_path: str, embedding_model_name: str, api_key: str):
            Initializes the RAGRetriever with the specified vector database path, embedding model, and API key.

        query(query_text: str, k: int = 4) -> list[tuple[Document, float]]:
            Queries the vector database for the most similar documents to the input query text.

        format_results(results: list[tuple[Document, float]]) -> tuple[str, list[str]]:
            Formats the query results into a readable context text and a list of unique sources.

        format_source(metadata: dict) -> str:
            Extracts and formats the source information from the metadata of a document.
    """
    def __init__(self, vector_db_path: str, embedding_model_name: str, api_key: str):
        self.vector_db_path = vector_db_path
        embeddings = EmbeddingFactory(model_name=embedding_model_name, api_key=api_key)
        self.embedding_function = embeddings.create_embedding_function()
        self.db = Chroma(persist_directory=self.vector_db_path, embedding_function=self.embedding_function)

    def query(self, query_text: str, k: int = 4):
        # compute similarity between embeddings of query and of pdf text chunks
        results = self.db.similarity_search_with_score(query_text, k=k)
        return results

    def format_results(self, results: list[tuple[Document, float]]):
        enhanced_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = set(self.format_source(doc.metadata) for doc, _score in results)  # set to ensure uniqueness
        return enhanced_context_text, list(sources)

    def format_source(self, metadata: dict):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        filename = source.split("\\")[-1]  # extract filename
        return f"{filename} page {page}"