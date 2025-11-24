import os
import logging
from datetime import datetime
from dotenv import load_dotenv, set_key
from llm.llm_factory import LLMFactory
from retrieval.RAGRetriever import RAGRetriever

# Load environment variables from .env file
load_dotenv()

# amit.dixit@inbravo
# Configuration class to manage the program components, including retriever and LLM model.
# It loads configuration from environment variables, initializes the retriever and LLM model and provides methods to update the configuration dynamically.
class AppConfig:
    """
    AppConfig

    This class is responsible for managing the configuration and initialization of
    components required for a Retrieval-Augmented Generation (RAG) system. It loads
    configuration values from environment variables, initializes the retriever and
    language model (LLM), and provides methods to update the configuration dynamically.

    Attributes:
        retriever (object): The retriever instance used for document retrieval.
        llm_model (object): The language model instance used for generating responses.
        VECTOR_DB_OPENAI_PATH (str): Path to the OpenAI vector database.
        VECTOR_DB_OLLAMA_PATH (str): Path to the Ollama vector database.
        LLM_MODEL_NAME (str): Name of the language model (e.g., 'gpt-3.5-turbo', 'llama3:8b').
        LLM_MODEL_TYPE (str): Type of the language model (e.g., 'ollama', 'gpt', 'claude').
        EMBEDDING_MODEL_NAME (str): Name of the embedding model (e.g., 'openai', 'ollama').
        NUM_RELEVANT_DOCS (int): Number of relevant documents to retrieve.
        OPENAI_API_KEY (str): API key for OpenAI services.
        CLAUDE_API_KEY (str): API key for Claude services.
        ENV_PATH (str): Path to the environment configuration file.

    Methods:
        get_vector_db_path(embedding_model_name: str) -> str:
            Determines and returns the vector database path based on the embedding model name.

        initialize_components():
            Initializes the retriever and language model components based on the current configuration.

        update_components(
            OPENAI_API_KEY: str
            Updates the configuration values in the environment file and reinitializes the components.
    """
    # Constructor
    def __init__(self):
        pass

    rag_retriever = None
    llm_model = None
    logger = None

    # Load configuration from environment variables
    ENV_PATH = ".env"

    # The path where all the documents are available on local system
    DATA_PATH = os.getenv("DATA_PATH")
    LOG_DIR = os.getenv("LOG_DIR")

    # The path to the vector DBs for different embedding models
    VECTOR_DB_OPENAI_PATH = os.getenv("VECTOR_DB_OPENAI_PATH")
    VECTOR_DB_OLLAMA_PATH = os.getenv("VECTOR_DB_OLLAMA_PATH")

    # 'gpt-3.5-turbo', 'GPT-4o' or local LLM like 'llama3:8b', 'gemma2', 'mistral:7b' etc.
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
    LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    NUM_RELEVANT_DOCS = int(os.getenv("NUM_RELEVANT_DOCS"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

    # Initialize RAG components: retriever and LLM model
    def initialize_components():

        # Setup logging
        logger = AppConfig.setup_logging()
        logger.info("=== Application Initialized ===")

        # Select the appropriate API key based on the embedding model
        # No API key is required for Ollama embeddings
        if AppConfig.EMBEDDING_MODEL_NAME == "openai":
            api_key = AppConfig.OPENAI_API_KEY
        else:
            api_key = AppConfig.CLAUDE_API_KEY

        # Initialize RAG retriever based on embedding model name
        AppConfig.rag_retriever = RAGRetriever(
            vector_db_path=AppConfig.get_vector_db_path(AppConfig.EMBEDDING_MODEL_NAME),
            embedding_model_name=AppConfig.EMBEDDING_MODEL_NAME,
            api_key=api_key
        )

        # Initialize LLM model based on embedding model name
        AppConfig.llm_model = LLMFactory.create_llm(
            model_type=AppConfig.LLM_MODEL_TYPE,
            model_name=AppConfig.LLM_MODEL_NAME,
            api_key=api_key
        )

        # Log the LLM details
        print(
            f"Instantiating model type: {AppConfig.LLM_MODEL_TYPE} | model name: {AppConfig.LLM_MODEL_NAME} | embedding model: {AppConfig.EMBEDDING_MODEL_NAME}"
        )

    # Update configuration components
    def update_components(
        LLM_MODEL_NAME: str,
        LLM_MODEL_TYPE: str,
        EMBEDDING_MODEL_NAME: str,
        NUM_RELEVANT_DOCS: str,
        OPENAI_API_KEY: str
    ):

        # Update the .env file
        set_key(AppConfig.ENV_PATH, "LLM_MODEL_NAME", LLM_MODEL_NAME)
        set_key(AppConfig.ENV_PATH, "LLM_MODEL_TYPE", LLM_MODEL_TYPE)
        set_key(AppConfig.ENV_PATH, "EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)
        set_key(AppConfig.ENV_PATH, "NUM_RELEVANT_DOCS", str(NUM_RELEVANT_DOCS))
        set_key(AppConfig.ENV_PATH, "OPENAI_API_KEY", OPENAI_API_KEY)

        # Reinitialize the components (llm and retriever objects)
        AppConfig.initialize_components()
        print(
            f"Updating model type: {LLM_MODEL_TYPE} | model name: {LLM_MODEL_NAME} | embedding model: {EMBEDDING_MODEL_NAME}"
        )

    # Get vector DB path based on embedding model name
    def get_vector_db_path(embedding_model_name):
        """
        Retrieve the file path for the vector database based on the specified embedding model name.
        Args:
            embedding_model_name (str): The name of the embedding model.
                Supported values are "openai" and "ollama".
        Returns:
            str: The file path to the vector database corresponding to the embedding model.
        Raises:
            ValueError: If the provided embedding model name is not supported.
        """

        # Determine vector DB path based on embedding model name
        if embedding_model_name == "openai":
            return AppConfig.VECTOR_DB_OPENAI_PATH
        elif embedding_model_name == "ollama":
            return AppConfig.VECTOR_DB_OLLAMA_PATH
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model_name}")

    # Setup logging configuration
    @staticmethod
    def setup_logging():
        """Configure logging for the database population process."""

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), AppConfig.LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'rag-bot_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Logging initialized. Log file: %s at location: %s", log_file, log_dir)
        return logger

    # Get logging configuration (if the logger is already set up)
    def get_default_logger(name:str=None):

        # Return the default logger instance if already set up
        if AppConfig.logger is None:
            AppConfig.logger = AppConfig.setup_logging()
        # Return named logger or default logger
        return logging.getLogger(name) if name else AppConfig.logger