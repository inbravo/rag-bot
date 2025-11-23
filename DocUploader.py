import argparse
import os
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFDirectoryLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AppConfig import AppConfig
from embeddings.EmbeddingFactory import EmbeddingFactory

# Initial components
AppConfig.initialize_components()

# Main function to manage the population of the database with embeddings.
def main():
    """
    Main function to manage the population of the database with embeddings.

    This function provides the following functionalities:
    - Resets the database based on the `--reset` flag.
    - Chooses the embedding model to use (`openai` or `ollama`) via the `--embedding-model` argument.
    - Loads the existing database or creates a new one based on the specified embedding model.
    - Processes documents by splitting them into chunks and adding them to the database.

    Command-line Arguments:
    - --reset: Optional flag to reset the database. Accepts values "ollama", "openai", or "both".
    - --embedding-model: Specifies the embedding model to use. Defaults to "ollama".

    Raises:
    - ValueError: If an unsupported embedding model is specified.
    """
    # Setup logging
    logger = logger = AppConfig.get_default_logger()

    try:
        logger.info("=== Database Population Script Started ===")

        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--reset",
            nargs="?",
            const="ollama",
            choices=["ollama", "openai", "both"],
            help="Reset the database.",
        )
        parser.add_argument(
            "--embedding-model",
            type=str,
            default="ollama",
            help="The embedding model to use (ollama or openai).",
        )
        args = parser.parse_args()

        logger.info(
            f"Command line arguments - reset: {args.reset}, embedding_model: {args.embedding_model}"
        )

        if args.reset:
            logger.info(f"Reset flag detected - resetting {args.reset} database(s)")
            reset_databases(args.reset)
            return

        # Choose the embedding model
        logger.info(f"Initializing embeddings with model: {args.embedding_model}")

        # Create the instance of embedding factory
        embeddings = EmbeddingFactory(
            model_name=args.embedding_model, api_key=AppConfig.OPENAI_API_KEY
        )

        # Call the embedding function from the factory
        embedding_function = embeddings.create_embedding_function()
        logger.info("Embedding function created successfully")

        # Determine the correct path for the database based on the embedding model
        if args.embedding_model == "openai":
            print("Using OpenAI embeddings")
            logger.info("Using OpenAI embeddings")
            db_path = AppConfig.VECTOR_DB_OPENAI_PATH
        elif args.embedding_model == "ollama":
            print("Using Ollama embeddings")
            logger.info("Using Ollama embeddings")
            db_path = AppConfig.VECTOR_DB_OLLAMA_PATH
        else:
            error_msg = f"Unsupported embedding model specified: {args.embedding_model}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Database path determined: {db_path}")

        # Load the existing database
        logger.info("Loading existing Chroma database")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        logger.info("Database loaded successfully")

        # Create (or update) the data store
        logger.info("Starting document processing pipeline")
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks, db)

        logger.info("=== Database Population Script Completed Successfully ===")

    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        print("\n⚠ Script interrupted by user")
    except Exception as e:
        logger.critical(f"Unexpected error in main execution: {e}", exc_info=True)
        print(f"💥 Critical error: {e}")
        raise

# Reset and rebuild databases based on user choice
# reset_choice can be "openai", "ollama", or "both"
def reset_databases(reset_choice):
    """
    Resets and rebuilds the specified databases based on the user's choice.

    Parameters:
        reset_choice (str): Specifies which database(s) to reset.
                            Acceptable values are:
                            - "openai": Resets the OpenAI database.
                            - "ollama": Resets the Ollama database.
                            - "both": Resets both the OpenAI and Ollama databases.
    """
    logger = AppConfig.get_default_logger()
    logger.info(f"Starting database reset process for: {reset_choice}")

    if reset_choice in ["openai", "both"]:
        logger.info("Processing OpenAI database reset")
        if ask_to_clear_database("openai"):
            print("✨ Rebuilding OpenAI Database")
            logger.info("User confirmed OpenAI database reset")
            clear_database("openai")
            rebuild_database("openai")
        else:
            logger.info("User cancelled OpenAI database reset")

    if reset_choice in ["ollama", "both"]:
        logger.info("Processing Ollama database reset")
        if ask_to_clear_database("ollama"):
            print("✨ Rebuilding Ollama Database")
            logger.info("User confirmed Ollama database reset")
            clear_database("ollama")
            rebuild_database("ollama")
        else:
            logger.info("User cancelled Ollama database reset")

# Rebuild the specified database
# embedding_model can be "openai" or "ollama"
def ask_to_clear_database(embedding_model):
    """
    Prompts the user to confirm whether they want to override the existing database for the specified embedding model.

    Args:
        embedding_model (str): The name of the embedding model whose database is being queried.

    Returns:
        bool: True if the user confirms by entering 'yes', False otherwise.
    """
    logger = AppConfig.get_default_logger()
    logger.info(f"Prompting user for confirmation to clear {embedding_model} database")

    response = (
        input(
            f"Do you want to override the existing {embedding_model} database? (yes/no): "
        )
        .strip()
        .lower()
    )
    logger.info(f"User response for {embedding_model} database clear: {response}")

    return response == "yes"

# Load all the user supplied documents in the 
def load_documents():
    """
    Loads documents from a specified directory containing multiple file types.
    Recursively searches subdirectories as well.

    Supports: PDF, DOCX, XLSX, CSV files

    Returns:
        List[Document]: A list of Document objects loaded from various file types.
    """
    logger = AppConfig.get_default_logger()
    logger.info("Loading documents from directory: %s", AppConfig.DATA_PATH)

    try:
        if not os.path.exists(AppConfig.DATA_PATH):
            error_msg = f"Data path does not exist: {AppConfig.DATA_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        all_documents = []
        file_counts = {"pdf": 0, "docx": 0, "xlsx": 0, "csv": 0}

        # Define file type loaders with their extensions
        loaders_config = {
            ".pdf": (PyPDFDirectoryLoader, "pdf"),
            ".docx": (UnstructuredWordDocumentLoader, "docx"),
            ".doc": (
                UnstructuredWordDocumentLoader,
                "docx",
            ),  # Treat .doc same as .docx
            ".xlsx": (UnstructuredExcelLoader, "xlsx"),
            ".xls": (UnstructuredExcelLoader, "xlsx"),  # Treat .xls same as .xlsx
            ".csv": (CSVLoader, "csv"),
        }

        # Get all files in the directory and subdirectories
        data_path = Path(AppConfig.DATA_PATH)
        logger.info("Scanning directory and subdirectories for supported file types")

        # Process each file type with recursive search
        for file_extension, (loader_class, file_type) in loaders_config.items():
            # Find files with this extension recursively using **/ pattern
            files = list(data_path.rglob(f"*{file_extension}"))

            if not files:
                logger.debug("No %s files found in directory tree", file_extension)
                continue

            logger.info(
                "Found %d %s files in directory tree", len(files), file_extension
            )

            # Log the directories where files were found
            directories = set(file_path.parent for file_path in files)
            logger.info("Files found in directories: %s", [str(d) for d in directories])

            # Load documents based on file type
            if file_extension == ".pdf":
                # For PDFs, we need to handle recursive loading differently
                logger.info("Loading PDF documents recursively")

                # Get all subdirectories that contain PDF files
                pdf_directories = set(file_path.parent for file_path in files)

                for pdf_dir in pdf_directories:
                    try:
                        logger.info("Loading PDFs from directory: %s", pdf_dir)
                        pdf_loader = PyPDFDirectoryLoader(str(pdf_dir))
                        pdf_documents = pdf_loader.load()

                        # Add directory information to metadata
                        for doc in pdf_documents:
                            doc.metadata["subdirectory"] = str(
                                pdf_dir.relative_to(data_path)
                            )
                            doc.metadata["file_type"] = "pdf"

                        all_documents.extend(pdf_documents)
                        file_counts["pdf"] += len(pdf_documents)
                        logger.info(
                            "Loaded %d PDF documents from %s",
                            len(pdf_documents),
                            pdf_dir,
                        )

                    except Exception as e:
                        logger.warning(
                            "Failed to load PDFs from directory %s: %s", pdf_dir, e
                        )

            elif file_extension == ".csv":
                # For CSV files, load each individually with proper encoding
                for csv_file in files:
                    try:
                        logger.info(
                            "Loading CSV file: %s", csv_file.relative_to(data_path)
                        )
                        csv_loader = CSVLoader(
                            file_path=str(csv_file),
                            encoding="utf-8",
                            csv_args={"delimiter": ","},
                        )
                        csv_documents = csv_loader.load()

                        # Add file information to metadata
                        for doc in csv_documents:
                            doc.metadata["source"] = str(csv_file)
                            doc.metadata["file_type"] = "csv"
                            doc.metadata["subdirectory"] = str(
                                csv_file.parent.relative_to(data_path)
                            )

                        all_documents.extend(csv_documents)
                        file_counts["csv"] += len(csv_documents)
                        logger.info(
                            "Loaded %d rows from CSV: %s",
                            len(csv_documents),
                            csv_file.relative_to(data_path),
                        )

                    except Exception as e:
                        logger.warning(
                            "Failed to load CSV file %s: %s",
                            csv_file.relative_to(data_path),
                            e,
                        )

            else:
                # For DOCX and XLSX files, load each individually
                for file_path in files:
                    try:
                        logger.info(
                            "Loading %s file: %s",
                            file_type.upper(),
                            file_path.relative_to(data_path),
                        )

                        loader = None
                        if file_extension in [".docx", ".doc"]:
                            loader = UnstructuredWordDocumentLoader(str(file_path))
                        elif file_extension in [".xlsx", ".xls"]:
                            loader = UnstructuredExcelLoader(str(file_path))

                        if loader is None:
                            logger.warning(
                                "No loader found for file extension: %s", file_extension
                            )
                            continue

                        # Load the documents
                        documents = loader.load()

                        # Add file information to metadata
                        for doc in documents:
                            doc.metadata["source"] = str(file_path)
                            doc.metadata["file_type"] = file_type
                            doc.metadata["subdirectory"] = str(
                                file_path.parent.relative_to(data_path)
                            )

                        all_documents.extend(documents)
                        file_counts[file_type] += len(documents)
                        logger.info(
                            "Loaded %d sections from %s: %s",
                            len(documents),
                            file_type.upper(),
                            file_path.relative_to(data_path),
                        )

                    except Exception as e:
                        logger.warning(
                            "Failed to load %s file %s: %s",
                            file_type.upper(),
                            file_path.relative_to(data_path),
                            e,
                        )

        # Log summary
        total_documents = len(all_documents)
        logger.info("Document loading completed - Total documents: %d", total_documents)
        logger.info("File type breakdown: %s", file_counts)

        print(
            f"📄 Loaded {total_documents} documents from various file types (including subdirectories):"
        )
        for file_type, count in file_counts.items():
            if count > 0:
                print(f"   📎 {file_type.upper()}: {count} documents")

        # Log document sources and subdirectories for debugging
        if all_documents:
            sources = set(
                doc.metadata.get("source", "Unknown") for doc in all_documents
            )
            logger.info("Document sources (first 10): %s", list(sources)[:10])

            # Log subdirectories found
            subdirectories = set(
                doc.metadata.get("subdirectory", ".") for doc in all_documents
            )
            logger.info("Subdirectories processed: %s", list(subdirectories))

            # Log file types
            file_types = set(
                doc.metadata.get("file_type", "Unknown") for doc in all_documents
            )
            logger.info("Document file types: %s", list(file_types))

        if not all_documents:
            logger.warning("No documents were loaded from any supported file types")
            print("⚠️ No supported documents found in the directory tree")

        return all_documents

    except Exception as e:
        logger.error("Error loading documents: %s", e, exc_info=True)
        print(f"❌ Error loading documents: {e}")
        raise

# Validate the data directory and report on available files recursively
# Also reports which subdirectories contain which file types
# Returns a dictionary with file type counts and subdirectory info
# Returns:
#     dict: Dictionary with file type counts and subdirectory informationß
def validate_data_directory():
    """
    Validates the data directory and reports on available files recursively.

    Returns:
        dict: Dictionary with file type counts and subdirectory information
    """
    logger = AppConfig.get_default_logger()

    if not os.path.exists(AppConfig.DATA_PATH):
        logger.error("Data directory does not exist: %s", AppConfig.DATA_PATH)
        return {}

    data_path = Path(AppConfig.DATA_PATH)
    supported_extensions = get_supported_file_types()
    file_counts = {}
    subdirectory_info = {}

    logger.info("Validating data directory recursively: %s", AppConfig.DATA_PATH)

    for ext in supported_extensions:
        # Use rglob for recursive search
        files = list(data_path.rglob(f"*{ext}"))
        file_counts[ext] = len(files)

        if files:
            logger.info("Found %d %s files recursively", len(files), ext)
            # Track which subdirectories contain files
            directories = set(str(f.parent.relative_to(data_path)) for f in files)
            subdirectory_info[ext] = list(directories)
            logger.info("%s files found in subdirectories: %s", ext, directories)

    total_files = sum(file_counts.values())
    logger.info("Total supported files found recursively: %d", total_files)

    # Log summary of directory structure
    all_subdirs = set()
    for dirs in subdirectory_info.values():
        all_subdirs.update(dirs)

    logger.info("All subdirectories with supported files: %s", list(all_subdirs))

    return {
        "file_counts": file_counts,
        "subdirectory_info": subdirectory_info,
        "all_subdirectories": list(all_subdirs),
    }


def get_supported_file_types():
    """
    Returns a list of supported file types for document loading.

    Returns:
        list: List of supported file extensions
    """
    return [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv"]

# Analyze and return the directory structure for debugging
# Returns:
#     dict: Directory structure informationß
def get_directory_structure():
    """
    Returns the directory structure of the data path for debugging.

    Returns:
        dict: Directory structure information
    """
    logger = AppConfig.get_default_logger()

    if not os.path.exists(AppConfig.DATA_PATH):
        logger.error("Data directory does not exist: %s", AppConfig.DATA_PATH)
        return {}

    data_path = Path(AppConfig.DATA_PATH)
    structure = {}

    try:
        # Get all directories recursively
        all_dirs = [d for d in data_path.rglob("*") if d.is_dir()]

        # Get all files recursively
        all_files = [f for f in data_path.rglob("*") if f.is_file()]

        structure = {
            "total_directories": len(all_dirs),
            "total_files": len(all_files),
            "directories": [
                str(d.relative_to(data_path)) for d in all_dirs[:20]
            ],  # First 20 dirs
            "sample_files": [
                str(f.relative_to(data_path)) for f in all_files[:20]
            ],  # First 20 files
        }

        logger.info(
            "Directory structure - Total dirs: %d, Total files: %d",
            len(all_dirs),
            len(all_files),
        )

    except Exception as e:
        logger.error("Error analyzing directory structure: %s", e)

    return structure

# Split documents into smaller chunks
# Returns a list of Document chunksß
def split_documents(documents: list[Document]):
    """
    Splits a list of documents into smaller chunks using a RecursiveCharacterTextSplitter.

    Args:
        documents (list[Document]): A list of Document objects to be split.

    Returns:
        list[Document]: A list of smaller Document chunks created by splitting the input documents.
    """
    logger = AppConfig.get_default_logger()
    logger.info(f"Starting document splitting process for {len(documents)} documents")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info("Text splitter configured - chunk_size: 800, chunk_overlap: 80")

        chunks = text_splitter.split_documents(documents)

        logger.info(f"Document splitting completed - created {len(chunks)} chunks")
        print(f"✂️ Split documents into {len(chunks)} chunks")

        # Log statistics about chunk sizes
        if chunks:
            chunk_lengths = [len(chunk.page_content) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            logger.info(
                f"Chunk statistics - avg_length: {avg_length:.0f}, min: {min(chunk_lengths)}, max: {max(chunk_lengths)}"
            )

        return chunks

    except Exception as e:
        logger.error(f"Error splitting documents: {e}", exc_info=True)
        print(f"❌ Error splitting documents: {e}")
        raise

# Add or update document chunks in the Chroma databaseß
def add_to_chroma(chunks: list[Document], db):
    """
    Adds or updates a list of document chunks in the Chroma database.

    Args:
        chunks (list[Document]): A list of document chunks to be added to the database.
        db: The Chroma database instance where the documents will be stored.
    """
    logger = AppConfig.get_default_logger()
    logger.info(f"Starting process to add {len(chunks)} chunks to Chroma database")

    try:
        # Calculate Page IDs
        logger.info("Calculating chunk IDs")
        chunks_with_ids = calculate_chunk_ids(chunks)
        logger.info("Chunk IDs calculated successfully")

        # Add or Update the documents
        logger.info("Retrieving existing document IDs from database")
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        logger.info(f"Found {len(existing_ids)} existing documents in database")
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB
        logger.info("Filtering out existing documents")
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        logger.info(f"Identified {len(new_chunks)} new chunks to add")

        if len(new_chunks):
            logger.info(f"Adding {len(new_chunks)} new documents to database")
            print(f"➕ Adding new documents: {len(new_chunks)}")

            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)

            logger.info("Documents added successfully to Chroma database")

            # Log some sample IDs for verification
            sample_ids = new_chunk_ids[:5]
            logger.info(f"Sample new document IDs: {sample_ids}")
        else:
            logger.info(
                "No new documents to add - all chunks already exist in database"
            )
            print("✅ No new documents to add")

    except Exception as e:
        logger.error(f"Error adding documents to Chroma: {e}", exc_info=True)
        print(f"❌ Error adding documents to database: {e}")
        raise

# Calculate unique chunk IDs based on metadata
def calculate_chunk_ids(chunks):
    """
    Assigns unique chunk IDs to a list of chunks based on their metadata.

    Args:
        chunks (list): A list of chunk objects with metadata containing "source" and "page" keys.

    Returns:
        list: The input list of chunks with updated metadata containing the unique IDs.
    """
    logger = AppConfig.get_default_logger()
    logger.info(f"Calculating unique IDs for {len(chunks)} chunks")

    try:
        # Create IDs like "data/some-pdf.pdf:6:2"
        # Page Source : Page Number : Chunk Index
        last_page_id = None
        current_chunk_index = 0
        generated_ids = []

        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown_source")
            page = chunk.metadata.get("page", "unknown_page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the unique chunk ID
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data
            chunk.metadata["id"] = chunk_id
            generated_ids.append(chunk_id)

        logger.info(f"Successfully generated {len(generated_ids)} unique chunk IDs")

        # Log some sample IDs for verification
        sample_ids = generated_ids[:5]
        logger.info(f"Sample generated IDs: {sample_ids}")

        return chunks

    except Exception as e:
        logger.error(f"Error calculating chunk IDs: {e}", exc_info=True)
        raise

# Clear the specified database by removing its directory
# embedding_model can be "openai" or "ollama"
def clear_database(embedding_model):
    """
    Clears the vector database for the specified embedding model by removing its directory.

    Args:
        embedding_model (str): The name of the embedding model ("openai" or "ollama").
    """
    logger = AppConfig.get_default_logger()
    logger.info(f"Clearing database for embedding model: {embedding_model}")

    try:
        if embedding_model == "openai":
            db_path = AppConfig.VECTOR_DB_OPENAI_PATH
        elif embedding_model == "ollama":
            db_path = AppConfig.VECTOR_DB_OLLAMA_PATH
        else:
            error_msg = f"Unsupported embedding model specified: {embedding_model}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Database path to clear: {db_path}")

        if os.path.exists(db_path):
            logger.info(f"Database directory exists - removing: {db_path}")
            shutil.rmtree(db_path)
            logger.info(f"Successfully cleared database directory: {db_path}")
            print(f"🗑️ Cleared {embedding_model} database")
        else:
            logger.info(f"Database directory does not exist: {db_path}")
            print(f"ℹ️ {embedding_model} database directory doesn't exist")

    except Exception as e:
        logger.error(f"Error clearing database: {e}", exc_info=True)
        print(f"❌ Error clearing {embedding_model} database: {e}")
        raise

# Rebuild the specified database
# embedding_model can be "openai" or "ollama"
def rebuild_database(embedding_model):
    """
    Rebuilds the vector database using the specified embedding model.

    Args:
        embedding_model (str): The name of the embedding model ("openai" or "ollama").
    """
    logger = AppConfig.get_default_logger()
    logger.info(f"Rebuilding database for embedding model: {embedding_model}")

    try:
        if embedding_model == "openai":
            logger.info("Initializing OpenAI embeddings")
            embeddings = EmbeddingFactory(
                model_name="openai", api_key=AppConfig.OPENAI_API_KEY
            )
            db_path = AppConfig.VECTOR_DB_OPENAI_PATH
        elif embedding_model == "ollama":
            logger.info("Initializing Ollama embeddings")
            embeddings = EmbeddingFactory(
                model_name="ollama", api_key=AppConfig.OPENAI_API_KEY
            )
            db_path = AppConfig.VECTOR_DB_OLLAMA_PATH
        else:
            error_msg = f"Unsupported embedding model specified: {embedding_model}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Database path for rebuild: {db_path}")
        embedding_function = embeddings.create_embedding_function()
        logger.info("Embedding function created successfully")

        # Load the existing database (will create new one if doesn't exist)
        logger.info("Creating new Chroma database instance")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        logger.info("Database instance created successfully")

        # Create (or update) the data store
        logger.info("Loading documents for database rebuild")
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks, db)

        logger.info(f"Successfully rebuilt {embedding_model} database")
        print(f"✅ {embedding_model} database rebuilt successfully")

    except Exception as e:
        logger.error(f"Error rebuilding database: {e}", exc_info=True)
        print(f"❌ Error rebuilding {embedding_model} database: {e}")
        raise

# Entry point
if __name__ == "__main__":
    main()
