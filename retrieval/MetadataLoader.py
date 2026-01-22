import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from embeddings.EmbeddingFactory import EmbeddingFactory

# MetadataLoader for treating SQL schema metadata as RAG documents
class MetadataLoader:
    """
    A specialized loader that treats SQL column descriptions and ACORD standard definitions
    as RAG documents for semantic mapping.

    This class converts structured metadata (from INFORMATION_SCHEMA, data dictionaries, and
    ACORD standards) into LangChain Document objects that can be indexed in ChromaDB.

    Attributes:
        embedding_model_name (str): Name of the embedding model to use
        api_key (str): API key for the embedding service
        embedding_function (Callable): The embedding function instance

    Methods:
        load_sql_schema_metadata(schema_json: Dict) -> List[Document]:
            Converts SQL INFORMATION_SCHEMA metadata into Documents.

        load_acord_standard(acord_definitions: Dict) -> List[Document]:
            Converts ACORD logical model definitions into Documents.

        load_customer_dictionary(dictionary_path: str) -> List[Document]:
            Loads customer business dictionary from Excel/PDF and converts to Documents.

        create_document_from_column(table_name: str, column_info: Dict) -> Document:
            Creates a Document from a single SQL column definition.
    """

    def __init__(self, embedding_model_name: str, api_key: str):
        """
        Initialize the MetadataLoader with embedding configuration.

        Args:
            embedding_model_name: Name of the embedding model (e.g., 'openai', 'ollama')
            api_key: API key for the embedding service
        """
        self.embedding_model_name = embedding_model_name
        self.api_key = api_key
        embeddings = EmbeddingFactory(model_name=embedding_model_name, api_key=api_key)
        self.embedding_function = embeddings.create_embedding_function()

    def load_sql_schema_metadata(self, schema_json: Dict[str, Any]) -> List[Document]:
        """
        Convert SQL INFORMATION_SCHEMA exports into RAG-indexable Documents.

        The input JSON should have the structure:
        {
            "database": "database_name",
            "tables": [
                {
                    "table_name": "TBL_POLICY",
                    "columns": [
                        {
                            "column_name": "EFF_DT",
                            "data_type": "DATE",
                            "is_nullable": false,
                            "description": "Policy effective date"
                        }
                    ]
                }
            ]
        }

        Args:
            schema_json: Dictionary containing database schema information

        Returns:
            List of Document objects, one per column
        """
        documents = []
        database_name = schema_json.get("database", "unknown")

        for table in schema_json.get("tables", []):
            table_name = table.get("table_name")
            for column in table.get("columns", []):
                doc = self.create_document_from_column(table_name, column, database_name)
                documents.append(doc)

        return documents

    def create_document_from_column(self, table_name: str, column_info: Dict,
                                    database_name: str = "") -> Document:
        """
        Create a LangChain Document from a SQL column definition.

        Args:
            table_name: Name of the table containing the column
            column_info: Dictionary with column metadata (name, type, description, etc.)
            database_name: Optional database name for context

        Returns:
            Document object ready for ChromaDB indexing
        """
        column_name = column_info.get("column_name", "unknown")
        data_type = column_info.get("data_type", "unknown")
        description = column_info.get("description", "No description provided")
        is_nullable = column_info.get("is_nullable", True)
        length = column_info.get("length", "")
        precision = column_info.get("precision", "")
        scale = column_info.get("scale", "")

        # Build comprehensive page content for the document
        page_content = f"""
SQL Column Metadata:
Database: {database_name}
Table: {table_name}
Column: {column_name}
Data Type: {data_type}{f' ({length})' if length else ''}{f'({precision},{scale})' if precision else ''}
Nullable: {is_nullable}
Description: {description}
"""

        # Metadata for filtering and source tracking
        metadata = {
            "source_type": "sql_schema",
            "database": database_name,
            "table": table_name,
            "column": column_name,
            "data_type": data_type,
            "is_nullable": str(is_nullable),
            "document_type": "sql_metadata"
        }

        return Document(page_content=page_content.strip(), metadata=metadata)

    def load_acord_standard(self, acord_definitions: Dict[str, Any]) -> List[Document]:
        """
        Convert ACORD standard definitions into Documents for semantic mapping reference.

        The input JSON should have the structure:
        {
            "entities": [
                {
                    "entity_name": "Policy",
                    "description": "Policy entity definition",
                    "fields": [
                        {
                            "field_name": "EffectiveDate",
                            "data_type": "Date",
                            "description": "The date when the policy becomes effective",
                            "required": true
                        }
                    ]
                }
            ]
        }

        Args:
            acord_definitions: Dictionary containing ACORD logical model definitions

        Returns:
            List of Document objects for ACORD entities and fields
        """
        documents = []

        for entity in acord_definitions.get("entities", []):
            entity_name = entity.get("entity_name")
            entity_desc = entity.get("description", "")
            fields = entity.get("fields", [])

            # Create a document for the entity itself
            entity_page_content = f"""
ACORD Standard Entity: {entity_name}
Description: {entity_desc}
Fields: {', '.join([f['field_name'] for f in fields])}
"""
            entity_metadata = {
                "source_type": "acord_standard",
                "entity": entity_name,
                "document_type": "acord_entity"
            }
            documents.append(Document(page_content=entity_page_content.strip(),
                                     metadata=entity_metadata))

            # Create documents for each field
            for field in fields:
                field_name = field.get("field_name")
                field_type = field.get("data_type", "")
                field_desc = field.get("description", "")
                required = field.get("required", False)

                field_page_content = f"""
ACORD Standard Field: {entity_name}.{field_name}
Data Type: {field_type}
Required: {required}
Description: {field_desc}
Business Context: This field is part of the {entity_name} entity in the ACORD standard.
"""
                field_metadata = {
                    "source_type": "acord_standard",
                    "entity": entity_name,
                    "field": field_name,
                    "data_type": field_type,
                    "required": str(required),
                    "document_type": "acord_field"
                }
                documents.append(Document(page_content=field_page_content.strip(),
                                         metadata=field_metadata))

        return documents

    def load_customer_dictionary(self, dictionary_json: Dict[str, Any]) -> List[Document]:
        """
        Load customer business dictionary definitions (from Excel/PDF exports).

        The input JSON should have the structure:
        {
            "terms": [
                {
                    "business_term": "Policy Effective Date",
                    "technical_term": "EFF_DT",
                    "definition": "The date when the policy becomes effective",
                    "system": "Policy_DB",
                    "related_systems": ["Premium_DB", "Claims_DB"]
                }
            ]
        }

        Args:
            dictionary_json: Dictionary containing business term definitions

        Returns:
            List of Document objects for business terms
        """
        documents = []

        for term in dictionary_json.get("terms", []):
            business_term = term.get("business_term", "")
            technical_term = term.get("technical_term", "")
            definition = term.get("definition", "")
            system = term.get("system", "")
            related_systems = term.get("related_systems", [])

            page_content = f"""
Business Term: {business_term}
Technical Term: {technical_term}
Definition: {definition}
Primary System: {system}
Related Systems: {', '.join(related_systems) if related_systems else 'None'}
Context: This is a business dictionary entry mapping technical terms to business concepts.
"""

            metadata = {
                "source_type": "customer_dictionary",
                "business_term": business_term,
                "technical_term": technical_term,
                "system": system,
                "document_type": "business_term"
            }

            documents.append(Document(page_content=page_content.strip(), metadata=metadata))

        return documents

    def index_metadata_documents(self, documents: List[Document], vector_db_path: str) -> None:
        """
        Index a collection of metadata documents into ChromaDB.

        Args:
            documents: List of Document objects to index
            vector_db_path: Path to the ChromaDB vector database

        Raises:
            Exception: If indexing fails
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        db = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_function
        )

        # Add documents to the database
        db.add_documents(documents)
        print(f"Successfully indexed {len(documents)} metadata documents")

    def load_all_metadata(self, schema_json: Dict, acord_definitions: Dict,
                         dictionary_json: Dict, vector_db_path: str) -> int:
        """
        Load all three types of metadata (SQL schema, ACORD standard, customer dictionary)
        and index them into the vector database.

        Args:
            schema_json: SQL schema metadata
            acord_definitions: ACORD standard definitions
            dictionary_json: Customer business dictionary
            vector_db_path: Path to ChromaDB

        Returns:
            Total number of documents indexed
        """
        all_documents = []

        # Load SQL schema metadata
        sql_docs = self.load_sql_schema_metadata(schema_json)
        all_documents.extend(sql_docs)
        print(f"Loaded {len(sql_docs)} SQL schema documents")

        # Load ACORD standard
        acord_docs = self.load_acord_standard(acord_definitions)
        all_documents.extend(acord_docs)
        print(f"Loaded {len(acord_docs)} ACORD standard documents")

        # Load customer dictionary
        dict_docs = self.load_customer_dictionary(dictionary_json)
        all_documents.extend(dict_docs)
        print(f"Loaded {len(dict_docs)} customer dictionary documents")

        # Index all documents
        self.index_metadata_documents(all_documents, vector_db_path)

        return len(all_documents)
