from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sqlalchemy import create_engine, inspect, MetaData, Table, Column
from sqlalchemy.orm import sessionmaker
import json

# Data class for SQL schema information
@dataclass
class ColumnInfo:
    """Represents a SQL column definition."""
    column_name: str
    data_type: str
    is_nullable: bool
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    description: str = ""


@dataclass
class TableInfo:
    """Represents a SQL table definition."""
    table_name: str
    columns: List[ColumnInfo]
    description: str = ""


@dataclass
class SchemaInfo:
    """Represents a complete SQL schema."""
    database_name: str
    tables: List[TableInfo]


# MSSQLScanner class for inspecting MSSQL database schemas
class MSSQLScanner:
    """
    A database inspector that connects to MSSQL databases and extracts schema
    metadata for semantic mapping purposes.

    This class uses SQLAlchemy to:
    1. Connect to MSSQL servers
    2. Extract table and column metadata
    3. Generate JSON exports for the MetadataLoader
    4. Provide column descriptions for RAG context

    Attributes:
        connection_string (str): MSSQL connection string
        engine: SQLAlchemy engine instance
        inspector: SQLAlchemy inspector instance

    Methods:
        scan_database() -> SchemaInfo:
            Scan the entire database schema.

        scan_table(table_name: str) -> TableInfo:
            Scan a specific table.

        get_schema_json() -> Dict:
            Export schema as JSON for MetadataLoader.

        get_column_metadata(table_name: str, column_name: str) -> Dict:
            Get detailed metadata for a specific column.
    """

    def __init__(self, connection_string: str):
        """
        Initialize the MSSQLScanner with a connection string.

        Args:
            connection_string: MSSQL connection string in the format:
            'mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server'
            or
            'mssql+pyodbc://server/database?trusted_connection=yes'

        Raises:
            Exception: If connection fails
        """
        self.connection_string = connection_string
        try:
            self.engine = create_engine(connection_string)
            self.inspector = inspect(self.engine)
            self.connection = self.engine.connect()
            print(f"Connected to database: {self.get_database_name()}")
        except Exception as e:
            raise Exception(f"Failed to connect to MSSQL database: {str(e)}")

    def get_database_name(self) -> str:
        """Get the name of the connected database."""
        try:
            result = self.connection.execute("SELECT DB_NAME() AS database_name")
            return result.fetchone()[0]
        except:
            return "unknown"

    def scan_database(self) -> SchemaInfo:
        """
        Scan the entire database schema and extract all tables and columns.

        Returns:
            SchemaInfo object containing complete schema metadata
        """
        database_name = self.get_database_name()
        tables = self.inspector.get_table_names()
        table_infos = []

        for table_name in tables:
            try:
                table_info = self.scan_table(table_name)
                table_infos.append(table_info)
            except Exception as e:
                print(f"Warning: Failed to scan table {table_name}: {str(e)}")
                continue

        return SchemaInfo(database_name=database_name, tables=table_infos)

    def scan_table(self, table_name: str) -> TableInfo:
        """
        Scan a specific table and extract column metadata.

        Args:
            table_name: Name of the table to scan

        Returns:
            TableInfo object with column definitions
        """
        columns_data = self.inspector.get_columns(table_name)
        columns = []

        for col in columns_data:
            column_info = ColumnInfo(
                column_name=col["name"],
                data_type=str(col["type"]),
                is_nullable=col.get("nullable", True),
                length=col.get("length"),
                precision=col.get("precision"),
                scale=col.get("scale"),
                description=self._get_column_description(table_name, col["name"])
            )
            columns.append(column_info)

        return TableInfo(table_name=table_name, columns=columns)

    def _get_column_description(self, table_name: str, column_name: str) -> str:
        """
        Extract the extended property (description) of a column from SQL Server.

        Args:
            table_name: Table name
            column_name: Column name

        Returns:
            Column description or empty string if not found
        """
        try:
            query = f"""
            SELECT p.value
            FROM sys.columns c
            INNER JOIN sys.tables t ON c.object_id = t.object_id
            LEFT JOIN sys.extended_properties p
                ON c.object_id = p.major_id
                AND c.column_id = p.minor_id
                AND p.name = 'MS_Description'
            WHERE t.name = '{table_name}' AND c.name = '{column_name}'
            """
            result = self.connection.execute(query)
            row = result.fetchone()
            return row[0] if row and row[0] else ""
        except:
            return ""

    def get_column_metadata(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """
        Get detailed metadata for a specific column.

        Args:
            table_name: Table name
            column_name: Column name

        Returns:
            Dictionary with column metadata
        """
        columns_data = self.inspector.get_columns(table_name)

        for col in columns_data:
            if col["name"] == column_name:
                return {
                    "table_name": table_name,
                    "column_name": col["name"],
                    "data_type": str(col["type"]),
                    "is_nullable": col.get("nullable", True),
                    "length": col.get("length"),
                    "precision": col.get("precision"),
                    "scale": col.get("scale"),
                    "description": self._get_column_description(table_name, column_name),
                    "primary_key": col.get("primary_key", False),
                    "autoincrement": col.get("autoincrement", False)
                }

        raise ValueError(f"Column {column_name} not found in table {table_name}")

    def get_schema_json(self) -> Dict[str, Any]:
        """
        Export the complete database schema as a JSON-compatible dictionary.

        This format is compatible with MetadataLoader.load_sql_schema_metadata().

        Returns:
            Dictionary with schema information suitable for JSON serialization
        """
        schema_info = self.scan_database()

        return {
            "database": schema_info.database_name,
            "tables": [
                {
                    "table_name": table.table_name,
                    "description": table.description,
                    "columns": [
                        {
                            "column_name": col.column_name,
                            "data_type": col.data_type,
                            "is_nullable": col.is_nullable,
                            "length": col.length,
                            "precision": col.precision,
                            "scale": col.scale,
                            "description": col.description
                        }
                        for col in table.columns
                    ]
                }
                for table in schema_info.tables
            ]
        }

    def get_table_columns_as_documents(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get all columns from a table formatted for RAG indexing.

        Args:
            table_name: Name of the table

        Returns:
            List of column metadata dictionaries
        """
        table_info = self.scan_table(table_name)
        return [
            {
                "column_name": col.column_name,
                "data_type": col.data_type,
                "is_nullable": col.is_nullable,
                "description": col.description,
                "length": str(col.length) if col.length else None
            }
            for col in table_info.columns
        ]

    def scan_multiple_databases(self, database_names: List[str]) -> Dict[str, Dict]:
        """
        Scan multiple MSSQL databases and return their schemas.

        Note: This requires changing the connection to each database sequentially.

        Args:
            database_names: List of database names to scan

        Returns:
            Dictionary mapping database names to their schema JSON
        """
        results = {}

        for db_name in database_names:
            try:
                # Change database context
                self.connection.execute(f"USE {db_name}")
                schema_json = self.get_schema_json()
                results[db_name] = schema_json
                print(f"Successfully scanned database: {db_name}")
            except Exception as e:
                print(f"Failed to scan database {db_name}: {str(e)}")
                results[db_name] = {"error": str(e)}

        return results

    def export_schema_to_json_file(self, file_path: str) -> None:
        """
        Export the database schema to a JSON file.

        Args:
            file_path: Path where the JSON file should be saved
        """
        schema_json = self.get_schema_json()

        with open(file_path, 'w') as f:
            json.dump(schema_json, f, indent=2, default=str)

        print(f"Schema exported to {file_path}")

    def get_tables_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all tables in the database.

        Returns:
            List of dictionaries with table summaries (name, column count, description)
        """
        schema_info = self.scan_database()
        return [
            {
                "table_name": table.table_name,
                "column_count": len(table.columns),
                "description": table.description,
                "columns": [col.column_name for col in table.columns]
            }
            for table in schema_info.tables
        ]

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
