from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from mapping_prompts.HarmonizationPrompts import get_code_generation_prompt


@dataclass
class SparkTransformation:
    """Represents a generated PySpark transformation."""
    source_table: str
    source_column: str
    target_acord_entity: str
    target_acord_field: str
    source_data_type: str
    target_data_type: str
    pyspark_code: str
    include_qc_checks: bool = True
    comments: str = ""


class SparkCodeGen:
    """
    A code generator that produces PySpark ETL transformations based on approved
    data mappings from the MappingEngine.

    This class translates semantic mappings into production-ready PySpark code that
    can be executed in a Lakehouse environment (Databricks, etc.).

    Attributes:
        llm_model: The language model for code generation
        enable_comments: Whether to include inline comments
        data_quality_checks: Whether to include data quality validation

    Methods:
        generate_column_transformation(mapping: ColumnMapping) -> SparkTransformation:
            Generate PySpark code for a single column mapping.

        generate_table_etl(table_mappings: List[ColumnMapping]) -> str:
            Generate a complete ETL script for a table.

        generate_batch_etl(all_mappings: List[ColumnMapping]) -> str:
            Generate ETL scripts for multiple tables.

        add_data_quality_checks(spark_code: str, columns: List[str]) -> str:
            Add DQ validations to generated code.
    """

    def __init__(self, llm_model, enable_comments: bool = True, enable_dq_checks: bool = True):
        """
        Initialize the SparkCodeGen.

        Args:
            llm_model: Initialized LLM model for code generation
            enable_comments: Whether to include explanatory comments
            enable_dq_checks: Whether to include data quality checks
        """
        self.llm_model = llm_model
        self.enable_comments = enable_comments
        self.enable_dq_checks = enable_dq_checks

    def generate_column_transformation(self, mapping: Any) -> SparkTransformation:
        """
        Generate PySpark code for transforming a single column mapping.

        Args:
            mapping: ColumnMapping object with source and target information

        Returns:
            SparkTransformation object with generated code
        """
        source_table = mapping.source_table
        source_column = mapping.source_column
        target_entity = mapping.target_acord_entity
        target_field = mapping.target_acord_field
        source_data_type = mapping.source_data_type
        target_data_type = "date" if "date" in source_data_type.lower() else "string"

        # Determine transformation logic based on data types
        business_logic = self._determine_business_logic(
            source_data_type, target_data_type, source_column
        )

        # Generate code using LLM
        code = self._generate_transformation_code(
            source_table, source_column, target_entity, target_field,
            source_data_type, target_data_type, business_logic
        )

        # Add data quality checks if enabled
        if self.enable_dq_checks:
            code = self._add_dq_checks(code, source_column)

        return SparkTransformation(
            source_table=source_table,
            source_column=source_column,
            target_acord_entity=target_entity,
            target_acord_field=target_field,
            source_data_type=source_data_type,
            target_data_type=target_data_type,
            pyspark_code=code,
            include_qc_checks=self.enable_dq_checks,
            comments=mapping.reasoning[:200] if mapping.reasoning else ""
        )

    def _determine_business_logic(self, source_type: str, target_type: str,
                                   column_name: str) -> str:
        """
        Determine the transformation logic based on data types and column semantics.

        Args:
            source_type: Source SQL data type
            target_type: Target data type
            column_name: Column name (for context)

        Returns:
            Description of the transformation logic
        """
        source_lower = source_type.lower()
        target_lower = target_type.lower()

        if "date" in source_lower and "date" in target_lower:
            return "Direct date format conversion with null handling"
        elif "decimal" in source_lower or "numeric" in source_lower:
            if "string" in target_lower:
                return "Convert numeric to string with precision preservation"
            else:
                return "Direct numeric type conversion"
        elif source_lower == target_lower:
            return "Direct type conversion (no transformation needed)"
        else:
            return f"Type conversion from {source_type} to {target_type}"

    def _generate_transformation_code(self, source_table: str, source_column: str,
                                       target_entity: str, target_field: str,
                                       source_type: str, target_type: str,
                                       business_logic: str) -> str:
        """
        Generate PySpark transformation code using LLM.

        Args:
            source_table: Source table name
            source_column: Source column name
            target_entity: Target ACORD entity
            target_field: Target ACORD field
            source_type: Source data type
            target_type: Target data type
            business_logic: Business logic description

        Returns:
            Generated PySpark code as string
        """
        try:
            prompt = get_code_generation_prompt(
                source_table=source_table,
                source_column=source_column,
                target_entity=target_entity,
                source_data_type=source_type,
                target_data_type=target_type,
                business_logic=business_logic
            )

            code = self.llm_model.invoke(prompt)
            return code
        except Exception as e:
            # Fallback to basic transformation
            return self._generate_basic_transformation(
                source_column, target_entity, target_field, source_type, target_type
            )

    def _generate_basic_transformation(self, source_column: str, target_entity: str,
                                        target_field: str, source_type: str,
                                        target_type: str) -> str:
        """
        Generate basic transformation code when LLM generation fails.

        Args:
            source_column: Source column name
            target_entity: Target entity
            target_field: Target field
            source_type: Source data type
            target_type: Target data type

        Returns:
            Basic PySpark transformation code
        """
        target_col_name = f"{target_entity}_{target_field}".lower()

        # Basic transformation logic
        if "date" in source_type.lower():
            code = f"""
# Transform {source_column} to {target_entity}.{target_field}
df_gold = df_bronze.withColumn(
    "{target_col_name}",
    col("{source_column}").cast("date")
)
"""
        elif "decimal" in source_type.lower() or "numeric" in source_type.lower():
            code = f"""
# Transform {source_column} to {target_entity}.{target_field}
df_gold = df_bronze.withColumn(
    "{target_col_name}",
    col("{source_column}").cast("double")
)
"""
        else:
            code = f"""
# Transform {source_column} to {target_entity}.{target_field}
df_gold = df_bronze.withColumn(
    "{target_col_name}",
    col("{source_column}").cast("string")
)
"""

        return code.strip()

    def _add_dq_checks(self, code: str, column_name: str) -> str:
        """
        Add data quality checks to the transformation code.

        Args:
            code: Base PySpark transformation code
            column_name: Column being transformed

        Returns:
            Code with data quality checks added
        """
        dq_code = f"""
# Data Quality Checks for {column_name}
df_gold = df_gold.filter(
    (col("{column_name}").isNotNull()) |
    (col("{column_name}") != "")
)

# Add DQ flags
df_gold = df_gold.withColumn(
    "dq_check_{column_name}",
    when(col("{column_name}").isNull(), "FAILED_NULL_CHECK")
        .when(col("{column_name}") == "", "FAILED_EMPTY_CHECK")
        .otherwise("PASSED")
)
"""
        return code + "\n" + dq_code.strip()

    def generate_table_etl(self, table_mappings: List[Any]) -> str:
        """
        Generate a complete ETL script for a table with all column transformations.

        Args:
            table_mappings: List of ColumnMapping objects for the table

        Returns:
            Complete PySpark ETL script as string
        """
        if not table_mappings:
            return "# No mappings provided"

        # Get table name from first mapping
        table_name = table_mappings[0].source_table
        system_name = table_mappings[0].source_system

        etl_script = f"""
# ============================================================================
# ETL Script: {system_name}.{table_name} -> ACORD Standard
# Generated by SparkCodeGen
# ============================================================================
from pyspark.sql.functions import col, when, to_date, regexp_replace, trim
from pyspark.sql.types import StructType, StructField, StringType, DateType

# Load source data
df_bronze = spark.read.table("{system_name}.{table_name}")

print(f"Loaded {{len(df_bronze.columns)}} columns from source table")

# Apply transformations
"""

        # Add each column transformation
        for mapping in table_mappings:
            if mapping.status == "APPROVED":
                transformation = self.generate_column_transformation(mapping)
                etl_script += f"\n# Mapping: {mapping.source_column} -> {mapping.target_acord_entity}.{mapping.target_acord_field}\n"
                etl_script += transformation.pyspark_code + "\n"

        # Final validation section
        etl_script += """
# ============================================================================
# Final Data Quality Validation
# ============================================================================
print(f"Gold table schema: {df_gold.schema}")
print(f"Gold table row count: {df_gold.count()}")

# Write to gold layer
df_gold.write.mode("overwrite").format("delta").save(
    "/mnt/gold/{system_name}_{table_name}"
)

print("ETL completed successfully")
"""

        return etl_script

    def generate_batch_etl(self, all_mappings: List[Any]) -> Dict[str, str]:
        """
        Generate ETL scripts for multiple tables.

        Args:
            all_mappings: List of ColumnMapping objects for multiple tables

        Returns:
            Dictionary mapping table names to their ETL scripts
        """
        # Group mappings by table
        tables = {}
        for mapping in all_mappings:
            table_key = mapping.source_table
            if table_key not in tables:
                tables[table_key] = []
            tables[table_key].append(mapping)

        # Generate ETL for each table
        etl_scripts = {}
        for table_name, mappings in tables.items():
            approved_mappings = [m for m in mappings if m.status == "APPROVED"]
            if approved_mappings:
                etl_scripts[table_name] = self.generate_table_etl(approved_mappings)

        return etl_scripts

    def generate_orchestration_script(self, etl_scripts: Dict[str, str],
                                      output_format: str = "job") -> str:
        """
        Generate an orchestration script to run all ETL transformations sequentially.

        Args:
            etl_scripts: Dictionary of ETL scripts by table
            output_format: Output format ('job' for Databricks job, 'notebook' for notebook)

        Returns:
            Orchestration script
        """
        if output_format == "job":
            return self._generate_job_orchestration(etl_scripts)
        else:
            return self._generate_notebook_orchestration(etl_scripts)

    def _generate_job_orchestration(self, etl_scripts: Dict[str, str]) -> str:
        """Generate a job orchestration script."""
        job_script = """
# ============================================================================
# Databricks Job Orchestration Script
# Executes all ETL transformations sequentially
# ============================================================================

import subprocess
import json
from datetime import datetime

WORKSPACE_PATH = "/Workspace/ETL/Harmonizer"
LOG_FILE = f"{WORKSPACE_PATH}/etl_run_{datetime.now().isoformat()}.log"

def run_etl_job(notebook_path: str) -> bool:
    '''Run a single ETL notebook and return success status'''
    try:
        result = dbutils.notebook.run(notebook_path, timeout_seconds=3600)
        return True
    except Exception as e:
        print(f"Error running {notebook_path}: {str(e)}")
        return False

# Execute ETL jobs
etl_jobs = {
"""
        for table_name in etl_scripts.keys():
            job_script += f'    "{table_name}": "{{{table_name}}}_etl",\n'

        job_script += """
}

results = {}
for table_name, job_name in etl_jobs.items():
    print(f"Starting ETL for {table_name}...")
    success = run_etl_job(f"{WORKSPACE_PATH}/jobs/{job_name}")
    results[table_name] = "SUCCESS" if success else "FAILED"
    print(f"ETL for {table_name}: {results[table_name]}")

# Summary
print("\\n" + "="*60)
print("ETL Orchestration Summary")
print("="*60)
for table, status in results.items():
    print(f"{table}: {status}")
"""
        return job_script

    def _generate_notebook_orchestration(self, etl_scripts: Dict[str, str]) -> str:
        """Generate a notebook orchestration script."""
        notebook_script = """
# Databricks notebook source
# ============================================================================
# Unified ETL Orchestration Notebook
# Executes all data harmonization transformations
# ============================================================================

from datetime import datetime

print(f"Starting ETL Orchestration at {datetime.now()}")
print("="*60)

etl_results = {}

"""
        for table_name, etl_code in etl_scripts.items():
            notebook_script += f"""
# Execute ETL for {table_name}
print(f"\\nProcessing {table_name}...")
try:
    {etl_code}
    etl_results["{table_name}"] = "SUCCESS"
except Exception as e:
    print(f"Error processing {table_name}: {{e}}")
    etl_results["{table_name}"] = "FAILED"

"""

        notebook_script += """
# Summary report
print("\\n" + "="*60)
print("ETL Orchestration Summary")
print("="*60)
for table, status in sorted(etl_results.items()):
    print(f"{table}: {status}")

total = len(etl_results)
successful = sum(1 for s in etl_results.values() if s == "SUCCESS")
print(f"\\nTotal Tables: {total}, Successful: {successful}, Failed: {total - successful}")
"""
        return notebook_script

    def export_code_to_file(self, code: str, file_path: str) -> None:
        """
        Export generated code to a Python file.

        Args:
            code: The generated code
            file_path: Path where the file should be saved
        """
        with open(file_path, 'w') as f:
            f.write(code)
        print(f"Code exported to {file_path}")

    def validate_generated_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Perform basic validation on generated PySpark code.

        Args:
            code: The generated code to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check for common issues
        if "df_bronze" not in code:
            warnings.append("Code does not reference source dataframe (df_bronze)")

        if "df_gold" not in code:
            warnings.append("Code does not reference target dataframe (df_gold)")

        if "spark.read" not in code and "spark.write" not in code:
            warnings.append("Code may not have I/O operations")

        if "from pyspark.sql.functions import" not in code:
            warnings.append("Code may be missing required imports")

        is_valid = len(warnings) == 0

        return is_valid, warnings
