"""
Integration Example: Complete Data Model Harmonization Workflow

This script demonstrates the end-to-end workflow for harmonizing a SQL database
schema to ACORD standards using the Data Model Harmonizer.

Steps:
1. Connect to source MSSQL database
2. Extract and export schema
3. Index metadata (ACORD, schema, dictionary)
4. Generate mappings for all tables
5. Review and approve mappings
6. Generate PySpark ETL code
7. Export orchestration script
"""

import json
from typing import List, Dict, Any
from datetime import datetime

# Import harmonizer components
from AppConfig import AppConfig
from source_inspectors.MSSQLScanner import MSSQLScanner
from retrieval.MetadataLoader import MetadataLoader
from mapping.MappingEngine import MappingEngine
from output_gen.SparkCodeGen import SparkCodeGen
from DocUploader import index_harmonizer_metadata


class HarmonizationWorkflow:
    """Complete harmonization workflow orchestrator."""

    def __init__(self, embedding_model: str = "openai"):
        """Initialize the workflow."""
        self.embedding_model = embedding_model
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "phases": {}
        }

    def phase_1_extract_database_schema(self, connection_string: str,
                                        output_dir: str = "./exports") -> str:
        """
        Phase 1: Extract schema from source MSSQL database.

        Args:
            connection_string: MSSQL connection string
            output_dir: Directory to save schema export

        Returns:
            Path to exported schema JSON file
        """
        print("\n" + "="*60)
        print("PHASE 1: Extract Database Schema")
        print("="*60)

        try:
            # Connect to database
            print(f"Connecting to: {connection_string[:50]}...")
            scanner = MSSQLScanner(connection_string)

            # Get database name
            db_name = scanner.get_database_name()
            print(f"Connected to database: {db_name}")

            # Get tables summary
            tables = scanner.get_tables_summary()
            print(f"Found {len(tables)} tables")
            for table in tables[:5]:  # Show first 5
                print(f"  - {table['table_name']}: {table['column_count']} columns")

            # Export schema
            schema_file = f"{output_dir}/{db_name}_schema.json"
            scanner.export_schema_to_json_file(schema_file)
            print(f"✓ Schema exported to: {schema_file}")

            # Store results
            self.results["phases"]["extract"] = {
                "database": db_name,
                "tables_count": len(tables),
                "schema_file": schema_file
            }

            scanner.close()
            return schema_file

        except Exception as e:
            print(f"✗ Error in Phase 1: {e}")
            raise

    def phase_2_index_metadata(self, acord_file: str, schema_file: str,
                               dictionary_file: str) -> int:
        """
        Phase 2: Index all metadata into ChromaDB.

        Args:
            acord_file: Path to ACORD standard JSON
            schema_file: Path to SQL schema JSON
            dictionary_file: Path to customer dictionary JSON

        Returns:
            Total documents indexed
        """
        print("\n" + "="*60)
        print("PHASE 2: Index Metadata into ChromaDB")
        print("="*60)

        try:
            # Initialize components
            AppConfig.initialize_components()

            # Index metadata
            print("Indexing ACORD standard...")
            print("Indexing SQL schema...")
            print("Indexing customer dictionary...")

            total_docs = index_harmonizer_metadata(
                embedding_model=self.embedding_model,
                acord_json_path=acord_file,
                schema_json_path=schema_file,
                dictionary_json_path=dictionary_file
            )

            print(f"✓ Indexed {total_docs} documents into ChromaDB")

            self.results["phases"]["index"] = {
                "total_documents": total_docs,
                "embedding_model": self.embedding_model
            }

            return total_docs

        except Exception as e:
            print(f"✗ Error in Phase 2: {e}")
            raise

    def phase_3_generate_mappings(self, schema_file: str,
                                   top_n_tables: int = None) -> List[Dict]:
        """
        Phase 3: Generate mappings for database tables.

        Args:
            schema_file: Path to SQL schema JSON
            top_n_tables: Limit to first N tables (for testing)

        Returns:
            List of mapping dictionaries
        """
        print("\n" + "="*60)
        print("PHASE 3: Generate Semantic Mappings")
        print("="*60)

        try:
            # Load schema
            with open(schema_file, 'r') as f:
                schema_data = json.load(f)

            # Initialize mapping engine
            engine = MappingEngine(
                retriever=AppConfig.rag_retriever,
                llm_model=AppConfig.llm_model,
                enable_chain_of_thought=True
            )

            # Generate mappings
            database = schema_data.get("database", "unknown")
            all_mappings = []

            tables = schema_data.get("tables", [])
            if top_n_tables:
                tables = tables[:top_n_tables]

            for i, table in enumerate(tables, 1):
                table_name = table.get("table_name")
                print(f"\n[{i}/{len(tables)}] Processing table: {table_name}")

                table_metadata = {
                    "source_system": database,
                    "table_name": table_name,
                    "columns": table.get("columns", [])
                }

                try:
                    # Generate batch mappings
                    mappings = engine.batch_generate_mappings(table_metadata)
                    all_mappings.extend(mappings)

                    # Show summary
                    summary = engine.get_mapping_summary(table_name)
                    print(f"  Generated: {summary['total_mappings']} mappings")
                    print(f"  Confidence: avg={summary['average_confidence']:.2f}")

                except Exception as e:
                    print(f"  ✗ Error processing table: {e}")
                    continue

            # Save mappings
            mappings_file = f"./exports/{database}_mappings.json"
            with open(mappings_file, 'w') as f:
                json.dump(
                    [m.to_dict() for m in all_mappings],
                    f,
                    indent=2,
                    default=str
                )

            print(f"\n✓ Generated {len(all_mappings)} total mappings")
            print(f"✓ Saved to: {mappings_file}")

            self.results["phases"]["generate"] = {
                "total_mappings": len(all_mappings),
                "tables_processed": len(tables),
                "mappings_file": mappings_file
            }

            return all_mappings

        except Exception as e:
            print(f"✗ Error in Phase 3: {e}")
            raise

    def phase_4_review_and_approve(self, mappings: List[Dict],
                                    confidence_threshold: float = 0.8) -> int:
        """
        Phase 4: Review and approve mappings (simulated).

        In production, this would be a web UI where analysts review.
        Here we auto-approve high-confidence mappings.

        Args:
            mappings: List of mapping dictionaries
            confidence_threshold: Auto-approve if confidence > this

        Returns:
            Number of approved mappings
        """
        print("\n" + "="*60)
        print("PHASE 4: Review and Approve Mappings")
        print("="*60)

        try:
            engine = MappingEngine(
                retriever=AppConfig.rag_retriever,
                llm_model=AppConfig.llm_model
            )

            approved_count = 0
            requires_review_count = 0

            for mapping in mappings:
                confidence = mapping.get("confidence_score", 0)

                if confidence >= confidence_threshold:
                    # Auto-approve high-confidence mappings
                    engine.approve_mapping(
                        mapping["source_table"],
                        mapping["source_column"],
                        approved_by="harmonizer_auto"
                    )
                    approved_count += 1
                elif confidence >= 0.5:
                    # Mark for review
                    print(f"  Review: {mapping['source_column']} → "
                          f"{mapping['target_acord_path']} ({confidence:.2f})")
                    requires_review_count += 1

            print(f"\n✓ Auto-approved: {approved_count} mappings (confidence ≥ {confidence_threshold})")
            print(f"⚠ Requires review: {requires_review_count} mappings")

            self.results["phases"]["review"] = {
                "approved": approved_count,
                "requires_review": requires_review_count,
                "threshold": confidence_threshold
            }

            return approved_count

        except Exception as e:
            print(f"✗ Error in Phase 4: {e}")
            raise

    def phase_5_generate_etl_code(self, mappings: List[Dict]) -> Dict[str, str]:
        """
        Phase 5: Generate PySpark ETL code for approved mappings.

        Args:
            mappings: List of approved mapping dictionaries

        Returns:
            Dictionary mapping table names to ETL scripts
        """
        print("\n" + "="*60)
        print("PHASE 5: Generate PySpark ETL Code")
        print("="*60)

        try:
            # Filter approved mappings
            approved = [m for m in mappings if m.get("status") == "APPROVED"]
            print(f"Generating code for {len(approved)} approved mappings")

            # Group by table
            tables_dict = {}
            for mapping in approved:
                table = mapping["source_table"]
                if table not in tables_dict:
                    tables_dict[table] = []
                tables_dict[table].append(mapping)

            # Create mock mapping objects and generate code
            class MockMapping:
                def __init__(self, data):
                    for k, v in data.items():
                        setattr(self, k, v)
                    self.reasoning = f"Transform {data['source_column']}"
                    self.status = "APPROVED"

            codegen = SparkCodeGen(AppConfig.llm_model)
            etl_scripts = {}

            for table_name, table_mappings in tables_dict.items():
                print(f"  Generating ETL for {table_name}...")
                mock_mappings = [MockMapping(m) for m in table_mappings]
                etl_script = codegen.generate_table_etl(mock_mappings)
                etl_scripts[table_name] = etl_script

            # Save ETL scripts
            for table_name, etl_code in etl_scripts.items():
                filename = f"./exports/{table_name}_etl.py"
                with open(filename, 'w') as f:
                    f.write(etl_code)
                print(f"  ✓ Saved: {filename}")

            print(f"\n✓ Generated {len(etl_scripts)} ETL scripts")

            self.results["phases"]["codegen"] = {
                "tables": len(etl_scripts),
                "scripts_generated": list(etl_scripts.keys())
            }

            return etl_scripts

        except Exception as e:
            print(f"✗ Error in Phase 5: {e}")
            raise

    def phase_6_generate_orchestration(self, etl_scripts: Dict[str, str]) -> str:
        """
        Phase 6: Generate orchestration script for Databricks/Spark.

        Args:
            etl_scripts: Dictionary of ETL scripts

        Returns:
            Path to orchestration script
        """
        print("\n" + "="*60)
        print("PHASE 6: Generate Orchestration Script")
        print("="*60)

        try:
            codegen = SparkCodeGen(AppConfig.llm_model)

            # Generate job orchestration
            orchestration = codegen.generate_orchestration_script(
                etl_scripts,
                output_format="job"
            )

            # Save orchestration script
            orchestration_file = "./exports/orchestration_job.py"
            with open(orchestration_file, 'w') as f:
                f.write(orchestration)

            print(f"✓ Generated orchestration script: {orchestration_file}")

            self.results["phases"]["orchestration"] = {
                "tables_orchestrated": len(etl_scripts),
                "orchestration_file": orchestration_file
            }

            return orchestration_file

        except Exception as e:
            print(f"✗ Error in Phase 6: {e}")
            raise

    def print_summary(self):
        """Print workflow summary."""
        print("\n" + "="*60)
        print("HARMONIZATION WORKFLOW SUMMARY")
        print("="*60)

        for phase, results in self.results.get("phases", {}).items():
            print(f"\n{phase.upper()}:")
            for key, value in results.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(str(v) for v in value)}")
                else:
                    print(f"  {key}: {value}")

        print("\n" + "="*60)


def main():
    """Run the complete harmonization workflow."""

    print("""
╔════════════════════════════════════════════════════════════╗
║     Data Model Harmonizer - Complete Workflow Example      ║
║                                                            ║
║  Transforms: MSSQL Schema → ACORD Standard Mappings       ║
║              + Generates PySpark ETL Code                 ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Initialize workflow
    workflow = HarmonizationWorkflow(embedding_model="openai")

    try:
        # PHASE 1: Extract database schema
        schema_file = workflow.phase_1_extract_database_schema(
            connection_string="mssql+pyodbc://server/Policy_DB?driver=ODBC+Driver+17+for+SQL+Server"
        )

        # PHASE 2: Index metadata
        total_docs = workflow.phase_2_index_metadata(
            acord_file="./data/acord_standard.json",
            schema_file=schema_file,
            dictionary_file="./data/customer_dictionary.json"
        )

        # PHASE 3: Generate mappings (process first 3 tables for demo)
        mappings = workflow.phase_3_generate_mappings(
            schema_file=schema_file,
            top_n_tables=3
        )

        # PHASE 4: Review and approve
        approved_count = workflow.phase_4_review_and_approve(
            mappings=mappings,
            confidence_threshold=0.8
        )

        # PHASE 5: Generate ETL code
        etl_scripts = workflow.phase_5_generate_etl_code(mappings)

        # PHASE 6: Generate orchestration
        orchestration_file = workflow.phase_6_generate_orchestration(etl_scripts)

        # Print summary
        workflow.print_summary()

        print("\n✅ Harmonization workflow completed successfully!")
        print("\nNext steps:")
        print("1. Review generated ETL scripts in ./exports/")
        print("2. Test orchestration script in Databricks/Spark environment")
        print("3. Adjust confidence thresholds and prompts as needed")
        print("4. Deploy orchestration to production")

    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
