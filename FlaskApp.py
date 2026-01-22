# amit.dixit@inbravo
# Flask web application for Data Model Harmonization
# Extended with semantic mapping endpoints for insurance data harmonization
from flask import Flask, jsonify, redirect, render_template, request, url_for
from typing import List, Dict, Any
import json

from AppConfig import AppConfig
from retrieval.AcordSearcher import AcordSearcher
from mapping.MappingEngine import MappingEngine
from source_inspectors.MSSQLScanner import MSSQLScanner
from output_gen.SparkCodeGen import SparkCodeGen

# Initial components
AppConfig.initialize_components()

# Setup logging
logger = AppConfig.get_default_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global mapping engine instance (for single session)
mapping_engine = None
spark_code_gen = None


def get_mapping_engine():
    """Get or initialize the mapping engine."""
    global mapping_engine
    if mapping_engine is None:
        mapping_engine = MappingEngine(AppConfig.rag_retriever, AppConfig.llm_model)
    return mapping_engine


def get_spark_code_gen():
    """Get or initialize the code generator."""
    global spark_code_gen
    if spark_code_gen is None:
        spark_code_gen = SparkCodeGen(AppConfig.llm_model, enable_comments=True, enable_dq_checks=True)
    return spark_code_gen


# ============================================================================
# Original Routes (Document Q&A mode)
# ============================================================================

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/admin")
def admin():
    """Admin settings page."""
    return render_template(
        "admin.html",
        llm_model_name=AppConfig.LLM_MODEL_NAME,
        llm_model_type=AppConfig.LLM_MODEL_TYPE,
        embedding_model_name=AppConfig.EMBEDDING_MODEL_NAME,
        num_relevant_docs=AppConfig.NUM_RELEVANT_DOCS,
        openai_api_key=AppConfig.OPENAI_API_KEY,
    )


@app.route("/update_settings", methods=["POST"])
def update_settings():
    """Update application settings."""
    logger.info("Updating application settings...")
    AppConfig.update_components(
        request.form["llm_model_name"],
        request.form["llm_model_type"],
        request.form["embedding_model_name"],
        int(request.form["num_relevant_docs"]),
        request.form["openai_api_key"],
    )
    logger.info(
        f"Updated: model_type={AppConfig.LLM_MODEL_TYPE}, "
        f"model_name={AppConfig.LLM_MODEL_NAME}, "
        f"embedding={AppConfig.EMBEDDING_MODEL_NAME}"
    )
    return redirect(url_for("admin"))


@app.route("/query", methods=["POST"])
def query():
    """Handle document Q&A queries (original functionality)."""
    query_text = request.json["query_text"]
    logger.info("Received user query: %s", query_text)

    results = AppConfig.rag_retriever.query(query_text, k=AppConfig.NUM_RELEVANT_DOCS)
    enhanced_context_text, sources = AppConfig.rag_retriever.format_results(results)
    llm_response = AppConfig.llm_model.generate_response(
        context=enhanced_context_text, question=query_text
    )

    logger.info("Generated response for query")
    sources_html = "<br>".join(sources)
    response_text = f"{llm_response}<br><br>Sources:<br>{sources_html}<br><br>Response given by: {AppConfig.LLM_MODEL_NAME}"
    return jsonify(response=response_text)


# ============================================================================
# NEW: Harmonizer Routes (Data Model Mapping mode)
# ============================================================================

@app.route("/api/v1/mappings", methods=["GET"])
def get_all_mappings():
    """
    Get all mappings with optional filtering.

    Query parameters:
    - table: Filter by source table
    - status: Filter by mapping status (PENDING, APPROVED, etc.)
    - format: Response format ('json' or 'csv')

    Returns:
        JSON array of mappings
    """
    try:
        engine = get_mapping_engine()
        source_table = request.args.get("table")
        status = request.args.get("status")
        format_type = request.args.get("format", "json")

        mappings = engine.export_mappings(source_table=source_table, status=status)

        if format_type == "csv":
            # Convert to CSV format for download
            return format_mappings_as_csv(mappings)
        else:
            return jsonify({
                "total": len(mappings),
                "mappings": mappings
            })
    except Exception as e:
        logger.error(f"Error retrieving mappings: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/mappings/<source_table>/<source_column>", methods=["GET"])
def get_mapping(source_table, source_column):
    """
    Get a specific mapping by source table and column.

    Returns:
        Mapping details or 404 if not found
    """
    try:
        engine = get_mapping_engine()
        mapping = engine.get_mapping(source_table, source_column)

        if mapping:
            return jsonify(mapping.to_dict())
        else:
            return jsonify({"error": "Mapping not found"}), 404
    except Exception as e:
        logger.error(f"Error retrieving mapping: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/mappings/generate", methods=["POST"])
def generate_mapping():
    """
    Generate a mapping for a single source column.

    Request body:
    {
        "source_system": "Policy_DB",
        "source_table": "TBL_POLICY",
        "source_column": "EFF_DT",
        "data_type": "DATE",
        "description": "Policy effective date"
    }

    Returns:
        Generated ColumnMapping as JSON
    """
    try:
        data = request.json
        engine = get_mapping_engine()

        mapping = engine.generate_mapping(
            source_system=data.get("source_system", "unknown"),
            source_table=data.get("source_table"),
            source_column=data.get("source_column"),
            data_type=data.get("data_type", "unknown"),
            description=data.get("description", "")
        )

        logger.info(f"Generated mapping for {data['source_table']}.{data['source_column']}")
        return jsonify(mapping.to_dict()), 201
    except Exception as e:
        logger.error(f"Error generating mapping: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/mappings/batch", methods=["POST"])
def generate_batch_mappings():
    """
    Generate mappings for all columns in a table.

    Request body:
    {
        "source_system": "Policy_DB",
        "table_name": "TBL_POLICY",
        "columns": [
            {
                "column_name": "EFF_DT",
                "data_type": "DATE",
                "description": "Policy effective date"
            },
            ...
        ]
    }

    Returns:
        Array of generated mappings with summary statistics
    """
    try:
        data = request.json
        engine = get_mapping_engine()

        table_metadata = {
            "source_system": data.get("source_system", "unknown"),
            "table_name": data.get("table_name"),
            "columns": data.get("columns", [])
        }

        mappings = engine.batch_generate_mappings(table_metadata)
        summary = engine.get_mapping_summary(data.get("table_name"))

        logger.info(f"Generated {len(mappings)} mappings for table {data['table_name']}")
        return jsonify({
            "table": data.get("table_name"),
            "total_mappings": len(mappings),
            "mappings": [m.to_dict() for m in mappings],
            "summary": summary
        }), 201
    except Exception as e:
        logger.error(f"Error generating batch mappings: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/mappings/<source_table>/<source_column>/approve", methods=["POST"])
def approve_mapping(source_table, source_column):
    """
    Approve a mapping.

    Request body (optional):
    {
        "approved_by": "user@example.com",
        "notes": "Mapping looks good"
    }

    Returns:
        Updated mapping
    """
    try:
        data = request.json or {}
        engine = get_mapping_engine()

        mapping = engine.approve_mapping(
            source_table=source_table,
            source_column=source_column,
            approved_by=data.get("approved_by", "admin"),
            notes=data.get("notes", "")
        )

        logger.info(f"Approved mapping for {source_table}.{source_column}")
        return jsonify(mapping.to_dict())
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error approving mapping: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/mappings/<source_table>/<source_column>/reject", methods=["POST"])
def reject_mapping(source_table, source_column):
    """
    Reject a mapping.

    Request body:
    {
        "reason": "Mapping does not align with business requirements"
    }

    Returns:
        Updated mapping
    """
    try:
        data = request.json or {}
        engine = get_mapping_engine()

        mapping = engine.reject_mapping(
            source_table=source_table,
            source_column=source_column,
            reason=data.get("reason", "")
        )

        logger.info(f"Rejected mapping for {source_table}.{source_column}")
        return jsonify(mapping.to_dict())
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error rejecting mapping: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/mappings/summary", methods=["GET"])
def get_mapping_summary():
    """
    Get summary statistics on mappings.

    Query parameters:
    - table: Optional filter by source table

    Returns:
        Mapping statistics including approval percentage
    """
    try:
        engine = get_mapping_engine()
        source_table = request.args.get("table")

        summary = engine.get_mapping_summary(source_table=source_table)
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting mapping summary: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/codegen/transform", methods=["POST"])
def generate_spark_transformation():
    """
    Generate PySpark transformation code for an approved mapping.

    Request body:
    {
        "source_table": "TBL_POLICY",
        "source_column": "EFF_DT",
        "target_acord_entity": "Policy",
        "target_acord_field": "EffectiveDate",
        "source_data_type": "DATE",
        "target_data_type": "date"
    }

    Returns:
        Generated PySpark code
    """
    try:
        data = request.json
        code_gen = get_spark_code_gen()

        # Create a mock mapping object for code generation
        class MockMapping:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.reasoning = f"Transform {kwargs.get('source_column')} to {kwargs.get('target_acord_entity')}.{kwargs.get('target_acord_field')}"
                self.status = "APPROVED"

        mapping = MockMapping(**data)
        transformation = code_gen.generate_column_transformation(mapping)

        logger.info(f"Generated transformation for {data['source_column']}")
        return jsonify({
            "source_column": data.get("source_column"),
            "target_path": f"{data.get('target_acord_entity')}.{data.get('target_acord_field')}",
            "pyspark_code": transformation.pyspark_code,
            "include_dq_checks": transformation.include_qc_checks
        })
    except Exception as e:
        logger.error(f"Error generating transformation: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/codegen/table-etl", methods=["POST"])
def generate_table_etl():
    """
    Generate complete ETL script for a table.

    Request body:
    {
        "table_mappings": [
            {
                "source_table": "TBL_POLICY",
                "source_column": "EFF_DT",
                "target_acord_entity": "Policy",
                "target_acord_field": "EffectiveDate",
                ...
            }
        ]
    }

    Returns:
        Complete PySpark ETL script
    """
    try:
        data = request.json
        code_gen = get_spark_code_gen()

        # Create mock mapping objects
        class MockMapping:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.reasoning = f"Transform {kwargs.get('source_column')}"
                self.status = "APPROVED"

        mappings = [MockMapping(**m) for m in data.get("table_mappings", [])]
        etl_script = code_gen.generate_table_etl(mappings)

        logger.info(f"Generated ETL script for table")
        return jsonify({
            "etl_script": etl_script,
            "table_count": len(mappings)
        })
    except Exception as e:
        logger.error(f"Error generating ETL script: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/acord/search", methods=["GET"])
def search_acord():
    """
    Search the ACORD standard for entities and fields.

    Query parameters:
    - q: Search term
    - type: Document type filter ('entity', 'field', or empty for all)

    Returns:
        Array of ACORD entities/fields matching the search
    """
    try:
        search_term = request.args.get("q", "")
        doc_type = request.args.get("type", "")

        if not search_term:
            return jsonify({"error": "Search term required"}), 400

        searcher = AcordSearcher(AppConfig.rag_retriever)

        if doc_type == "entity":
            results = searcher.search_acord_entity(search_term, k=5)
        elif doc_type == "field":
            results = searcher.search_acord_field(search_term, k=5)
        else:
            # Search across all types
            cross_results = searcher.search_across_metadata_types(search_term, k=3)
            return jsonify(cross_results)

        formatted_results = [
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

        logger.info(f"ACORD search for: {search_term}")
        return jsonify({"results": formatted_results})
    except Exception as e:
        logger.error(f"Error searching ACORD: {e}")
        return jsonify({"error": str(e)}), 500


def format_mappings_as_csv(mappings: List[Dict]) -> str:
    """Format mappings as CSV for download."""
    import csv
    from io import StringIO

    output = StringIO()
    if not mappings:
        return "No mappings to export"

    fieldnames = ["source_table", "source_column", "target_acord_path", "confidence_score", "status"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for mapping in mappings:
        writer.writerow({
            "source_table": mapping.get("source_table"),
            "source_column": mapping.get("source_column"),
            "target_acord_path": mapping.get("target_acord_path"),
            "confidence_score": mapping.get("confidence_score"),
            "status": mapping.get("status")
        })

    return output.getvalue()


# ============================================================================
# Health check and utility endpoints
# ============================================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "llm_model": AppConfig.LLM_MODEL_NAME,
        "embedding_model": AppConfig.EMBEDDING_MODEL_NAME
    })


@app.route("/api/v1/health", methods=["GET"])
def api_health_check():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "api_version": "v1",
        "features": ["document_qa", "semantic_mapping", "code_generation"]
    })


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
