# Data Model Harmonizer: Architecture Transformation Guide

## Overview

This document describes the architectural transformation of the `rag-bot` codebase from a generic **document Q&A RAG application** into a specialized **Data Model Harmonizer** for insurance data (e.g., MSSQL to ACORD mapping).

The harmonizer enables automated semantic mapping between operational databases and industry standards through LLM-powered chain-of-thought reasoning, with built-in PySpark code generation for Lakehouse ETL pipelines.

---

## Key Architectural Shifts

### 1. Knowledge Base Evolution: From Documents to Structured Metadata

**Before:** Generic PDF/document indexing via `DocUploader.py`  
**After:** Three-tier metadata indexing via enhanced `DocUploader.py` + `MetadataLoader.py`

#### Three Types of Indexed Metadata:

1. **Target Standards (ACORD)**
   - Official ACORD logical model definitions
   - Entity and field specifications with business semantics
   - Indexed as `acord_standard` documents in ChromaDB

2. **Customer Dictionary**
   - Business term → Technical term mappings
   - Domain-specific glossaries from internal PDFs/Excel
   - Indexed as `customer_dictionary` documents

3. **Schema Metadata (SQL)**
   - Direct `INFORMATION_SCHEMA` exports from MSSQL
   - Column-level metadata: names, types, descriptions
   - Indexed as `sql_schema` documents

#### Implementation:
```python
from retrieval.MetadataLoader import MetadataLoader
from DocUploader import index_harmonizer_metadata

# Load all three metadata types
total_docs = index_harmonizer_metadata(
    embedding_model="openai",
    acord_json_path="data/acord_standard.json",
    schema_json_path="data/sql_schema.json",
    dictionary_json_path="data/customer_dictionary.json"
)
```

---

### 2. Retrieval Layer: Specialized ACORD Searching

**New Class:** `retrieval/AcordSearcher.py`

Extends the base `RAGRetriever` with ACORD-aware search:

```python
from retrieval.AcordSearcher import AcordSearcher

searcher = AcordSearcher(retriever)

# Search for ACORD entities
results = searcher.search_acord_entity("policy effective date", k=5)

# Find best mapping for a source column
best_target, confidence, alternatives = searcher.find_best_mapping(
    source_column_name="EFF_DT",
    source_description="Policy effective date from underwriting system",
    k=10
)
# Returns: ("Policy.EffectiveDate", 0.92, [("PolicyCoverage.EffectiveDate", 0.85), ...])
```

Key methods:
- `search_acord_entity()` - Find ACORD entities
- `search_acord_field()` - Find ACORD fields
- `find_best_mapping()` - Semantic mapping with ranking
- `search_across_metadata_types()` - Cross-metadata discovery
- `validate_mapping_consistency()` - Verify mapping validity

---

### 3. SQL Schema Inspection: Direct Database Access

**New Class:** `source_inspectors/MSSQLScanner.py`

Connects directly to MSSQL databases and extracts metadata:

```python
from source_inspectors.MSSQLScanner import MSSQLScanner

scanner = MSSQLScanner(
    connection_string="mssql+pyodbc://user:pass@server/database?driver=ODBC+Driver+17+for+SQL+Server"
)

# Export entire schema to JSON
schema_json = scanner.get_schema_json()

# Or scan specific tables
table_info = scanner.scan_table("TBL_POLICY")

# Export to file for later indexing
scanner.export_schema_to_json_file("exports/policy_schema.json")

scanner.close()
```

Key features:
- Recursive database scanning
- Column-level metadata extraction (including extended properties)
- JSON export compatible with `MetadataLoader`
- Support for multiple databases in one session
- SQLAlchemy-based for database agnostic design

---

### 4. Semantic Mapping Engine: The Core Harmonizer

**New Class:** `mapping/MappingEngine.py`

Orchestrates the entire mapping workflow:

```python
from mapping.MappingEngine import MappingEngine

engine = MappingEngine(
    retriever=AppConfig.rag_retriever,
    llm_model=AppConfig.llm_model,
    enable_chain_of_thought=True
)

# Generate single mapping
mapping = engine.generate_mapping(
    source_system="Policy_DB",
    source_table="TBL_POLICY",
    source_column="EFF_DT",
    data_type="DATE",
    description="Policy effective date from underwriting system"
)
# Returns ColumnMapping object with:
# - target_acord_entity, target_acord_field
# - confidence_score (0-1)
# - reasoning (LLM explanation)
# - alternative_mappings

# Batch generate for entire table
table_metadata = {
    "source_system": "Policy_DB",
    "table_name": "TBL_POLICY",
    "columns": [
        {"column_name": "EFF_DT", "data_type": "DATE", "description": "..."},
        {"column_name": "RENEWAL_DT", "data_type": "DATE", "description": "..."},
        # ...
    ]
}
mappings = engine.batch_generate_mappings(table_metadata)

# Workflow
engine.approve_mapping("TBL_POLICY", "EFF_DT", approved_by="analyst@company.com")
engine.get_mapping_summary("TBL_POLICY")
# Returns: {
#     "total_mappings": 25,
#     "approved": 18,
#     "pending": 5,
#     "requires_review": 2,
#     "average_confidence": 0.87,
#     "completion_percentage": 72.0
# }
```

### Mapping Status Lifecycle:
- **PENDING**: Generated, confidence > 0.5, awaiting review
- **REQUIRES_REVIEW**: Generated, confidence ≤ 0.5, needs manual intervention
- **APPROVED**: Validated and approved by user
- **REJECTED**: User rejected, alternative mapping needed

---

### 5. Prompt Architecture: Chain-of-Thought for Semantics

**New Module:** `mapping_prompts/HarmonizationPrompts.py`

Replaces generic Salesforce prompts with insurance-domain prompts:

#### Semantic Mapping Prompt:
```
You are an Insurance Data Architect. Given the source SQL column {source_column} 
with description {source_desc}, search the ACORD standard and customer dictionary. 
Suggest the best target ACORD entity.

**Reasoning:** Explain why you chose this mapping.
**Confidence:** Provide a score from 0-100.
```

#### Chain-of-Thought Prompt (for complex columns):
```
**STEP 1: Analyze the Source Column**
**STEP 2: Understand the ACORD Standard**
**STEP 3: Evaluate Mapping Candidates**
**STEP 4: Make Your Recommendation**
```

#### Code Generation Prompt:
```
Based on the approved data mapping, generate a PySpark code snippet for transforming 
the source data to match the ACORD target format.
```

---

### 6. PySpark Code Generation: From Mapping to ETL

**New Class:** `output_gen/SparkCodeGen.py`

Translates approved mappings into production-ready PySpark code:

```python
from output_gen.SparkCodeGen import SparkCodeGen

codegen = SparkCodeGen(
    llm_model=AppConfig.llm_model,
    enable_comments=True,
    enable_dq_checks=True
)

# Generate transformation for single column
transformation = codegen.generate_column_transformation(mapping)
# Returns code like:
# df_gold = df_bronze.withColumn("Policy_EffectiveDate", col("EFF_DT").cast("date"))

# Generate complete table ETL
etl_script = codegen.generate_table_etl(all_mappings_for_table)

# Generate orchestration script
orchestration = codegen.generate_orchestration_script(
    etl_scripts={
        "TBL_POLICY": policy_etl,
        "TBL_COVERAGE": coverage_etl,
    },
    output_format="job"  # or "notebook"
)
```

Features:
- Intelligent type conversion logic
- Data quality checks (null handling, type validation)
- Databricks job orchestration templates
- Code validation (basic syntax check)
- Comments and documentation

---

## Unified Call Flow: The Harmonizer Pipeline

```
1. INGEST
   ├─ DocUploader loads PDFs (traditional docs)
   ├─ MetadataLoader indexes ACORD standard
   ├─ MetadataLoader indexes SQL schema from MSSQLScanner
   └─ MetadataLoader indexes customer dictionary
   
2. INSPECT
   ├─ MSSQLScanner connects to source databases
   ├─ Extracts INFORMATION_SCHEMA for each table
   └─ Exports schema JSON for indexing
   
3. RETRIEVE
   ├─ AcordSearcher performs semantic search
   ├─ Finds candidate ACORD mappings
   └─ Ranks by confidence and consistency
   
4. HARMONIZE
   ├─ MappingEngine invokes LLM with chain-of-thought
   ├─ LLM reasons through mapping options
   ├─ Returns mapping with explanation and confidence
   └─ Alternative options provided
   
5. REVIEW
   ├─ User reviews mappings in UI table
   ├─ Can approve, reject, or request alternatives
   └─ Confidence scores guide prioritization
   
6. BUILD
   ├─ SparkCodeGen generates PySpark code
   ├─ Creates data quality checks
   └─ Produces orchestration script for Lakehouse
```

---

## New API Endpoints

### Mapping Management

#### Get All Mappings
```
GET /api/v1/mappings?table=TBL_POLICY&status=PENDING&format=json
```

#### Generate Single Mapping
```
POST /api/v1/mappings/generate
{
    "source_system": "Policy_DB",
    "source_table": "TBL_POLICY",
    "source_column": "EFF_DT",
    "data_type": "DATE",
    "description": "Policy effective date"
}
```

#### Generate Batch Mappings
```
POST /api/v1/mappings/batch
{
    "source_system": "Policy_DB",
    "table_name": "TBL_POLICY",
    "columns": [...]
}
```

#### Approve/Reject Mappings
```
POST /api/v1/mappings/{table}/{column}/approve
POST /api/v1/mappings/{table}/{column}/reject
```

#### Get Mapping Summary
```
GET /api/v1/mappings/summary?table=TBL_POLICY
```

### Code Generation

#### Generate Column Transformation
```
POST /api/v1/codegen/transform
{
    "source_table": "TBL_POLICY",
    "source_column": "EFF_DT",
    "target_acord_entity": "Policy",
    "target_acord_field": "EffectiveDate",
    "source_data_type": "DATE",
    "target_data_type": "date"
}
```

#### Generate Table ETL
```
POST /api/v1/codegen/table-etl
{
    "table_mappings": [...]
}
```

### ACORD Search
```
GET /api/v1/acord/search?q=policy+effective&type=field
```

---

## Data Model: ColumnMapping

```python
@dataclass
class ColumnMapping:
    source_system: str               # e.g., "Policy_DB"
    source_table: str                # e.g., "TBL_POLICY"
    source_column: str               # e.g., "EFF_DT"
    source_data_type: str            # e.g., "DATE"
    source_description: str          # Business definition
    target_acord_entity: str         # e.g., "Policy"
    target_acord_field: str          # e.g., "EffectiveDate"
    confidence_score: float          # 0-1, calculated from semantic search
    reasoning: str                   # LLM explanation of why this mapping
    alternative_mappings: List[Tuple[str, float]]  # [(path, confidence), ...]
    status: str                      # PENDING, APPROVED, REJECTED, REQUIRES_REVIEW
    created_at: str                  # ISO timestamp
    approved_by: str                 # User who approved
    notes: str                       # Optional approval/rejection notes
```

---

## Folder Structure (Updated)

```
rag-bot/
├── AppConfig.py
├── DocUploader.py              # Enhanced with metadata loading
├── FlaskApp.py                 # Enhanced with mapping endpoints
├── requirements.txt
├── retrieval/
│   ├── RAGRetriever.py
│   ├── MetadataLoader.py       # NEW: Converts metadata to documents
│   ├── AcordSearcher.py        # NEW: ACORD-aware searching
│   └── MetadataLoader.py
├── mapping/                    # NEW directory
│   └── MappingEngine.py        # NEW: Core harmonization orchestrator
├── source_inspectors/          # NEW directory
│   └── MSSQLScanner.py         # NEW: Database schema extraction
├── output_gen/                 # NEW directory
│   └── SparkCodeGen.py         # NEW: PySpark code generation
├── mapping_prompts/            # NEW directory
│   └── HarmonizationPrompts.py # NEW: Domain-specific prompts
├── llm/
│   ├── llm.py
│   └── llm_factory.py
├── embeddings/
│   ├── EmbeddingFactory.py
│   └── OpenAIEmbeddings.py
├── templates/
│   ├── index.html
│   └── admin.html
├── static/
│   ├── admin_settings.js
│   └── styles.css
└── tests/
    └── test_one.py
```

---

## Usage Examples

### Example 1: Complete Harmonization Workflow

```python
from retrieval.MetadataLoader import MetadataLoader
from source_inspectors.MSSQLScanner import MSSQLScanner
from mapping.MappingEngine import MappingEngine
from output_gen.SparkCodeGen import SparkCodeGen
from AppConfig import AppConfig

AppConfig.initialize_components()

# Step 1: Scan source database
scanner = MSSQLScanner("mssql+pyodbc://server/Policy_DB")
schema_json = scanner.get_schema_json()
scanner.export_schema_to_json_file("policy_schema.json")

# Step 2: Index metadata
from DocUploader import index_harmonizer_metadata
total_docs = index_harmonizer_metadata(
    acord_json_path="acord_standard.json",
    schema_json_path="policy_schema.json",
    dictionary_json_path="customer_dictionary.json"
)

# Step 3: Generate mappings
engine = MappingEngine(AppConfig.rag_retriever, AppConfig.llm_model)
table_metadata = {
    "source_system": "Policy_DB",
    "table_name": "TBL_POLICY",
    "columns": [
        {"column_name": "EFF_DT", "data_type": "DATE", "description": "Effective date"},
        {"column_name": "RENEWAL_DT", "data_type": "DATE", "description": "Renewal date"},
        # ...
    ]
}
mappings = engine.batch_generate_mappings(table_metadata)

# Step 4: Review and approve (in UI, then in code)
for mapping in mappings:
    if mapping.confidence_score > 0.8:
        engine.approve_mapping(mapping.source_table, mapping.source_column)

# Step 5: Generate ETL code
codegen = SparkCodeGen(AppConfig.llm_model)
etl_script = codegen.generate_table_etl(
    [m for m in mappings if m.status == "APPROVED"]
)
codegen.export_code_to_file(etl_script, "policy_etl.py")
```

### Example 2: REST API Usage

```bash
# Generate mapping for a column
curl -X POST http://localhost:5000/api/v1/mappings/generate \
  -H "Content-Type: application/json" \
  -d '{
    "source_system": "Policy_DB",
    "source_table": "TBL_POLICY",
    "source_column": "EFF_DT",
    "data_type": "DATE",
    "description": "Policy effective date"
  }'

# Approve the mapping
curl -X POST http://localhost:5000/api/v1/mappings/TBL_POLICY/EFF_DT/approve \
  -H "Content-Type: application/json" \
  -d '{"approved_by": "analyst@company.com"}'

# Generate PySpark code
curl -X POST http://localhost:5000/api/v1/codegen/transform \
  -H "Content-Type: application/json" \
  -d '{
    "source_table": "TBL_POLICY",
    "source_column": "EFF_DT",
    "target_acord_entity": "Policy",
    "target_acord_field": "EffectiveDate",
    "source_data_type": "DATE",
    "target_data_type": "date"
  }'
```

---

## Dependencies Added

```
sqlalchemy          # Database inspection and ORM
pyodbc             # MSSQL connectivity
```

Add to `requirements.txt`:
```
sqlalchemy>=2.0.0
pyodbc>=4.0.35
```

---

## Configuration

### Environment Variables

Add to `.env`:
```
# MSSQL Connection Strings
MSSQL_POLICY_DB=mssql+pyodbc://user:pass@server/Policy_DB?driver=ODBC+Driver+17+for+SQL+Server
MSSQL_COVERAGE_DB=mssql+pyodbc://user:pass@server/Coverage_DB?driver=ODBC+Driver+17+for+SQL+Server

# Metadata Paths
ACORD_STANDARD_PATH=./data/acord_standard.json
CUSTOMER_DICTIONARY_PATH=./data/customer_dictionary.json
```

---

## JSON Data Formats

### ACORD Standard JSON

```json
{
  "entities": [
    {
      "entity_name": "Policy",
      "description": "The insurance policy entity",
      "fields": [
        {
          "field_name": "EffectiveDate",
          "data_type": "Date",
          "description": "The date the policy becomes effective",
          "required": true
        }
      ]
    }
  ]
}
```

### SQL Schema JSON

```json
{
  "database": "Policy_DB",
  "tables": [
    {
      "table_name": "TBL_POLICY",
      "description": "Master policy table",
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
```

### Customer Dictionary JSON

```json
{
  "terms": [
    {
      "business_term": "Policy Effective Date",
      "technical_term": "EFF_DT",
      "definition": "The date when the policy becomes effective for coverage",
      "system": "Policy_DB",
      "related_systems": ["Coverage_DB", "Claims_DB"]
    }
  ]
}
```

---

## Testing & Validation

### Unit Tests

```python
# Test MappingEngine
def test_mapping_generation():
    engine = MappingEngine(retriever, llm_model)
    mapping = engine.generate_mapping(
        "Policy_DB", "TBL_POLICY", "EFF_DT", "DATE", "Policy effective date"
    )
    assert mapping.target_acord_entity == "Policy"
    assert mapping.confidence_score > 0

# Test SparkCodeGen
def test_code_generation():
    codegen = SparkCodeGen(llm_model)
    transformation = codegen.generate_column_transformation(mapping)
    is_valid, warnings = codegen.validate_generated_code(transformation.pyspark_code)
    assert is_valid
```

### Validation Checklist

- [ ] All three metadata types indexed (ACORD, SQL schema, dictionary)
- [ ] Mapping confidence scores in 0-1 range
- [ ] Generated PySpark code passes basic syntax validation
- [ ] API endpoints return proper JSON responses
- [ ] Mapping approval/rejection workflow functional
- [ ] Code generation respects data types
- [ ] Data quality checks included in generated code

---

## Performance Considerations

1. **Semantic Search:** Cache frequently accessed ACORD definitions
2. **LLM Invocation:** Batch mapping generation for large tables
3. **ChromaDB:** Use vector DB filters (metadata) to reduce search space
4. **SQL Scanning:** Implement connection pooling for multiple databases

---

## Future Enhancements

1. **Advanced ML Ranking:** Train classifier on approved mappings
2. **Custom Dictionary Learning:** Improve suggestions based on approved mappings
3. **Multi-Standard Support:** Extend beyond ACORD (e.g., NAIC)
4. **Incremental Indexing:** Add/update metadata without re-indexing
5. **UI Dashboard:** Interactive mapping review and approval interface
6. **Audit Logging:** Track all mapping decisions for compliance
7. **Conflict Detection:** Identify conflicting mappings across tables

---

## Summary

This transformation converts `rag-bot` from a generic document Q&A system into an enterprise-grade **Data Model Harmonizer** by:

1. **Structuring the knowledge base** into ACORD standards, SQL schemas, and business dictionaries
2. **Adding specialized retrieval** for ACORD-aware semantic search
3. **Enabling database inspection** to understand source systems
4. **Orchestrating mappings** through chain-of-thought LLM reasoning
5. **Generating code** to implement harmonized data models in Lakehouse ETL

The result is a powerful tool for insurance companies to rapidly harmonize disparate data sources toward industry-standard logical models like ACORD.
