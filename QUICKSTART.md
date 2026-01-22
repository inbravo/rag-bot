# Quick Start Guide: Data Model Harmonizer

Get up and running with the Data Model Harmonizer in 10 minutes.

## Prerequisites

- Python 3.9+
- ChromaDB vector database installed
- OpenAI API key or Ollama running locally
- MSSQL ODBC driver (for database scanning)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install sqlalchemy pyodbc
```

### 2. Update `.env` File

```env
# Existing settings
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_MODEL_TYPE=gpt
EMBEDDING_MODEL_NAME=openai
OPENAI_API_KEY=your_api_key_here

# New harmonizer settings
MSSQL_CONNECTION=mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
```

### 3. Prepare Metadata Files

Create three JSON files in your `data/` folder:

**`data/acord_standard.json`** - ACORD definitions
```json
{
  "entities": [
    {
      "entity_name": "Policy",
      "description": "Insurance policy",
      "fields": [
        {
          "field_name": "EffectiveDate",
          "data_type": "Date",
          "description": "When policy becomes effective",
          "required": true
        }
      ]
    }
  ]
}
```

**`data/sql_schema.json`** - From MSSQLScanner export (or manually created)
```json
{
  "database": "Policy_DB",
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
```

**`data/customer_dictionary.json`** - Business glossary
```json
{
  "terms": [
    {
      "business_term": "Policy Effective Date",
      "technical_term": "EFF_DT",
      "definition": "Date when policy coverage begins",
      "system": "Policy_DB",
      "related_systems": []
    }
  ]
}
```

## Quick Start Workflow

### Step 1: Index Metadata

```python
from DocUploader import index_harmonizer_metadata

# Load and index all three metadata types
total_docs = index_harmonizer_metadata(
    embedding_model="openai",
    acord_json_path="data/acord_standard.json",
    schema_json_path="data/sql_schema.json",
    dictionary_json_path="data/customer_dictionary.json"
)
print(f"Indexed {total_docs} documents")
```

### Step 2: Generate Mappings

```python
from AppConfig import AppConfig
from mapping.MappingEngine import MappingEngine

AppConfig.initialize_components()

engine = MappingEngine(
    retriever=AppConfig.rag_retriever,
    llm_model=AppConfig.llm_model,
    enable_chain_of_thought=True
)

# Generate mapping for a single column
mapping = engine.generate_mapping(
    source_system="Policy_DB",
    source_table="TBL_POLICY",
    source_column="EFF_DT",
    data_type="DATE",
    description="Policy effective date"
)

print(f"Target: {mapping.target_acord_entity}.{mapping.target_acord_field}")
print(f"Confidence: {mapping.confidence_score:.2%}")
print(f"Reasoning: {mapping.reasoning}")
```

### Step 3: Approve Mapping

```python
# Approve high-confidence mappings
if mapping.confidence_score > 0.8:
    engine.approve_mapping(
        source_table="TBL_POLICY",
        source_column="EFF_DT",
        approved_by="analyst@company.com"
    )
    print("Mapping approved")
```

### Step 4: Generate PySpark Code

```python
from output_gen.SparkCodeGen import SparkCodeGen

codegen = SparkCodeGen(AppConfig.llm_model)

# Generate transformation
transformation = codegen.generate_column_transformation(mapping)
print(transformation.pyspark_code)
```

## REST API Examples

### Start Flask App

```bash
python FlaskApp.py
```

### Generate a Mapping

```bash
curl -X POST http://localhost:5000/api/v1/mappings/generate \
  -H "Content-Type: application/json" \
  -d '{
    "source_system": "Policy_DB",
    "source_table": "TBL_POLICY",
    "source_column": "EFF_DT",
    "data_type": "DATE",
    "description": "Policy effective date from underwriting"
  }'
```

**Response:**
```json
{
  "source_system": "Policy_DB",
  "source_table": "TBL_POLICY",
  "source_column": "EFF_DT",
  "target_acord_entity": "Policy",
  "target_acord_field": "EffectiveDate",
  "confidence_score": 0.92,
  "status": "PENDING",
  "reasoning": "EFF_DT directly maps to Policy.EffectiveDate based on semantic similarity..."
}
```

### Batch Generate Mappings

```bash
curl -X POST http://localhost:5000/api/v1/mappings/batch \
  -H "Content-Type: application/json" \
  -d '{
    "source_system": "Policy_DB",
    "table_name": "TBL_POLICY",
    "columns": [
      {"column_name": "EFF_DT", "data_type": "DATE", "description": "Effective date"},
      {"column_name": "RENEWAL_DT", "data_type": "DATE", "description": "Renewal date"}
    ]
  }'
```

### Approve a Mapping

```bash
curl -X POST http://localhost:5000/api/v1/mappings/TBL_POLICY/EFF_DT/approve \
  -H "Content-Type: application/json" \
  -d '{"approved_by": "analyst@company.com", "notes": "Mapping verified"}'
```

### Get Mapping Summary

```bash
curl http://localhost:5000/api/v1/mappings/summary?table=TBL_POLICY
```

**Response:**
```json
{
  "total_mappings": 25,
  "approved": 20,
  "pending": 3,
  "requires_review": 2,
  "rejected": 0,
  "average_confidence": 0.87,
  "completion_percentage": 80.0
}
```

### Generate PySpark Code

```bash
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

**Response:**
```json
{
  "source_column": "EFF_DT",
  "target_path": "Policy.EffectiveDate",
  "pyspark_code": "df_gold = df_bronze.withColumn('Policy_EffectiveDate', col('EFF_DT').cast('date'))",
  "include_dq_checks": true
}
```

## Advanced: Scan Database Directly

```python
from source_inspectors.MSSQLScanner import MSSQLScanner

# Connect to MSSQL
scanner = MSSQLScanner("mssql+pyodbc://server/Policy_DB")

# Export entire schema
schema = scanner.get_schema_json()
scanner.export_schema_to_json_file("policy_schema.json")

# Or scan specific table
table = scanner.scan_table("TBL_POLICY")
for col in table.columns:
    print(f"  {col.column_name}: {col.data_type}")

scanner.close()
```

## Batch Processing

### Generate Mappings for Multiple Tables

```python
from retrieval.MetadataLoader import MetadataLoader

engine = MappingEngine(AppConfig.rag_retriever, AppConfig.llm_model)

# Tables to process
tables = ["TBL_POLICY", "TBL_COVERAGE", "TBL_CLAIM"]

all_mappings = []
for table_name in tables:
    table_metadata = {
        "source_system": "Policy_DB",
        "table_name": table_name,
        "columns": [...]  # Load from schema
    }
    mappings = engine.batch_generate_mappings(table_metadata)
    all_mappings.extend(mappings)

# Export all mappings
export_data = engine.export_mappings()
```

### Generate Complete ETL

```python
from output_gen.SparkCodeGen import SparkCodeGen

codegen = SparkCodeGen(AppConfig.llm_model)

# Get approved mappings
approved_mappings = [m for m in all_mappings if m.status == "APPROVED"]

# Generate ETL script
etl_script = codegen.generate_table_etl(approved_mappings)

# Export to file
with open("policy_etl.py", "w") as f:
    f.write(etl_script)

# Generate orchestration
etl_scripts = {"TBL_POLICY": etl_script}
orchestration = codegen.generate_orchestration_script(
    etl_scripts,
    output_format="job"
)

with open("orchestration.py", "w") as f:
    f.write(orchestration)
```

## Troubleshooting

### Issue: "No matching ACORD standard found"
**Solution:** Ensure ACORD JSON is indexed correctly
```bash
curl http://localhost:5000/api/v1/acord/search?q=policy
```

### Issue: "Column description not found"
**Solution:** SQL Server extended properties may not be set. Add descriptions:
```sql
EXEC sp_addextendedproperty 
  @name = N'MS_Description', 
  @value = N'Policy effective date', 
  @level0type = N'SCHEMA', @level0name = 'dbo',
  @level1type = N'TABLE', @level1name = 'TBL_POLICY',
  @level2type = N'COLUMN', @level2name = 'EFF_DT'
```

### Issue: ODBC connection error
**Solution:** Install ODBC driver and verify connection string
```bash
# On Windows
# Download: Microsoft ODBC Driver for SQL Server

# On Linux
sudo apt-get install odbc-mssql unixodbc-dev

# Test connection
isql -v YourDSN username password
```

### Issue: "LLM rate limit exceeded"
**Solution:** Reduce batch size or add delays
```python
import time
for mapping in mappings:
    time.sleep(0.5)  # Add delay between LLM calls
```

## Next Steps

1. **Customize Prompts:** Edit `mapping_prompts/HarmonizationPrompts.py` for your domain
2. **Fine-tune Confidence:** Adjust weights in `AcordSearcher.rank_mappings_by_confidence()`
3. **Add Data Quality:** Extend `SparkCodeGen._add_dq_checks()` with business rules
4. **Build UI:** Create React/Vue dashboard using `/api/v1/mappings/*` endpoints
5. **Integrate with Lakehouse:** Export generated code to Databricks or Spark

## Documentation

- **Full Architecture:** See [HARMONIZER_ARCHITECTURE.md](HARMONIZER_ARCHITECTURE.md)
- **API Reference:** See docstrings in `FlaskApp.py`
- **Data Formats:** See JSON examples in architecture doc

---

**Need Help?** Check the example scripts in `/tests/` or review docstrings in source files.
