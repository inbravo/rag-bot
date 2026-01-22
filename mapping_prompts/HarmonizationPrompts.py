from AppConfig import AppConfig

# Initialize application components
AppConfig.initialize_components()

# Semantic Mapping Prompt Template for Insurance Data Harmonization
SEMANTIC_MAPPING_PROMPT = """
You are an Insurance Data Architect specializing in data model harmonization between operational systems and ACORD standards.

Given the following source SQL metadata:
- Source System: {source_system}
- Source Table: {source_table}
- Source Column: {source_column}
- Data Type: {data_type}
- Column Description: {source_desc}

Your task is to:
1. Search the provided ACORD standard context and customer business dictionaries.
2. Suggest the BEST matching target ACORD entity/field.
3. Provide your reasoning for this mapping.
4. Rate your confidence on a scale of 0-100.

Format your response as follows:
MAPPING_SUGGESTION: <ACORD entity path, e.g., Policy.EffectiveDate>
REASONING: <Brief explanation of why this mapping is appropriate>
CONFIDENCE_SCORE: <0-100>
ALTERNATIVE_MAPPINGS: <List any reasonable alternatives with their confidence scores>

Context:
{context}

---
Answer:
"""

# Advanced Chain-of-Thought Mapping Prompt (for complex columns)
CHAIN_OF_THOUGHT_MAPPING_PROMPT = """
You are an Insurance Data Architect specializing in complex data model transformations.

**STEP 1: Analyze the Source Column**
- System: {source_system}
- Table: {source_table}
- Column: {source_column}
- Data Type: {data_type}
- Business Definition: {source_desc}

**STEP 2: Understand the ACORD Standard**
Review the provided ACORD documentation context to understand:
- The target logical data model
- Available entity types and their fields
- Business semantics of ACORD attributes

**STEP 3: Evaluate Mapping Candidates**
For each potential ACORD target:
- Assess semantic similarity
- Check data type compatibility
- Validate business context alignment

**STEP 4: Make Your Recommendation**
Choose the best mapping based on:
1. Semantic alignment (70% weight)
2. Data type compatibility (15% weight)
3. Business context fit (15% weight)

Context:
{context}

---
Your detailed reasoning and recommendation:
"""

# Batch Mapping Validation Prompt
BATCH_MAPPING_VALIDATION_PROMPT = """
You are an Insurance Data Architect validating a batch of proposed data mappings.

Review the following proposed mappings and validate each one:

{proposed_mappings}

For each mapping, provide:
1. VALIDATION_STATUS: (VALID | REQUIRES_REVIEW | INVALID)
2. REASONING: Brief explanation
3. SUGGESTED_ALTERNATIVES: If status is REQUIRES_REVIEW or INVALID

ACORD Standard Context:
{context}

---
Validation Results:
"""

# Code Generation Guidance Prompt
CODE_GENERATION_PROMPT = """
You are a PySpark ETL code generator for insurance data transformations.

Based on the approved data mapping, generate a PySpark code snippet for transforming the source data to match the ACORD target format.

Mapping Details:
- Source Table: {source_table}
- Source Column: {source_column}
- Target ACORD Entity: {target_entity}
- Data Type Conversion: From {source_data_type} to {target_data_type}
- Business Logic: {business_logic}

Requirements:
1. Use PySpark DataFrame API
2. Include data quality checks (null handling, type validation)
3. Add comments explaining the transformation
4. Follow naming conventions: df_bronze (source) → df_gold (target)

Generate the code:
"""

# Configuration mapping prompt for testing
def get_semantic_mapping_prompt(source_system: str, source_table: str, source_column: str,
                                 data_type: str, source_desc: str, context: str) -> str:
    """
    Generate a semantic mapping prompt for a specific source column.

    Args:
        source_system: Name of the source system (e.g., 'Policy_DB')
        source_table: Name of the source table
        source_column: Name of the source column
        data_type: SQL data type of the column
        source_desc: Business description of the column
        context: Retrieved ACORD and dictionary context from RAG

    Returns:
        Formatted prompt ready for LLM invocation
    """
    return SEMANTIC_MAPPING_PROMPT.format(
        source_system=source_system,
        source_table=source_table,
        source_column=source_column,
        data_type=data_type,
        source_desc=source_desc,
        context=context
    )


def get_chain_of_thought_prompt(source_system: str, source_table: str, source_column: str,
                                 data_type: str, source_desc: str, context: str) -> str:
    """
    Generate a chain-of-thought mapping prompt for complex columns.

    Args:
        source_system: Name of the source system
        source_table: Name of the source table
        source_column: Name of the source column
        data_type: SQL data type
        source_desc: Business description
        context: Retrieved ACORD context

    Returns:
        Formatted prompt with step-by-step reasoning
    """
    return CHAIN_OF_THOUGHT_MAPPING_PROMPT.format(
        source_system=source_system,
        source_table=source_table,
        source_column=source_column,
        data_type=data_type,
        source_desc=source_desc,
        context=context
    )


def get_code_generation_prompt(source_table: str, source_column: str, target_entity: str,
                               source_data_type: str, target_data_type: str,
                               business_logic: str = "") -> str:
    """
    Generate a prompt for PySpark code generation based on an approved mapping.

    Args:
        source_table: Source table name
        source_column: Source column name
        target_entity: Target ACORD entity path
        source_data_type: Source data type
        target_data_type: Target data type
        business_logic: Optional business logic description

    Returns:
        Formatted prompt for code generation
    """
    return CODE_GENERATION_PROMPT.format(
        source_table=source_table,
        source_column=source_column,
        target_entity=target_entity,
        source_data_type=source_data_type,
        target_data_type=target_data_type,
        business_logic=business_logic if business_logic else "Direct type conversion"
    )


# Test function to validate semantic mapping generation
def test_semantic_mapping(source_system: str, source_column: str, expected_acord_entity: str):
    """
    Test function to validate semantic mapping generation for a source column.

    Args:
        source_system: Name of source system
        source_column: Name of source column
        expected_acord_entity: Expected ACORD entity for validation
    """
    # This would be used in integration tests
    pass
