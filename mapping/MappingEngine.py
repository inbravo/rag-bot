from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from retrieval.RAGRetriever import RAGRetriever
from retrieval.AcordSearcher import AcordSearcher
from mapping_prompts.HarmonizationPrompts import get_semantic_mapping_prompt, get_chain_of_thought_prompt

# Data class for mapping results
@dataclass
class ColumnMapping:
    """Represents a data mapping from source to target."""
    source_system: str
    source_table: str
    source_column: str
    source_data_type: str
    source_description: str
    target_acord_entity: str
    target_acord_field: str
    confidence_score: float
    reasoning: str
    alternative_mappings: List[Tuple[str, float]]
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED, REQUIRES_REVIEW
    created_at: str = None
    approved_by: str = None
    notes: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['target_acord_path'] = f"{self.target_acord_entity}.{self.target_acord_field}"
        data['alternative_mappings'] = [
            {"path": path, "confidence": conf} for path, conf in self.alternative_mappings
        ]
        return data


# MappingEngine class to coordinate mapping between SQL metadata and RAG context
class MappingEngine:
    """
    The core orchestrator for semantic data mapping between source SQL schemas
    and target ACORD standards.

    This engine coordinates:
    1. SQL schema inspection (via MSSQLScanner)
    2. Semantic retrieval (via RAGRetriever and AcordSearcher)
    3. LLM-based harmonization (via prompts)
    4. Mapping validation and storage

    Attributes:
        retriever (RAGRetriever): The underlying RAG retriever
        acord_searcher (AcordSearcher): Specialized ACORD searcher
        llm_model: The language model for generating mappings
        mappings (List[ColumnMapping]): Accumulated mappings

    Methods:
        generate_mapping(source_system, source_table, source_column, data_type, description):
            Generate a single mapping using semantic search and LLM reasoning.

        batch_generate_mappings(table_metadata):
            Generate mappings for all columns in a table.

        approve_mapping(source_table, source_column, approved_by):
            Mark a mapping as approved.

        get_mapping_summary(source_table):
            Get statistics on mappings for a table.
    """

    def __init__(self, retriever: RAGRetriever, llm_model, enable_chain_of_thought: bool = True):
        """
        Initialize the MappingEngine.

        Args:
            retriever: Initialized RAGRetriever instance
            llm_model: Initialized LLM model instance
            enable_chain_of_thought: Use advanced chain-of-thought prompts for complex columns
        """
        self.retriever = retriever
        self.acord_searcher = AcordSearcher(retriever)
        self.llm_model = llm_model
        self.enable_chain_of_thought = enable_chain_of_thought
        self.mappings: Dict[str, ColumnMapping] = {}
        self.mapping_history: List[ColumnMapping] = []

    def generate_mapping(self, source_system: str, source_table: str, source_column: str,
                         data_type: str, description: str, use_llm: bool = True) -> ColumnMapping:
        """
        Generate a semantic mapping for a single source column.

        The process:
        1. Use AcordSearcher to find candidate ACORD targets
        2. Optionally use LLM to refine and explain the choice
        3. Calculate confidence scores
        4. Identify alternative mappings

        Args:
            source_system: Name of source system (e.g., 'Policy_DB')
            source_table: Source table name
            source_column: Source column name
            data_type: SQL data type
            description: Column business description
            use_llm: Whether to use LLM for reasoning (default True)

        Returns:
            ColumnMapping object with mapping recommendation
        """
        # Step 1: Semantic search for ACORD candidates
        best_target, confidence, alternatives = self.acord_searcher.find_best_mapping(
            source_column, description, k=10
        )

        if not best_target:
            # Fallback to basic search if no ACORD match found
            return ColumnMapping(
                source_system=source_system,
                source_table=source_table,
                source_column=source_column,
                source_data_type=data_type,
                source_description=description,
                target_acord_entity="UNKNOWN",
                target_acord_field="UNKNOWN",
                confidence_score=0.0,
                reasoning="No matching ACORD standard found",
                alternative_mappings=[],
                status="REQUIRES_REVIEW"
            )

        # Parse the best target
        target_parts = best_target.split(".")
        target_entity = target_parts[0] if len(target_parts) > 0 else ""
        target_field = target_parts[1] if len(target_parts) > 1 else ""

        reasoning = ""

        # Step 2: Use LLM to generate detailed reasoning (optional)
        if use_llm:
            reasoning = self._generate_llm_reasoning(
                source_system, source_table, source_column, data_type, description,
                best_target, confidence
            )

        # Step 3: Validate mapping consistency
        validation = self.acord_searcher.validate_mapping_consistency(
            source_column, best_target, description
        )

        # Adjust confidence based on validation
        adjusted_confidence = confidence * 0.7 + validation["consistency_score"] * 0.3

        # Create mapping object
        mapping = ColumnMapping(
            source_system=source_system,
            source_table=source_table,
            source_column=source_column,
            source_data_type=data_type,
            source_description=description,
            target_acord_entity=target_entity,
            target_acord_field=target_field,
            confidence_score=min(1.0, adjusted_confidence),
            reasoning=reasoning if reasoning else f"Semantic match with {best_target}",
            alternative_mappings=alternatives,
            status="PENDING" if adjusted_confidence > 0.5 else "REQUIRES_REVIEW"
        )

        # Store mapping
        mapping_key = f"{source_table}.{source_column}"
        self.mappings[mapping_key] = mapping
        self.mapping_history.append(mapping)

        return mapping

    def _generate_llm_reasoning(self, source_system: str, source_table: str,
                                source_column: str, data_type: str, description: str,
                                target_acord: str, confidence: float) -> str:
        """
        Use the LLM to generate detailed reasoning for a mapping choice.

        Args:
            source_system: Source system name
            source_table: Source table name
            source_column: Source column name
            data_type: Data type
            description: Column description
            target_acord: Proposed ACORD target
            confidence: Initial confidence score

        Returns:
            LLM-generated reasoning string
        """
        try:
            # Retrieve context from RAG
            context_results = self.retriever.query(
                f"{source_column} {target_acord}", k=5
            )
            context, _ = self.retriever.format_results(context_results)

            # Generate prompt
            if self.enable_chain_of_thought and len(description) > 50:
                prompt_text = get_chain_of_thought_prompt(
                    source_system, source_table, source_column, data_type, description, context
                )
            else:
                prompt_text = get_semantic_mapping_prompt(
                    source_system, source_table, source_column, data_type, description, context
                )

            # Invoke LLM
            reasoning = self.llm_model.invoke(prompt_text)
            return reasoning[:500]  # Limit to 500 chars for storage

        except Exception as e:
            return f"LLM reasoning generation failed: {str(e)}"

    def batch_generate_mappings(self, table_metadata: Dict[str, Any]) -> List[ColumnMapping]:
        """
        Generate mappings for all columns in a table definition.

        Args:
            table_metadata: Dictionary with table information:
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
            List of ColumnMapping objects for all columns
        """
        results = []
        source_system = table_metadata.get("source_system", "unknown")
        table_name = table_metadata.get("table_name")
        columns = table_metadata.get("columns", [])

        for column in columns:
            mapping = self.generate_mapping(
                source_system=source_system,
                source_table=table_name,
                source_column=column.get("column_name"),
                data_type=column.get("data_type", "unknown"),
                description=column.get("description", "")
            )
            results.append(mapping)

        return results

    def approve_mapping(self, source_table: str, source_column: str,
                        approved_by: str = "admin", notes: str = "") -> ColumnMapping:
        """
        Mark a mapping as approved.

        Args:
            source_table: Source table name
            source_column: Source column name
            approved_by: User approving the mapping
            notes: Optional approval notes

        Returns:
            Updated ColumnMapping object

        Raises:
            KeyError: If mapping not found
        """
        mapping_key = f"{source_table}.{source_column}"

        if mapping_key not in self.mappings:
            raise KeyError(f"Mapping not found for {mapping_key}")

        mapping = self.mappings[mapping_key]
        mapping.status = "APPROVED"
        mapping.approved_by = approved_by
        mapping.notes = notes

        return mapping

    def reject_mapping(self, source_table: str, source_column: str,
                       reason: str = "") -> ColumnMapping:
        """
        Mark a mapping as rejected.

        Args:
            source_table: Source table name
            source_column: Source column name
            reason: Reason for rejection

        Returns:
            Updated ColumnMapping object
        """
        mapping_key = f"{source_table}.{source_column}"

        if mapping_key not in self.mappings:
            raise KeyError(f"Mapping not found for {mapping_key}")

        mapping = self.mappings[mapping_key]
        mapping.status = "REJECTED"
        mapping.notes = reason

        return mapping

    def get_mapping(self, source_table: str, source_column: str) -> ColumnMapping:
        """
        Retrieve a specific mapping.

        Args:
            source_table: Source table name
            source_column: Source column name

        Returns:
            ColumnMapping object or None
        """
        mapping_key = f"{source_table}.{source_column}"
        return self.mappings.get(mapping_key)

    def get_table_mappings(self, source_table: str) -> List[ColumnMapping]:
        """
        Get all mappings for a specific source table.

        Args:
            source_table: Source table name

        Returns:
            List of ColumnMapping objects for the table
        """
        return [m for m in self.mappings.values() if m.source_table == source_table]

    def get_mapping_summary(self, source_table: str = None) -> Dict[str, Any]:
        """
        Get statistics on mappings for a table or across all tables.

        Args:
            source_table: Optional source table name. If None, returns summary for all tables.

        Returns:
            Dictionary with mapping statistics
        """
        if source_table:
            mappings = self.get_table_mappings(source_table)
        else:
            mappings = list(self.mappings.values())

        if not mappings:
            return {
                "total_mappings": 0,
                "approved": 0,
                "pending": 0,
                "requires_review": 0,
                "rejected": 0,
                "average_confidence": 0.0
            }

        approved = len([m for m in mappings if m.status == "APPROVED"])
        pending = len([m for m in mappings if m.status == "PENDING"])
        requires_review = len([m for m in mappings if m.status == "REQUIRES_REVIEW"])
        rejected = len([m for m in mappings if m.status == "REJECTED"])
        avg_confidence = sum(m.confidence_score for m in mappings) / len(mappings)

        return {
            "total_mappings": len(mappings),
            "approved": approved,
            "pending": pending,
            "requires_review": requires_review,
            "rejected": rejected,
            "average_confidence": round(avg_confidence, 3),
            "completion_percentage": round((approved / len(mappings) * 100) if mappings else 0, 1)
        }

    def export_mappings(self, source_table: str = None, status: str = None) -> List[Dict]:
        """
        Export mappings as a list of dictionaries for JSON serialization.

        Args:
            source_table: Optional filter by source table
            status: Optional filter by status (APPROVED, PENDING, etc.)

        Returns:
            List of mapping dictionaries
        """
        mappings = list(self.mappings.values())

        if source_table:
            mappings = [m for m in mappings if m.source_table == source_table]

        if status:
            mappings = [m for m in mappings if m.status == status]

        return [m.to_dict() for m in mappings]
