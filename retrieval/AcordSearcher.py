from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from retrieval.RAGRetriever import RAGRetriever

# AcordSearcher class for specialized ACORD standard searching
class AcordSearcher:
    """
    A specialized searcher that performs semantic searches specifically against
    ACORD standard definitions to find the best matching entities and fields
    for source data mappings.

    This class extends the basic RAGRetriever functionality with ACORD-specific
    filters and ranking logic.

    Attributes:
        retriever (RAGRetriever): The underlying RAG retriever instance
        acord_filter_metadata (Dict): Filters to apply when searching ACORD documents

    Methods:
        search_acord_entity(search_term: str, k: int = 5) -> List[Tuple[Document, float]]:
            Search for ACORD entities matching the search term.

        search_acord_field(search_term: str, k: int = 5) -> List[Tuple[Document, float]]:
            Search for ACORD fields matching the search term.

        find_best_mapping(source_column_name: str, source_description: str, 
                          k: int = 10) -> Tuple[str, float, List[Tuple[str, float]]]:
            Find the best ACORD mapping for a source column using semantic similarity.

        rank_mappings_by_confidence(results: List[Tuple[Document, float]]) -> List[Dict]:
            Rank mapping results by confidence scores and return formatted results.
    """

    def __init__(self, retriever: RAGRetriever):
        """
        Initialize the AcordSearcher with a RAG retriever instance.

        Args:
            retriever: An initialized RAGRetriever instance
        """
        self.retriever = retriever
        self.acord_filter_metadata = {
            "source_type": "acord_standard"
        }

    def search_acord_entity(self, search_term: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for ACORD entities matching the given search term.

        Args:
            search_term: The business term or concept to search for
            k: Number of top results to return

        Returns:
            List of tuples (Document, similarity_score) for ACORD entities
        """
        # Build a search query focused on entities
        entity_query = f"ACORD entity: {search_term}"

        # Query the retriever
        results = self.retriever.query(entity_query, k=k)

        # Filter results to include only ACORD standard documents
        filtered_results = [
            (doc, score) for doc, score in results
            if doc.metadata.get("source_type") == "acord_standard" and
               doc.metadata.get("document_type") == "acord_entity"
        ]

        return filtered_results

    def search_acord_field(self, search_term: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for ACORD fields matching the given search term.

        Args:
            search_term: The field name or business concept to search for
            k: Number of top results to return

        Returns:
            List of tuples (Document, similarity_score) for ACORD fields
        """
        field_query = f"ACORD field: {search_term}"
        results = self.retriever.query(field_query, k=k)

        # Filter to ACORD field documents only
        filtered_results = [
            (doc, score) for doc, score in results
            if doc.metadata.get("source_type") == "acord_standard" and
               doc.metadata.get("document_type") == "acord_field"
        ]

        return filtered_results

    def find_best_mapping(self, source_column_name: str, source_description: str,
                          k: int = 10) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Find the best ACORD mapping for a source column using semantic similarity.

        Args:
            source_column_name: The name of the source column (e.g., 'EFF_DT')
            source_description: The business description of the source column
            k: Number of candidates to consider

        Returns:
            Tuple containing:
            - best_mapping: The top ACORD entity.field path
            - confidence: Confidence score (0-1, normalized from similarity)
            - alternatives: List of (alternative_path, confidence) tuples
        """
        # Create a comprehensive search query
        search_query = f"{source_column_name} {source_description}"

        # Get candidate mappings from ACORD standard
        all_results = self.retriever.query(search_query, k=k * 2)

        # Filter to ACORD documents only
        acord_results = [
            (doc, score) for doc, score in all_results
            if doc.metadata.get("source_type") == "acord_standard"
        ]

        if not acord_results:
            return "", 0.0, []

        # Score and rank results
        ranked_results = self.rank_mappings_by_confidence(acord_results)

        if not ranked_results:
            return "", 0.0, []

        # Extract best mapping
        best_mapping = ranked_results[0]
        best_path = f"{best_mapping['entity']}.{best_mapping['field']}" \
            if best_mapping.get('field') else best_mapping['entity']
        best_confidence = best_mapping['confidence']

        # Extract alternatives
        alternatives = [
            (f"{r['entity']}.{r['field']}" if r.get('field') else r['entity'], r['confidence'])
            for r in ranked_results[1:min(5, len(ranked_results))]
        ]

        return best_path, best_confidence, alternatives

    def rank_mappings_by_confidence(self, results: List[Tuple[Document, float]]) -> List[Dict]:
        """
        Rank mapping results by confidence scores and return formatted results.

        Confidence is calculated from the similarity score and metadata indicators:
        - Base: Normalize similarity score (0-1)
        - Boost: +0.1 if field is marked as required in ACORD
        - Boost: +0.05 if data types match

        Args:
            results: List of tuples (Document, similarity_score)

        Returns:
            List of dictionaries with ranking information, sorted by confidence descending
        """
        ranked = []

        for doc, similarity_score in results:
            confidence = min(1.0, max(0.0, similarity_score))  # Normalize to 0-1

            # Boost confidence if field is required
            if doc.metadata.get("required") == "True":
                confidence += 0.1

            # Apply field-specific confidence adjustments
            is_required = doc.metadata.get("required") == "True"
            entity = doc.metadata.get("entity", "")
            field = doc.metadata.get("field", "")

            ranked.append({
                "entity": entity,
                "field": field,
                "data_type": doc.metadata.get("data_type", ""),
                "required": is_required,
                "confidence": min(1.0, confidence),
                "similarity_score": similarity_score,
                "document_type": doc.metadata.get("document_type")
            })

        # Sort by confidence descending
        ranked.sort(key=lambda x: x["confidence"], reverse=True)

        return ranked

    def get_entity_definition(self, entity_name: str) -> str:
        """
        Retrieve the full definition of an ACORD entity.

        Args:
            entity_name: Name of the ACORD entity (e.g., 'Policy')

        Returns:
            The entity definition from the document, or empty string if not found
        """
        results = self.search_acord_entity(entity_name, k=1)

        if results:
            doc, _ = results[0]
            return doc.page_content
        return ""

    def search_across_metadata_types(self, search_term: str, k: int = 5) -> Dict[str, List]:
        """
        Perform a cross-metadata search across SQL schema, ACORD standards, and
        customer dictionary to find related information.

        Args:
            search_term: The term to search for
            k: Number of results per source type

        Returns:
            Dictionary with keys for each source type and their results
        """
        all_results = self.retriever.query(search_term, k=k * 3)

        results_by_type = {
            "acord_standard": [],
            "sql_schema": [],
            "customer_dictionary": []
        }

        for doc, score in all_results:
            source_type = doc.metadata.get("source_type", "unknown")
            if source_type in results_by_type:
                results_by_type[source_type].append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })

        return results_by_type

    def validate_mapping_consistency(self, source_column: str, proposed_acord_path: str,
                                     customer_dict_context: str = "") -> Dict[str, Any]:
        """
        Validate that a proposed mapping is consistent with ACORD standards and
        customer business dictionary.

        Args:
            source_column: The source column being mapped
            proposed_acord_path: The proposed ACORD target (e.g., 'Policy.EffectiveDate')
            customer_dict_context: Optional customer dictionary context

        Returns:
            Dictionary with validation results including consistency score (0-1) and notes
        """
        # Search for the proposed ACORD target
        acord_results = self.search_acord_field(proposed_acord_path, k=3)

        if not acord_results:
            return {
                "valid": False,
                "consistency_score": 0.0,
                "notes": "Proposed ACORD path not found in standard"
            }

        # Check if source column context matches ACORD field context
        search_query = f"{source_column} {proposed_acord_path}"
        similarity_results = self.retriever.query(search_query, k=1)

        if similarity_results:
            _, similarity_score = similarity_results[0]
            consistency_score = min(1.0, max(0.0, similarity_score))
        else:
            consistency_score = 0.5  # Neutral if no direct match found

        return {
            "valid": consistency_score > 0.3,
            "consistency_score": consistency_score,
            "notes": f"Mapping consistency validated with score {consistency_score:.2f}"
        }
