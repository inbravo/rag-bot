from AppConfig import AppConfig

# Initialize application components
AppConfig.initialize_components()

# Evaluation prompt template for response validation
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


# amit.dixit@inbravo
# Test function to validate the sales narrative generation
def test_sales_narrative(dealName: str, painPoints: str):
    """ Test function to validate sales narrative generation"""

    question = f"Create a winning sales storyline for '{dealName}'. Their main pain points are: {painPoints}. How should I structure my pitch to address these challenges and position our solutions?"

    assert query_and_validate(
        question=question,
        expected_response="<TO_BE_FILLED>",
        retriever=AppConfig.rag_retriever,
        llm_model=AppConfig.llm_model,
    )

# amit.dixit@inbravo
# Test function to validate the ROI business case generation
def test_roi_businesscase(amount: str, industry: str, decisionMakers: str):
    """ Test function to validate ROI business case generation"""

    question = f"Build a compelling ROI story for this {amount} {industry} opportunity. What are the financial metrics and value propositions we should emphasize for {decisionMakers}?"

    assert query_and_validate(
        question=question,
        expected_response="<TO_BE_FILLED>",
        retriever=AppConfig.rag_retriever,
        llm_model=AppConfig.llm_model,
    )

# Helper function to query the LLM and validate the response
# To be used by the test functions
def query_and_validate(question: str, expected_response: str, retriever, llm_model):
    """
    Queries the language model (LLM) to get a response for the given question, and then validates this response
    against the expected response using the LLM itself.

    Parameters:
    question (str): The question to be asked to the LLM.
    expected_response (str): The expected response to validate against.
    retriever: An instance of the RAGRetriever used to retrieve relevant documents.
    llm_model: An instance of the LLM to generate responses.

    Returns:
    bool: True if the LLM validates that the actual response matches the expected response, False otherwise.
    """

    # Retrieve relevant results from the Vector DB (Chroma)
    results = retriever.query(question, k=AppConfig.NUM_RELEVANT_DOCS)

    # Format the results to get enhanced context
    enhanced_context_text, sources = retriever.format_results(results)

    # Pass on the enhanced context and user query to LLM to get the best answer
    response_text = llm_model.generate_response(
        context=enhanced_context_text, question=question
    )

    # Log the question, expected response, actual response, and sources
    AppConfig.get_default_logger(__name__).info(
        "Testing question: %s | expected response: %s | LLM response: %s | Info Sources: %s",
        question,
        expected_response,
        response_text,
        sources,
    )

    # Use the same LLM also for response validation
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    # Get evaluation result from LLM and clean it
    evaluation_results_str = llm_model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    # Determine if the response is correct or not based on LLM evaluation
    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            "Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )


# Main block to run tests
if __name__ == "__main__":

    # Run test functions
    test_sales_narrative("Example Deal", "high costs and inefficiency")
    test_roi_businesscase("$100K", "manufacturing", "CFO and CTO")