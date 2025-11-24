from AppConfig import AppConfig

# Initial components
AppConfig.initialize_components()

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""
# amit.dixit@inbravo
# Test function to validate LeapLogic introduction
def test_leaplogic_intro():
    assert query_and_validate(
        question="what is leaplogic?",
        expected_response="LeapLogic is a product used by Impetus for migration services. It can be installed in a non-production environment/Sandbox, allowing team members to work without accessing production environments.",
        retriever=AppConfig.rag_retriever,
        llm_model=AppConfig.llm_model,
    )


# Test function to validate COBIT details
def test_cobit_details():
    assert query_and_validate(
        question="what is COBIT?",
        expected_response="COBIT (Control Objectives for Information and Related Technology) is a set of guidelines, best practices, and standards for ensuring the effective governance and management of information technology. It provides a comprehensive framework for IT audit, control, and security professionals to assess and improve the organization's IT processes and systems.",
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

    # Retrieve relevant documents
    results = retriever.query(question, k=AppConfig.NUM_RELEVANT_DOCS)
    enhanced_context_text, sources = retriever.format_results(results)

    # Generate response from LLM
    response_text = llm_model.generate_response(
        context=enhanced_context_text, question=question
    )

    # Log the question, expected response, actual response, and sources
    AppConfig.get_default_logger(__name__).info(
        "Testing question: %s | expected response: %s | LLM response: %s | Info Sources: %s",
        question, expected_response, response_text, sources
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
        raise ValueError("Invalid evaluation result. Cannot determine if 'true' or 'false'.")

# Main block to run tests
if __name__ == "__main__":
    test_leaplogic_intro()
    test_cobit_details()
