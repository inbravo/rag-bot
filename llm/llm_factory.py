from llm.llm import LLM, AnthropicModel, GPTModel, OllamaModel

# amit.dixit@inbravo
# Factory class to create LLM instances based on model type
class LLMFactory:
    """
    LLMFactory is a factory class responsible for creating instances of different
    language models (LLMs) based on the specified model type.

    Methods:
        create_llm(model_type: str, model_name: str, api_key: str = None) -> LLM:
            Creates and returns an instance of a language model based on the
            provided model type. Supported model types include:
            - 'ollama': Returns an instance of OllamaModel.
            - 'gpt': Returns an instance of GPTModel, requiring an API key.
            - 'claude': Returns an instance of AnthropicModel, requiring an API key.
            Raises a ValueError if the model type is unsupported.

    Args:
        model_type (str): The type of the model to create (e.g., 'ollama', 'gpt', 'claude').
        model_name (str): The name of the model to create.
        api_key (str, optional): The API key required for certain model types (e.g., 'gpt', 'claude').

    Returns:
        LLM: An instance of the specified language model.
    """

    @staticmethod
    def create_llm(model_type: str, model_name: str, api_key: str = None) -> LLM:
        if model_type == "ollama":
            return OllamaModel(model_name)
        elif model_type == "gpt":
            return GPTModel(model_name, api_key)
        elif model_type == "claude":
            return AnthropicModel(model_name, api_key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")