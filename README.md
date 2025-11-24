A Python based local Retrieval-Augmented Generation (RAG) chatbot that can answer to questions by scanning the documents (pdf, doc, excel etc.).

## Solution components
| #  |  Python code | Purpose   | Design Principle   | Technology  |
|---|---|---|---|---|
| 1 | [FlaskApp][Link_1.md] |  Server side to manage user request from web browser | [WSGI][Link_2.md] |  [Flask][Link_3.md] |
| 2 | [AppConfig][Link_4.md] | Application module to load configuration from environment variables, initializes the retriever and LLM model and provides methods to update the configuration dynamically. | [12 Factor Config][Link_5.md] | [DotEnv][Link_6.md] |
| 3 | [RagRetriever][Link_7.md] | RAGRetriever class to handle retrieval-augmented generation. It interacts with a vector database to fetch relevant documents based on query similarity. It uses an embedding model to compute text embeddings for similarity comparison. It extracts relevant context and source information from the retrieved documents. | [2 Step RAG][Link_8.md]  | [Langchain Chroma Vectorstore][Link_9.md]  |
| 4 | [EmbeddingFactory][Link_10.md] | Embedding Factory class to select and return the appropriate embedding function based on the specified model name (Ollama, OpenAI etc) | [Vectorization & Similarity Scoring][Link_11.md] |  [Langchain Embeddings][Link_12.md]|
| 5 | [DocUploader][Link_13.md]| Manage the database population with embeddings | [Vector Database][Link_14.md] | [Langchain Chroma][Link_15.md] | 

##  System control flow
<div style="text-align: center;">
<img src="https://github.com/inbravo/rag-bot/tree/main/images/call-flow.png" alt="call-flow" width="600" height="300">
</div>

## 🛠️ Setup and Local Deployment

1. **Choose Your Setup**:
   - You have three different options for setting up the LLMs:
     1. Local setup using Ollama.
     2. Using the OpenAI API for GPT models.
     3. Using the Anthropic API for Claude models.

### Option 1: Local Setup with Ollama

- **Download and install Ollama on your PC**:
   - Visit [Ollama's official website](https://ollama.com/download) to download and install Ollama. Ensure you have sufficient hardware resources to run the local language model.
   - Pull a LMM of your choice:
   ```sh
   ollama pull <model_name>  # e.g. ollama pull llama3:8b

### Option 2: Use OpenAI API for GPT Models
- **Set up OpenAI API**: you can sign up and get your API key from [OpenAI's website](https://openai.com/api/).

### Option 3: Use Anthropic API for Claude Models
- **Set up Anthropic API**: you can sign up and get your API key from [Anthropic's website](https://www.anthropic.com/api).

## Common Steps

2. **Clone the repository and navigate to the project directory**:
    ```sh
    git clone https://github.com/enricollen/rag-conversational-agent.git
    cd rag-conversational-agent
    ```

3. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. **Install the required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Insert you own PDFs in /data folder**

6. **Run once the populate_database script to index the pdf files into the vector db:**
    ```sh
    python populate_database.py
    ```

7. **Run the application:**
    ```sh
    python app.py
    ```

8. **Navigate to `http://localhost:5000/`**

9. **If needed, click on ⚙️ icon to access the admin panel and adjust app parameters**

10. **Perform a query** 

## 🚀 Future Improvements
- [ ] Add Langchain Tools compatibility, allowing users to define custom Python functions that can be utilized by the LLMs.
- [ ] Add automated test cases


[Link_1.md]: https://github.com/inbravo/rag-bot/blob/main/FlaskApp.py
[Link_2.md]: https://flask.palletsprojects.com/en/stable/design
[Link_3.md]: https://flask.palletsprojects.com/en/stable/design](https://flask.palletsprojects.com/en/stable
[Link_4.md]: https://github.com/inbravo/rag-bot/blob/main/AppConfig.py
[Link_5.md]: https://12factor.net/config
[Link_6.md]: https://github.com/motdotla/dotenv
[Link_7.md]: https://github.com/inbravo/rag-bot/blob/main/retrieval/RAGRetriever.py
[Link_8.md]: https://docs.langchain.com/oss/python/langchain/retrieval
[Link_9.md]: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html
[Link_10.md]: https://github.com/inbravo/rag-bot/blob/main/embeddings/EmbeddingFactory.py
[Link_11.md]: https://docs.langchain.com/oss/python/integrations/text_embedding
[Link_12.md]: https://api.python.langchain.com/en/latest/community/embeddings.html
[Link_13.md]: https://github.com/inbravo/rag-bot/blob/main/DocUploader.py
[Link_14.md]: https://docs.trychroma.com/docs/overview/getting-started
[Link_15.md]: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
