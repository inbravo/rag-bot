A Python based local Retrieval-Augmented Generation (RAG) chatbot that can answer to questions by scanning the documents (pdf, doc, excel etc.).

## Solution components and purpose
| #  |  Python code | Purpose   | Design Principle   | Technology  |
|---|---|---|---|---|
| 1 | [FlaskApp](https://github.com/inbravo/rag-bot/blob/main/FlaskApp.py) |  Server side to manage user request from web browser | [WSGI](https://flask.palletsprojects.com/en/stable/design)  |  [Flask](https://flask.palletsprojects.com/en/stable/) |
| 2 | [AppConfig](https://github.com/inbravo/rag-bot/blob/main/AppConfig.py) | Application module to load configuration from environment variables, initializes the retriever and LLM model and provides methods to update the configuration dynamically. | | [DotEnv](https://github.com/motdotla/dotenv) |
| 3 | [RagRetriever](https://github.com/inbravo/rag-bot/blob/main/retrieval/RAGRetriever.py)  | RAGRetriever class to handle retrieval-augmented generation. It interacts with a vector database to fetch relevant documents based on query similarity. It uses an embedding model to compute text embeddings for similarity comparison. It extracts relevant context and source information from the retrieved documents. |   | [Langchain Chroma Vectorstore](https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html)  |
| 4 | [EmbeddingFactory](https://github.com/inbravo/rag-bot/blob/main/embeddings/EmbeddingFactory.py) | Embedding Factory class to select and return the appropriate embedding function based on the specified model name (Ollama, OpenAI etc) |   |  [Langchain Embeddings](https://api.python.langchain.com/en/latest/community/embeddings.html) |


##  What is Retrieval-Augmented Generation (RAG)?
<div style="text-align: center;">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*J7vyY3EjY46AlduMvr9FbQ.png" alt="rag_pipeline" width="600" height="300">
</div>
Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of information retrieval and natural language generation. In a RAG system, a retriever fetches relevant documents or text chunks from a database, and then a generator produces a response based on the retrieved context.

1. **Data Indexing**
- Documents: This is the starting point where multiple documents are stored.
- Vector DB: The documents are processed and indexed into a Vector Database.

2. **User Query**
- A user query is input into the system, which interacts with the Vector Database.

3. **Data Retrieval & Generation**
- Top-K Chunks: The Vector Database retrieves the top-K relevant chunks based on the user query.
- LLM (Large Language Model): These chunks are then fed into a Large Language Model.
- Response: The LLM generates a response based on the relevant chunks.

## 🏗️ Implementation Components
For this project, i exploited the following components to build the RAG architecture:
1. **Chroma**: A vector database used to store and retrieve document embeddings efficiently.
2. **Flask**: Framework for rendering web page and handling user interactions.
3. **Ollama**: Manages the local language model for generating responses.
4. **LangChain**: A framework for integrating language models and retrieval systems.

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

