# amit.dixit@inbravo
# Flask web application to interact with LLM and RAG retriever
# This app provides an interface to submit queries, view responses, and manage settings.
# Enhanced with Redis-backed server-side sessions for per-user conversation history.

import os
import uuid
import json
from datetime import timedelta
from flask import Flask, request, session, jsonify, redirect, render_template, url_for
from flask_session import Session
import redis
from AppConfig import AppConfig

# Initial components
AppConfig.initialize_components()

# Setup logging
logger = AppConfig.get_default_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# App secret key from environment (important for session security)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-prod")

# Redis config for session storage and conversation history
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Redis client used for both Flask-Session and direct conversation storage
redis_client = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    db=REDIS_DB, 
    password=REDIS_PASSWORD, 
    decode_responses=True
)

# Flask-Session config (server-side sessions)
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = redis_client
app.config["SESSION_PERMANENT"] = False
# Optional: session lifetime for cookie
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

Session(app)

# Conversation key TTL in seconds (default 7 days)
CONVERSATION_TTL = int(os.environ.get("CONVERSATION_TTL_SECONDS", 7 * 24 * 3600))


def get_session_id():
    """Return or create a stable per-client session id stored in Flask server-side session."""
    sid = session.get("user_sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["user_sid"] = sid
    return sid


def conv_key(sid: str) -> str:
    """Generate Redis key for storing conversation history."""
    return f"conv:{sid}"


def save_message_to_redis(role: str, text: str):
    """Save a message (user or assistant) to the conversation history in Redis."""
    sid = get_session_id()
    key = conv_key(sid)
    payload = json.dumps({"role": role, "text": text})
    # rpush is atomic; store as string
    redis_client.rpush(key, payload)
    redis_client.expire(key, CONVERSATION_TTL)


def load_conversation_from_redis():
    """Load the full conversation history for the current session from Redis."""
    sid = get_session_id()
    key = conv_key(sid)
    raw = redis_client.lrange(key, 0, -1)
    messages = [json.loads(item) for item in raw]
    return messages


# Define routes
@app.route("/")
def index():
    return render_template("index.html")


# Define admin route to view and update settings
@app.route("/admin")
def admin():
    return render_template(
        "admin.html",
        llm_model_name=AppConfig.LLM_MODEL_NAME,
        llm_model_type=AppConfig.LLM_MODEL_TYPE,
        embedding_model_name=AppConfig.EMBEDDING_MODEL_NAME,
        num_relevant_docs=AppConfig.NUM_RELEVANT_DOCS,
        openai_api_key=AppConfig.OPENAI_API_KEY,
    )


# Define route to handle settings update
@app.route("/update_settings", methods=["POST"])
def update_settings():
    logger.info("Updating application settings...")
    # Reinitialize the components (llm and retriever objects)
    AppConfig.update_components(
        request.form["llm_model_name"],
        request.form["llm_model_type"],
        request.form["embedding_model_name"],
        int(request.form["num_relevant_docs"]),
        request.form["openai_api_key"],
    )

    # Log the updated LLM details
    print(
        f"Updating model type: {AppConfig.LLM_MODEL_TYPE} | model name: {AppConfig.LLM_MODEL_NAME} | embedding model: {AppConfig.EMBEDDING_MODEL_NAME}"
    )
    return redirect(url_for("admin"))


# Define route to handle user queries (legacy endpoint - preserved for compatibility)
@app.route("/query", methods=["POST"])
def query():
    # Extract query text from request
    query_text = request.json["query_text"]
    logger.info("Received user query: %s", query_text)

    # Retrieve relevant documents
    results = AppConfig.rag_retriever.query(query_text, k=AppConfig.NUM_RELEVANT_DOCS)
    logger.info("Received RAG Response: %s", results)

    # Format retrieved results
    enhanced_context_text, sources = AppConfig.rag_retriever.format_results(results)
    logger.info("Received Formatted RAG Response: %s", results)

    # Generate response from LLM
    llm_response = AppConfig.llm_model.generate_response(
        context=enhanced_context_text, question=query_text
    )

    logger.info("Received LLM Response: %s", llm_response)

    sources_html = "<br>".join(sources)
    response_text = f"{llm_response}<br><br>Sources:<br>{sources_html}<br><br>Response given by: {AppConfig.LLM_MODEL_NAME}"
    return jsonify(response=response_text)


# New endpoint: /chat - handles user messages with session-based conversation history
@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat messages with per-user conversation history stored in Redis.
    Integrates with existing RAG/LangChain pipeline using session context.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'text required'}), 400

    user_text = data['text']
    logger.info("Chat request - session: %s, text: %s", get_session_id(), user_text)
    
    # Save user message to conversation history
    save_message_to_redis('user', user_text)

    # Load conversation history for context
    conversation_history = load_conversation_from_redis()
    logger.info("Loaded conversation history: %d messages", len(conversation_history))

    # TODO: Integration point for RAG/LangChain with session context
    # Option 1: Use session_id as namespace for per-user vectorstore
    #   session_id = get_session_id()
    #   user_vectorstore = get_or_create_user_vectorstore(session_id)
    #   results = user_vectorstore.query(user_text, k=AppConfig.NUM_RELEVANT_DOCS)
    #
    # Option 2: Pass conversation history to LangChain Memory
    #   from langchain.memory import ConversationBufferMemory
    #   memory = ConversationBufferMemory()
    #   for msg in conversation_history[:-1]:  # Exclude current message
    #       if msg['role'] == 'user':
    #           memory.chat_memory.add_user_message(msg['text'])
    #       elif msg['role'] == 'assistant':
    #           memory.chat_memory.add_ai_message(msg['text'])
    #
    # Option 3: Include conversation context in prompt
    #   context_with_history = format_history_for_prompt(conversation_history)
    #
    # Example using existing RAG pipeline with conversation context:
    try:
        # Retrieve relevant documents using existing RAG retriever
        results = AppConfig.rag_retriever.query(user_text, k=AppConfig.NUM_RELEVANT_DOCS)
        
        # Format retrieved results
        enhanced_context_text, sources = AppConfig.rag_retriever.format_results(results)
        
        # Generate response from LLM with context
        # Note: For full conversation support, you may want to pass conversation_history
        # to the LLM to maintain context across multiple turns
        assistant_reply = AppConfig.llm_model.generate_response(
            context=enhanced_context_text, 
            question=user_text
        )
        
        logger.info("Generated assistant reply: %s", assistant_reply)
    except Exception as e:
        logger.error("Error in RAG pipeline: %s", str(e))
        assistant_reply = f"[Error processing request: {str(e)}]"

    # Save assistant reply to conversation history
    save_message_to_redis('assistant', assistant_reply)

    return jsonify({'reply': assistant_reply})


# New endpoint: /history - retrieve conversation history for current session
@app.route('/history', methods=['GET'])
def history():
    """Return the conversation history for the current user session."""
    return jsonify({
        'session_id': get_session_id(), 
        'history': load_conversation_from_redis()
    })


# Run the Flask app
if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )
