# amit.dixit@inbravo
# Flask web application to interact with LLM and RAG retriever
# This app provides an interface to submit queries, view responses, and manage settings.
# Uses Redis-backed server-side sessions for conversation history management.

import os
import uuid
import json
import redis
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_session import Session
from AppConfig import AppConfig

# Initial components
AppConfig.initialize_components()

# Setup logging
logger = AppConfig.get_default_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure Flask-Session with Redis backend
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'session:'

# Initialize Redis client for session storage and conversation history
redis_client = redis.Redis(
    host=os.environ.get('REDIS_HOST', 'localhost'),
    port=int(os.environ.get('REDIS_PORT', 6379)),
    db=int(os.environ.get('REDIS_DB', 0)),
    password=os.environ.get('REDIS_PASSWORD', None),
    decode_responses=True
)

# Set Redis client for Flask-Session
app.config['SESSION_REDIS'] = redis_client

# Initialize Flask-Session
Session(app)

# Configuration for conversation TTL (time to live)
CONVERSATION_TTL_SECONDS = int(os.environ.get('CONVERSATION_TTL_SECONDS', 3600))

# Helper function to get or create a stable session ID
def get_session_id():
    """
    Get or create a stable session ID for the current user session.
    Returns the session ID stored under session['user_sid'].
    """
    if 'user_sid' not in session:
        session['user_sid'] = str(uuid.uuid4())
        logger.info("Created new session ID: %s", session['user_sid'])
    return session['user_sid']

# Helper function to generate conversation key for Redis
def conv_key(sid):
    """
    Generate the Redis key for storing conversation history.
    Args:
        sid: Session ID
    Returns:
        Redis key string in format 'conv:<session_id>'
    """
    return f"conv:{sid}"

# Helper function to save message to Redis
def save_message_to_redis(role, text):
    """
    Save a message to the conversation history in Redis.
    Args:
        role: Message role ('user' or 'bot')
        text: Message text content
    """
    sid = get_session_id()
    key = conv_key(sid)
    
    # Store message as JSON string in Redis list
    message = json.dumps({"role": role, "text": text})
    redis_client.rpush(key, message)
    
    # Set TTL on the conversation key
    redis_client.expire(key, CONVERSATION_TTL_SECONDS)
    
    logger.info("Saved %s message to Redis for session %s", role, sid)

# Helper function to load conversation from Redis
def load_conversation_from_redis():
    """
    Load the conversation history from Redis for the current session.
    Returns:
        List of message dictionaries with 'role' and 'text' keys
    """
    sid = get_session_id()
    key = conv_key(sid)
    
    # Retrieve all messages from Redis list
    messages = redis_client.lrange(key, 0, -1)
    
    # Parse JSON messages
    conversation = []
    for msg in messages:
        try:
            conversation.append(json.loads(msg))
        except json.JSONDecodeError:
            logger.error("Failed to parse message from Redis: %s", msg)
    
    logger.info("Loaded %d messages from Redis for session %s", len(conversation), sid)
    return conversation

# Define routes
@app.route("/")
def index():
    # Initialize session ID on first visit
    get_session_id()
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

# Define route to handle user queries
@app.route("/query", methods=["POST"])
def query():

    # Extract query text from request
    query_text = request.json["query_text"]
    logger.info("Received user query: %s", query_text)
    
    # Save user message to conversation history
    save_message_to_redis("user", query_text)

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
    
    # Save bot response to conversation history
    save_message_to_redis("bot", response_text)
    
    return jsonify(response=response_text)

# Optional route to get conversation history
@app.route("/conversation", methods=["GET"])
def get_conversation():
    """
    Endpoint to retrieve conversation history for the current session.
    """
    conversation = load_conversation_from_redis()
    return jsonify(conversation=conversation)

# Optional route to clear conversation history
@app.route("/clear_conversation", methods=["POST"])
def clear_conversation():
    """
    Endpoint to clear conversation history for the current session.
    """
    sid = get_session_id()
    key = conv_key(sid)
    redis_client.delete(key)
    logger.info("Cleared conversation history for session %s", sid)
    return jsonify(success=True)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
