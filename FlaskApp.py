# amit.dixit@inbravo
# Flask web application to interact with LLM and RAG retriever
# This app provides an interface to submit queries, view responses, and manage settings.
import json
import os
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from AppConfig import AppConfig

# Initial components
AppConfig.initialize_components()

# Setup logging
logger = AppConfig.get_default_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Try to initialize Redis client using AppConfig helper
redis_client = None
if hasattr(AppConfig, 'redis_client'):
    redis_client = AppConfig.redis_client()
else:
    # Fallback to environment-based config if helper not present
    try:
        import redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
    except (ImportError, Exception) as e:
        logger.warning(f"Redis client initialization failed: {e}")

# Apply session configuration using AppConfig helper or fallback
if hasattr(AppConfig, 'apply_session_config'):
    AppConfig.apply_session_config(app, redis_client)
else:
    # Fallback to basic configuration
    from datetime import timedelta
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    if redis_client is not None:
        try:
            from flask_session import Session
            app.config['SESSION_TYPE'] = 'redis'
            app.config['SESSION_REDIS'] = redis_client
            app.config['SESSION_PERMANENT'] = True
            app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=int(os.getenv("SESSION_LIFETIME_DAYS", "7")))
            Session(app)
        except ImportError:
            logger.warning("Flask-Session not available, using default sessions")

# Helper functions for session and conversation management
def get_session_id():
    """Get or create a session ID for the current user session."""
    if 'sid' not in session:
        import uuid
        session['sid'] = str(uuid.uuid4())
    return session['sid']

def conv_key(sid):
    """Generate Redis key for conversation storage."""
    if hasattr(AppConfig, 'conv_key'):
        return AppConfig.conv_key(sid)
    return f"conv:{sid}"

def save_message_to_redis(sid, role, content):
    """Save a message to Redis conversation history."""
    if redis_client is None:
        return
    try:
        key = conv_key(sid)
        message = json.dumps({"role": role, "content": content})
        redis_client.rpush(key, message)
        ttl = getattr(AppConfig, 'CONVERSATION_TTL_SECONDS', int(os.getenv("CONVERSATION_TTL_SECONDS", "86400")))
        redis_client.expire(key, ttl)
    except Exception as e:
        logger.warning(f"Failed to save message to Redis: {e}")

def load_conversation_from_redis(sid):
    """Load conversation history from Redis."""
    if redis_client is None:
        return []
    try:
        key = conv_key(sid)
        messages = redis_client.lrange(key, 0, -1)
        return [json.loads(msg) for msg in messages]
    except Exception as e:
        logger.warning(f"Failed to load conversation from Redis: {e}")
        return []

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

# Define route to handle user queriesß
@app.route("/query", methods=["POST"])
def query():

    # Get session ID
    sid = get_session_id()

    # Extract query text from request
    query_text = request.json["query_text"]
    logger.info("Received user query: %s", query_text)

    # Save user message to Redis
    save_message_to_redis(sid, "user", query_text)

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
    
    # Save assistant response to Redis
    save_message_to_redis(sid, "assistant", llm_response)
    
    return jsonify(response=response_text)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
