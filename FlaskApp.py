# amit.dixit@inbravo
# Flask web application to interact with LLM and RAG retriever
# This app provides an interface to submit queries, view responses, and manage settings.
from flask import Flask, jsonify, redirect, render_template, request, url_for, session
from flask_session import Session
from AppConfig import AppConfig

# Initial components
AppConfig.initialize_components()

# Setup logging
logger = AppConfig.get_default_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure Flask-Session with Redis
app.config['SECRET_KEY'] = AppConfig.FLASK_SECRET_KEY
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'flask_session:'
app.config['SESSION_REDIS'] = AppConfig.get_redis_client()

# Initialize Flask-Session
Session(app)

# Define routes
@app.route("/")
def index():
    # Initialize session if needed
    if 'session_id' not in session:
        import uuid
        session['session_id'] = str(uuid.uuid4())
        logger.info(f"Created new session: {session['session_id']}")
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
    session_id = session.get('session_id')
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        logger.info(f"Created new session in query: {session_id}")

    # Extract query text from request
    query_text = request.json["query_text"]
    logger.info("Received user query: %s", query_text)

    # Save user message to conversation history
    AppConfig.save_conversation_message(session_id, "user", query_text)

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

    # Save assistant response to conversation history
    AppConfig.save_conversation_message(session_id, "assistant", llm_response)

    sources_html = "<br>".join(sources)
    response_text = f"{llm_response}<br><br>Sources:<br>{sources_html}<br><br>Response given by: {AppConfig.LLM_MODEL_NAME}"
    return jsonify(response=response_text)

# Optional route to get conversation history
@app.route("/conversation_history", methods=["GET"])
def conversation_history():
    """Get conversation history for the current session."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify(history=[])
    
    history = AppConfig.get_conversation_history(session_id)
    formatted_history = [{"role": role, "message": message} for role, message in history]
    return jsonify(history=formatted_history)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
