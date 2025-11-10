# Flask web application to interact with LLM and RAG retriever
# This app provides an interface to submit queries, view responses, and manage settings.
from AppConfig import AppConfig
from flask import Flask, jsonify, redirect, render_template, request, url_for

# Initial components
AppConfig.initialize_components()

# Initialize Flask app
app = Flask(__name__)

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
    query_text = request.json["query_text"]

    results = AppConfig.rag_retriever.query(query_text, k=AppConfig.NUM_RELEVANT_DOCS)
    enhanced_context_text, sources = AppConfig.rag_retriever.format_results(results)

    # Generate response from LLM
    llm_response = AppConfig.llm_model.generate_response(
        context=enhanced_context_text, question=query_text
    )

    sources_html = "<br>".join(sources)
    response_text = f"{llm_response}<br><br>Sources:<br>{sources_html}<br><br>Response given by: {AppConfig.LLM_MODEL_NAME}"
    return jsonify(response=response_text)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
