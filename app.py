# app.py
from flask import Flask, request, render_template, jsonify
from retrieval.retrieval_system import RetrievalSystem
from generation.generate_response import ResponseGenerator
import os
import threading
from dotenv import load_dotenv
import logging
import utils.pdf_to_text
import utils.split_text
# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to track initialization
is_initialized = False
initialization_error = None

# Placeholder for RetrievalSystem and ResponseGenerator instances
retrieval = None
generator = None


def initialize_system():
    global is_initialized, initialization_error, retrieval, generator
    try:
        print("PDF to Text Conversion...")
        utils.pdf_to_text.convert_pdfs_to_text("source_pdfs", "source_txts")

        print("Text Splitting...")
        utils.split_text.process_texts("source_txts", "source_chunks")

        # Initialize Retrieval System with OpenAI API Key
        cohere_api_key = os.getenv("OPENAI_API_KEY")
        if not cohere_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        retrieval = RetrievalSystem(
            chunk_dir="source_chunks",
            index_path="faiss.index",
            mapping_path="id_mapping.pkl",
            cache_file="embedding_cache.json",
        )

        # Initialize Response Generator with OpenAI API Key
        generator = ResponseGenerator(
            openai_api_key=cohere_api_key, model="gpt-3.5-turbo", max_tokens=2048
        )  # Adjust max_tokens as needed

        # Set initialization flag to True
        is_initialized = True
        logger.info("System initialized successfully.")
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Initialization error: {initialization_error}")


@app.route('/', methods=['GET', 'POST'])
def index():
    if not is_initialized:
        if initialization_error:
            return render_template("error.html", error_message=initialization_error)
        return render_template("loading.html")

    answer = ""
    if request.method == 'POST':
        question = request.form['question']
        language = detect_language(question)

        # Retrieve relevant texts with token limit consideration
        retrieved_texts = retrieve_with_token_limit(
            question, max_total_tokens=800, average_chunk_tokens=200
        )

        # Generate answer
        answer = generator.generate(retrieved_texts, question, language)
    return render_template('index.html', answer=answer)

def detect_language(text):
    from langdetect import detect
    try:
        lang = detect(text)
        return "fi" if lang == "fi" else "en"
    except:
        return "en"  # Default to English if detection fails


def retrieve_with_token_limit(query, max_total_tokens=800, average_chunk_tokens=200):
    """
    Retrieves the maximum number of chunks without exceeding the token limit.

    Args:
        query (str): User's question.
        max_total_tokens (int): Maximum tokens allowed for retrieved texts.
        average_chunk_tokens (int): Average tokens per chunk (estimate).

    Returns:
        List[str]: List of retrieved text chunks.
    """
    # Estimate how many chunks can fit
    max_chunks = max_total_tokens // average_chunk_tokens

    # Retrieve more chunks than needed to allow for variable lengths
    retrieved_chunks = retrieval.retrieve(query, top_k=max_chunks * 2)

    selected_chunks = []
    current_token_count = 0

    for chunk in retrieved_chunks:
        chunk_tokens = generator.count_tokens(chunk)
        if current_token_count + chunk_tokens > max_total_tokens:
            break
        selected_chunks.append(chunk)
        current_token_count += chunk_tokens

    return selected_chunks


@app.route("/status")
def status():
    if is_initialized:
        return jsonify({"status": "ready"})
    elif initialization_error:
        return jsonify({"status": "error"})
    else:
        return jsonify({"status": "initializing"})


@app.route("/error")
def error_page():
    return render_template(
        "error.html",
        error_message=initialization_error or "Unknown error during initialization.",
    )


if __name__ == '__main__':
    # Start the initialization in a separate thread
    init_thread = threading.Thread(target=initialize_system)
    init_thread.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=8000)
