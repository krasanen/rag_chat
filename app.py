# app.py
from flask import Flask, request, render_template
from retrieval.retrieval_system import RetrievalSystem
from generation.generate_response import ResponseGenerator
import os

app = Flask(__name__)

# Initialize Retrieval System
print("Initializing Retrieval System...")
retrieval = RetrievalSystem(chunk_dir='source_chunks')

# Initialize Response Generator with Cohere API Key
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable not set.")
print("Initializing Response Generator...")
generator = ResponseGenerator(cohere_api_key=COHERE_API_KEY)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    if request.method == 'POST':
        question = request.form['question']
        language = detect_language(question)
        retrieved_texts = retrieval.retrieve(question)
        answer = generator.generate(retrieved_texts, question, language)
    return render_template('index.html', answer=answer)

def detect_language(text):
    from langdetect import detect
    lang = detect(text)
    return 'fi' if lang == 'fi' else 'en'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
