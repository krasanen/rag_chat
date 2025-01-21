# RAG Chatbot

## Features
- Conversational AI with context retention
- Document retrieval and question answering
- Streamlit-based interactive UI
- Docker and Docker Compose support

## Prerequisites
- Docker
- Docker Compose
- OpenAI API Key

## Setup

### 1. Environment Configuration
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Running with Docker Compose
```bash
docker-compose up --build
```

### 3. Accessing the Application
Open your browser and navigate to:
http://localhost:8501

### 4. Local Development
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Improvements
- Conversation history persistence
- Enhanced context retrieval
- Multi-language support
- Advanced document indexing