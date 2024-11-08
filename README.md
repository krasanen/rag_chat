# Architecture
see architecture.md

# Building and running the application on docker container, replace your_cohere_api_key with yours
docker build -t rag_chatbot .
docker run -d -p 8000:8000 --name=rag_chatbox --env OPENAI_API_KEY=xxx rag_chatbot
docker logs -f rag_chatbox

# for testing it browse to
http://localhost:8000

# for developement
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# improvements ideas that came along development
- thread starting so we could show starting page while retrievalsystem index is being build
- cache indexes for faster start
- health check checking status endpoint frequently and restarting container on error