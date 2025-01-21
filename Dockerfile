# Dockerfile
FROM python:3.9-slim-bullseye

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/huggingface
ENV TRANSFORMERS_CACHE=/huggingface
ENV XDG_CACHE_HOME=/huggingface

# Create necessary directories with proper permissions
RUN mkdir -p /app /huggingface \
    && chmod -R 777 /huggingface

# Set work directory
WORKDIR /app

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose Streamlit port
EXPOSE 8502

# Set environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]
