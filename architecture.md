# Architecture Overview

This document provides a detailed overview of the **Retrieval-Augmented Generation (RAG) Chatbot** designed to interact with users regarding Finnish collective agreements. The architecture leverages a combination of retrieval-based and generation-based techniques to deliver accurate and contextually relevant responses in both Finnish and English.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Breakdown](#component-breakdown)
    - [1. User Interface (UI)](#1-user-interface-ui)
    - [2. Backend Server](#2-backend-server)
    - [3. Retrieval System](#3-retrieval-system)
    - [4. Language Model Integration](#4-language-model-integration)
    - [5. Data Processing Utilities](#5-data-processing-utilities)
    - [6. Deployment with Docker](#6-deployment-with-docker)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Security Considerations](#security-considerations)
7. [Scalability and Performance](#scalability-and-performance)
8. [Conclusion](#conclusion)

---

## System Overview

The RAG Chatbot system is designed to facilitate seamless interactions between users and a knowledge base comprising Finnish collective agreements. By integrating retrieval mechanisms with advanced language generation models, the chatbot can provide precise answers grounded in the source documents. The system supports bilingual communication, handling queries in both Finnish and English.

---

## Architecture Diagram

+------------------------------------------------------------+
|                        Docker Container                    |
|                                                            |
|  +----------------+         +--------------------------+   |
|  | User Interface | <-----> |     Backend Server       |   |
|  |     (UI)       |         |     (Flask)              |   |
|  +----------------+         +------------+-------------+   |
|                                       |                    |
|                                       |                    |
|                                       |                    |
|                      +----------------v-------------+      |
|                      |      Retrieval System        |      |
|                      |   (sentence-transformers,    |      |
|                      |           FAISS)             |      |
|                      +----------------+-------------+      |
|                                       |                    |
|                                       |                    |
|                      +----------------v-------------+      |
|                      | Language Model Integration   |      |
|                      |      (Cohere API)            |      |
|                      +----------------+-------------+      |
|                                                            |
+------------------------------------------------------------+
          |                                   |
          |                                   |
          |                                   |
+---------v---------+             +-----------v-----------+
|  Source Documents |             |      Cohere API       |
+-------------------+             +-----------------------+

---

## Component Breakdown

### 1. User Interface (UI)

- **Technology:** Flask (Python Web Framework)
- **Functionality:**
  - Provides a web-based frontend for user interactions.
  - Allows users to input queries in Finnish or English.
  - Displays the chatbot’s responses in a user-friendly format.
  - Handles user sessions and manages the state of interactions.

**Key Features:**

- Responsive design for accessibility across devices.
- Input validation to ensure queries are in supported languages.
- Display area for chatbot responses with clear formatting.

### 2. Backend Server

- **Technology:** Flask (Python)
- **Functionality:**
  - Acts as the intermediary between the UI and other system components.
  - Processes incoming user queries.
  - Manages interactions with the retrieval system and language model.
  - Handles language detection to route queries appropriately.
  - Ensures smooth data flow and error handling.

**Key Features:**

- RESTful API endpoints for scalable interactions.
- Middleware for logging and monitoring requests.
- Environment variable management for sensitive data like API keys.

### 3. Retrieval System

- **Technologies:** `sentence-transformers`, `FAISS`
- **Functionality:**
  - Processes and indexes the source documents (collective agreements).
  - Generates vector embeddings for text chunks using multilingual models.
  - Facilitates efficient similarity searches to retrieve relevant document sections based on user queries.

**Key Components:**

- **Embedding Generation:**
  - Utilizes `paraphrase-multilingual-MiniLM-L12-v2` from `sentence-transformers` for creating vector representations of text.
  
- **Indexing with FAISS:**
  - Employs FAISS (Facebook AI Similarity Search) for high-performance similarity searches.
  - Stores embeddings in a FAISS index to enable quick retrieval of the most relevant document chunks.

**Key Features:**

- Supports multi-language embeddings for Finnish and English.
- Scalable to handle large volumes of documents and queries.
- Persistent storage of indexes for quick initialization.

### 4. Language Model Integration

- **Technology:** Cohere's Language Models via API
- **Functionality:**
  - Generates natural language responses based on retrieved document sections and user queries.
  - Ensures that responses are contextually accurate and linguistically coherent.
  - Supports both Finnish and English languages for response generation.

**Key Components:**

- **Prompt Engineering:**
  - Constructs prompts that include relevant document excerpts and user queries to guide the language model.
  
- **API Interaction:**
  - Interfaces with Cohere's API to send prompts and receive generated responses.
  - Manages API rate limits and handles potential errors gracefully.

**Key Features:**

- Customizable response generation parameters (e.g., temperature, max tokens).
- Language-specific response handling to maintain accuracy in both Finnish and English.
- Secure management of API keys and credentials.

### 5. Data Processing Utilities

- **Technologies:** `pdfminer.six`, Python Scripts
- **Functionality:**
  - Converts source PDFs of collective agreements into plain text.
  - Splits large text documents into manageable chunks for efficient processing and retrieval.
  
**Key Components:**

- **PDF to Text Conversion:**
  - Extracts textual content from PDF files while preserving structural integrity.
  
- **Text Chunking:**
  - Divides lengthy documents into smaller, coherent chunks (e.g., 500 tokens) to facilitate precise retrieval.
  - Ensures that chunks do not break the context, maintaining readability and relevance.

**Key Features:**

- Automated preprocessing pipeline for new or updated documents.
- Configurable chunk sizes to balance retrieval accuracy and performance.
- Error handling for corrupted or unreadable PDFs.

### 6. Deployment with Docker

- **Technology:** Docker
- **Functionality:**
  - Containerizes the entire application for consistent deployment across different environments.
  - Ensures that all dependencies are bundled and configured correctly.
  - Simplifies the deployment process, enabling scalability and portability.

**Key Components:**

- **Dockerfile:**
  - Defines the environment, dependencies, and steps to build the application container.
  
- **Docker Compose (Optional):**
  - Manages multi-container deployments if the system scales to include additional services (e.g., databases, caching layers).

**Key Features:**

- Lightweight and reproducible containers for development and production.
- Environment variable management for configuration and secrets.
- Optimized image sizes for faster deployment and reduced resource usage.

---

## Data Flow

1. **User Interaction:**
   - The user accesses the chatbot via the web-based UI and submits a query in Finnish or English.

2. **Backend Processing:**
   - The backend server receives the query and first detects the language using `langdetect`.
   
3. **Retrieval Phase:**
   - The query is passed to the Retrieval System.
   - The system generates an embedding for the query and performs a similarity search using FAISS.
   - The top `k` relevant document chunks are retrieved based on similarity scores.

4. **Generation Phase:**
   - The retrieved document chunks and the original query are formatted into a prompt.
   - The prompt is sent to Cohere's language model via API.
   - The language model generates a coherent and contextually relevant response.

5. **Response Delivery:**
   - The generated response is sent back to the backend server.
   - The backend sends the response to the UI, where it is displayed to the user.

6. **Logging and Monitoring:**
   - All interactions are logged for monitoring, analytics, and potential debugging.

---

## Technology Stack

| Component                | Technology/Library                        | Description                                      |
|--------------------------|-------------------------------------------|--------------------------------------------------|
| **Frontend UI**          | Flask, HTML, CSS, JavaScript              | Web interface for user interactions             |
| **Backend Server**       | Flask                                     | Handles request processing and business logic    |
| **Retrieval System**     | `sentence-transformers`, FAISS             | Embedding generation and similarity search       |
| **Language Model**       | Cohere API                                | Generates natural language responses             |
| **Data Processing**      | `pdfminer.six`, Python Scripts             | Converts and preprocesses source documents       |
| **Containerization**     | Docker                                    | Encapsulates the application for deployment       |
| **Language Detection**   | `langdetect`                              | Identifies the language of user queries           |
| **Dependencies Management** | `pip`, `requirements.txt`               | Manages Python packages and libraries            |

---

## Security Considerations

- **API Key Management:**
  - Cohere API keys are managed via environment variables and not hard-coded into the source code.
  - Utilize Docker secrets or environment variable management tools for enhanced security in production.
  
- **Input Validation:**
  - Sanitize and validate user inputs to prevent injection attacks and ensure data integrity.
  
- **Secure Communication:**
  - Implement HTTPS for secure data transmission between the client and server.
  - Ensure that any external API communications (e.g., with Cohere) are encrypted.
  
- **Access Controls:**
  - Restrict access to the backend server and API endpoints as necessary.
  - Implement authentication and authorization mechanisms if exposing sensitive functionalities.

---

## Scalability and Performance

- **Scalable Architecture:**
  - The modular design allows for individual components to be scaled horizontally based on demand (e.g., multiple backend servers, FAISS clusters).
  
- **Efficient Retrieval:**
  - FAISS provides high-performance similarity searches, enabling quick retrieval even with large datasets.
  
- **Asynchronous Processing:**
  - Future enhancements can include asynchronous request handling to improve response times and handle high concurrency.
  
- **Caching Mechanisms:**
  - Implement caching strategies for frequently asked queries to reduce latency and API usage costs.
  
- **Resource Optimization:**
  - Optimize Docker containers for minimal resource usage, enabling deployment on various infrastructures, including cloud platforms.

---

## Conclusion

The RAG Chatbot architecture is thoughtfully designed to combine robust retrieval mechanisms with advanced language generation capabilities, ensuring accurate and contextually relevant responses to user queries about Finnish collective agreements. By leveraging scalable technologies and adhering to best practices in security and performance optimization, the system is well-equipped to handle diverse user interactions efficiently.

This modular and containerized architecture not only facilitates ease of deployment and maintenance but also allows for future enhancements and scalability as user demands grow.

---

**Note:** Ensure that all dependencies are correctly installed and that you have the necessary permissions and API keys. If you encounter any issues, refer to the documentation of the respective libraries or seek support from their communities.