import os
import streamlit as st
from dotenv import load_dotenv
import utils.pdf_to_text
import utils.split_text
import uuid
import time

# Import local modules
from retrieval.retrieval_system import RetrievalSystem
from generation.generate_response import ResponseGenerator

# Load environment variables
load_dotenv()


def initialize_system():
    # check if source chunks directory has text files
    if os.path.exists("source_chunks") and len(os.listdir("source_chunks")) > 1:
        print("Using existing chunks and index...")
    else:
        print("Creating chunks and index...")
        os.makedirs("source_chunks", exist_ok=True)
        print("PDF to Text Conversion...")
        utils.pdf_to_text.convert_pdfs_to_text("source_pdfs", "source_txts")
        print("Text Splitting...")
        utils.split_text.process_texts("source_txts", "source_chunks")

        # Safely remove files if they exist
        for file in ["faiss.index", "id_mapping.pkl", "embedding_cache.json"]:
            if os.path.exists(file):
                os.remove(file)

    """Initializing the retrieval and generation systems."""
    # Initialize Retrieval System
    retrieval = RetrievalSystem(
        chunk_dir="source_chunks",
        index_path="faiss.index",
        mapping_path="id_mapping.pkl",
        cache_file="embedding_cache.json",
    )

    # Initialize Response Generator
    openai_api_key = os.getenv("OPENAI_API_KEY")
    generator = ResponseGenerator(
        openai_api_key=openai_api_key,
        model="gpt-4o",
        max_tokens=4096,
        max_conversation_history=10,  # Increased for Streamlit UI
    )

    return retrieval, generator


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="")
    st.title("Finnish Collective Agreements Chatbot ")

    # Initialize systems
    retrieval, generator = initialize_system()

    """Ready for user interaction."""

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for context
    st.sidebar.title("Retrieved Context")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # If it's a bot message, show retrieved context in sidebar
            if message["role"] == "assistant" and "retrieved_texts" in message:
                st.sidebar.markdown("### Retrieved Texts:")
                for idx, text in enumerate(message.get("retrieved_texts", []), 1):
                    # Generate a unique key for each text area
                    unique_key = f"context_{uuid.uuid4()}"
                    st.sidebar.text_area(
                        f"Context {idx}",
                        value=text,
                        height=100,
                        key=unique_key,
                    )

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Prepare conversation history context
        conversation_history_context = "\n".join(
            [f"User: {msg['content']}" for msg in st.session_state.messages]
        )

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # time retrieval
        start_time = time.time()
        # Retrieve relevant texts
        retrieved_texts = retrieval.retrieve(prompt)
        end_time = time.time()
        print(f"Retrieval time: {end_time - start_time} seconds")

        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""

            # Generate response with streaming
            token_generator = generator.generate(
                retrieved_texts,
                prompt,
                previous_context=conversation_history_context,
            )

            # Stream tokens to the response placeholder
            for token in token_generator:
                response_text += token
                response_placeholder.markdown(response_text)

            # Show retrieved context in sidebar
            st.sidebar.markdown("### Retrieved Texts:")
            for idx, text in enumerate(retrieved_texts, 1):
                # Generate a unique key for each text area
                unique_key = f"context_{uuid.uuid4()}"
                st.sidebar.text_area(
                    f"Context {idx}",
                    value=text,
                    height=100,
                    key=unique_key,
                )

        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "retrieved_texts": retrieved_texts,
            }
        )

    # Sidebar option to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
