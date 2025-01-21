import os
import streamlit as st
from dotenv import load_dotenv

# Import local modules
from retrieval.retrieval_system import RetrievalSystem
from generation.generate_response import ResponseGenerator

# Load environment variables
load_dotenv()


def initialize_system():
    """Initialize the retrieval and generation systems."""
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
        max_tokens=2048,
        max_conversation_history=10,  # Increased for Streamlit UI
    )

    return retrieval, generator


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
    st.title("Finnish Collective Agreements Chatbot 💬")

    # Initialize systems
    retrieval, generator = initialize_system()

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
                for idx, text in enumerate(message["retrieved_texts"], 1):
                    st.sidebar.text_area(
                        f"Context {idx}",
                        value=text,
                        height=100,
                        key=f"context_{len(st.session_state.messages)}_{idx}",
                    )

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve relevant texts
        retrieved_texts = retrieval.retrieve(prompt)

        # Generate response
        response, _ = generator.generate(retrieved_texts, prompt, language="fi")

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

            # Show retrieved context in sidebar
            st.sidebar.markdown("### Retrieved Texts:")
            for idx, text in enumerate(retrieved_texts, 1):
                st.sidebar.text_area(
                    f"Context {idx}",
                    value=text,
                    height=100,
                    key=f"context_{len(st.session_state.messages)}_{idx}",
                )

        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "retrieved_texts": retrieved_texts,
            }
        )

    # Sidebar option to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
