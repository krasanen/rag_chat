import os
import streamlit as st
from dotenv import load_dotenv
import utils.pdf_to_text
import utils.split_text
import uuid
import time
from pathlib import Path
from typing import List, Dict

# Import local modules
from retrieval.retrieval_system import RetrievalSystem
from generation.generate_response import ResponseGenerator
from workflows.conversation_workflow import ConversationWorkflow
from agents.ice_breaker_agent import IceBreakerAgent

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


def get_source_pdf_links() -> List[Dict[str, str]]:
    """
    Retrieve links to source PDF files with their web URLs
    
    Returns:
        List of dictionaries with PDF file information and web URLs
    """
    source_pdfs_dir = Path(__file__).parent / 'source_pdfs'
    pdf_links = []
    
    for pdf_file in source_pdfs_dir.glob('*.pdf'):
        # Try to find corresponding .url file
        url_file = source_pdfs_dir / f"{pdf_file.stem}.url"
        web_url = ''
        
        # Read URL from .url file if it exists
        if url_file.exists():
            with open(url_file, 'r') as f:
                web_url = f.read().strip()
        
        pdf_links.append({
            'filename': pdf_file.name,
            'path': str(pdf_file),
            'web_url': web_url,
            'url': f'file://{pdf_file}'
        })
    
    return pdf_links


def main():
    # Set page configuration
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
    st.title("Finnish Collective Agreements Chatbot")

    # Initialize session state for agent settings if not exists
    if 'agent_settings' not in st.session_state:
        st.session_state.agent_settings = {
            'ice_breaker': {
                'enabled': True,
                'type': 'default'
            },
            # Placeholder for future agents
            'other_agents': []
        }

    # Sidebar for agent settings
    with st.sidebar:
        st.header("🤖 Agent Settings")
        
        # Ice Breaker Agent Section
        st.subheader("Ice Breaker Agent")
        
        # Enable/Disable Toggle
        st.session_state.agent_settings['ice_breaker']['enabled'] = st.toggle(
            "Enable Ice Breaker", 
            value=st.session_state.agent_settings['ice_breaker']['enabled'],
            help="Automatically detect and respond to greeting phrases"
        )

    # Sidebar for source PDF links
    with st.sidebar:
        st.header("📄 Source Documents")
        source_pdfs = get_source_pdf_links()
        
        for pdf in source_pdfs:
            if pdf['web_url']:
                st.markdown(f"[{pdf['filename']}]({pdf['web_url']})")
        
    # Initialize systems
    retrieval, generator = initialize_system()

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

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

        # Conditionally initialize ice breaker agent based on settings
        ice_breaker_agent = (
            IceBreakerAgent() 
            if st.session_state.agent_settings['ice_breaker']['enabled'] 
            else None
        )

        # Initialize conversation workflow
        conversation_workflow = ConversationWorkflow(
            retrieval_system=retrieval, 
            response_generator=generator,
            ice_breaker_agent=ice_breaker_agent,
            enable_ice_breaker=st.session_state.agent_settings['ice_breaker']['enabled']
        )

        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""

            # Run the workflow and get the result
            token_generator = conversation_workflow.run(
                input_text=prompt, 
                chat_history=st.session_state.messages
            )

            # Stream the response
            for token in token_generator:
                response_text += token
                response_placeholder.markdown(response_text)

            # Extract retrieved texts (use empty list for ice breakers)
            retrieved_texts = conversation_workflow.workflow_result.get('retrieved_texts', []) or []

            # Show retrieved context in sidebar only for non-ice breaker responses
            if not conversation_workflow.workflow_result.get('is_ice_breaker', False):
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

        # Display the response
        response_placeholder.markdown(response_text)

        # Update chat history after displaying the response
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
