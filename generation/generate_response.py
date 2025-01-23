# generation/generate_response.py
from typing import List, Optional, Generator, Union, Dict
from openai import OpenAI

import tiktoken
import logging
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:

    def __init__(
        self, openai_api_key: str, model: str = "gpt-4o", max_tokens: int = 1024, 
        max_conversation_history: int = 5
    ):
        """
        Initializes the ResponseGenerator.

        Args:
            openai_api_key (str): OpenAI API key.
            model (str): OpenAI model to use.
            max_tokens (int): Maximum tokens allowed for the response.
            max_conversation_history (int): Maximum number of previous interactions to remember.
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding("gpt2")

        # Conversation history management
        self.max_conversation_history = max_conversation_history
        self.conversation_history = []

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: Number of tokens.
        """
        return len(self.encoding.encode(text))

    def add_to_conversation_history(self, user_query: str, bot_response: str):
        """
        Adds a user query and bot response to the conversation history.

        Args:
            user_query (str): The user's input.
            bot_response (str): The bot's response.
        """
        self.conversation_history.append({
            "user": user_query,
            "bot": bot_response
        })

        # Trim conversation history if it exceeds max_conversation_history
        if len(self.conversation_history) > self.max_conversation_history:
            self.conversation_history = self.conversation_history[-self.max_conversation_history:]

    def generate(
        self, 
        retrieved_texts: List[str], 
        query: str, 
        previous_context: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """
        Generate a response based on retrieved context and query
        
        Args:
            retrieved_texts: List of retrieved context texts
            query: User's input query
            previous_context: Previous conversation history
        
        Returns:
            Generator yielding response tokens
        """
        # Logging
        logger.info(f"Question: {query}")
        logger.info(f"Number of retrieved chunks: {len(retrieved_texts)}")

        # Prepare conversation context
        if previous_context is None:
            previous_context = []

        # Detect language of the query
        try:
            language = detect(query)
        except LangDetectException:
            language = 'en'

        logger.info(f"Detected language: {language}")

        # Language mapping for OpenAI
        language_map = {
            'fi': 'Finnish',
            'en': 'English',
            'sv': 'Swedish',
            'ru': 'Russian',
            'de': 'German'
        }

        # Fallback to English if language not in map
        language_name = language_map.get(language, 'English')

        # Prepare system prompt based on language
        system_prompt = (
            "You are a helpful AI assistant that provides accurate and concise answers. "
            "Use the provided context to answer the question. "
            "If the context does not contain relevant information, acknowledge that honestly. "
            f"Respond in {language_name}"
        )

        # Prepare conversation context messages
        conversation_context = []
        for msg in previous_context:
            if msg.get('role') == 'user':
                conversation_context.append({"role": "user", "content": msg.get('content', '')})
            elif msg.get('role') == 'assistant':
                conversation_context.append({"role": "assistant", "content": msg.get('content', '')})

        # Combine retrieved texts into context
        context = "\n\n".join(retrieved_texts) if retrieved_texts else "No additional context available."

        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_context,  # Add previous conversation context
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        # Stream response from OpenAI
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=self.openai_api_key)

            # Stream the response
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                stream=True
            )

            # Yield tokens one by one
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"An error occurred: {e}"
