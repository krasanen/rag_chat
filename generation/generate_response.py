# generation/generate_response.py
from typing import List, Optional, Generator, Union
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
        previous_context: Optional[List[Union[dict, str]]] = None,
        language: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate a response based on retrieved context and query.

        Args:
            retrieved_texts (List[str]): List of retrieved context texts.
            query (str): User's query.
            previous_context (Optional[List[Union[dict, str]]]): Previous conversation context.
            language (Optional[str]): Language of the query.

        Yields:
            Generator of response tokens.
        """
        # Logging
        logger.info(f"Question: {query}")
        logger.info(f"Number of retrieved chunks: {len(retrieved_texts)}")

        # Prepare conversation context
        if previous_context is None:
            previous_context = []
        
        # Convert previous context to message format if needed
        messages = []
        for context in previous_context:
            if isinstance(context, dict):
                messages.append(context)
            elif isinstance(context, str):
                messages.append({"role": "user", "content": context})

        # Detect language if not provided
        try:
            if not language:
                language = detect(query)
        except LangDetectException:
            language = 'en'

        # Prepare system prompt based on language
        system_prompt = (
            "You are a helpful AI assistant that provides accurate and concise answers. "
            "Use the provided context to inform your response, but do not simply repeat it. "
            "If the context does not contain relevant information, acknowledge that honestly."
        )

        # Combine retrieved texts into context
        context_text = "\n\n".join(retrieved_texts)
        logger.info(f"Total tokens in retrieved texts: {self.count_tokens(context_text)}")

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuery: {query}"}
        ]

        # Generate response with streaming
        def token_generator():
            full_response = ""
            response_chunks = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in response_chunks:
                # Check if the chunk has a content delta
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token
                
                # Break if finish reason is 'stop'
                if chunk.choices and chunk.choices[0].finish_reason == 'stop':
                    break

            # Optional: log the full response
            logger.info(f"Full response: {full_response}")

        # Return the generator
        return token_generator()
