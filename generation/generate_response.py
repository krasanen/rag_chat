# generation/generate_response.py
from typing import List, Optional, Generator, Union
import openai
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
        openai.api_key = self.openai_api_key
        self.model = model
        self.max_tokens = max_tokens
        
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

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.

        Args:
            text (str): Text to detect language for.

        Returns:
            str: Two-letter language code (default to 'en' if detection fails).
        """
        try:
            return detect(text)
        except LangDetectException:
            logger.warning(f"Language detection failed for text: {text}")
            return 'en'

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
        question: str, 
        language: Optional[str] = None, 
        previous_context: Optional[str] = None
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generates a response based on retrieved texts and the user's question.

        Args:
            retrieved_texts (List[str]): Relevant texts for context.
            question (str): User's question.
            language (str, optional): Specific language to respond in. If None, auto-detect.
            previous_context (str, optional): Additional context from previous interactions.

        Returns:
            Union[str, Generator[str, None, None]]: Generated response or token generator.
        """
        try:
            # Log the retrieval results
            logger.info(f"Question: {question}")
            logger.info(f"Number of retrieved chunks: {len(retrieved_texts)}")
            logger.info(f"previous_context: {previous_context}")

            # Detect language if not specified
            if not language:
                language = self.detect_language(question)

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

            # Calculate and log token usage
            total_tokens = sum(self.count_tokens(text) for text in retrieved_texts)
            logger.info(f"Total tokens in retrieved texts: {total_tokens}")

            # Check if total tokens exceed the limit and truncate if necessary
            max_context_tokens = (
                self.max_tokens - self.count_tokens(question) - 100
            )  # Leave some buffer
            if total_tokens > max_context_tokens:
                logger.warning(
                    f"Total tokens {total_tokens} exceed max context tokens {max_context_tokens}. Truncating retrieved texts."
                )
                truncated_texts = []
                current_tokens = 0
                for text in retrieved_texts:
                    text_tokens = self.count_tokens(text)
                    if current_tokens + text_tokens <= max_context_tokens:
                        truncated_texts.append(text)
                        current_tokens += text_tokens
                    else:
                        break
                retrieved_texts = truncated_texts
                total_tokens = current_tokens
                logger.info(f"Truncated retrieved texts to {total_tokens} tokens.")

            # Prepare context from retrieved texts
            context = "\n".join(retrieved_texts)

            # Prepare conversation history context
            history_context = ""
            for interaction in self.conversation_history:
                history_context += f"User: {interaction['user']}\nBot: {interaction['bot']}\n\n"
            
            # Combine all context sources
            full_context = f"{history_context}\n{previous_context or ''}\n{context}"
            
            # Prepare messages for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant responding in {language_name}. Use the provided context to answer the question precisely. Answer only to the question, not to the context. You are expert on Finnish Collective Agreements.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{full_context}\n\nQuestion: {question}",
                },
            ]
            
            # Generate response with streaming
            def token_generator():
                full_response = ""
                for chunk in openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    stream=True
                ):
                    if chunk['choices'][0]['finish_reason'] is not None:
                        break
                    
                    token = chunk['choices'][0]['delta'].get('content', '')
                    if token:
                        full_response += token
                        yield token
                
                # Add to conversation history after full generation
                self.add_to_conversation_history(question, full_response)
                logger.info("Response generated successfully.")
            
            return token_generator()
        
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}", exc_info=True)
            return f"Sorry, there was an error processing your request: {str(e)}"
