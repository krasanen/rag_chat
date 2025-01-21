# generation/generate_response.py
from typing import List, Optional
import openai
import tiktoken
import logging

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
        language: str = "fi", 
        previous_context: Optional[str] = None
    ) -> str:
        """
        Generates a response based on retrieved texts and the user's question.

        Args:
            retrieved_texts (List[str]): Relevant texts for context.
            question (str): User's question.
            language (str, optional): Language of the response. Defaults to "fi".
            previous_context (str, optional): Additional context from previous interactions.

        Returns:
            str: Generated response.
        """
        try:
            # Log the retrieval results
            logger.info(f"Question: {question}")
            logger.info(f"Number of retrieved chunks: {len(retrieved_texts)}")
            logger.info(f"retrieved_texts: {retrieved_texts}")

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
                {"role": "system", "content": f"You are a helpful assistant responding in {language}. Use the provided context to answer the question precisely."},
                {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {question}"}
            ]
            
            # Generate response
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            
            bot_response = response.choices[0].message.content.strip()
            
            # Add to conversation history
            self.add_to_conversation_history(question, bot_response)
            
            logger.info("Response generated successfully.")
            return bot_response, question
        
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}", exc_info=True)
            return f"Sorry, there was an error processing your request: {str(e)}"
