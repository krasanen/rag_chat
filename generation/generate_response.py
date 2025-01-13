# generation/generate_response.py
from typing import List
import openai
import os
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:

    def __init__(
        self, openai_api_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1024
    ):
        """
        Initializes the ResponseGenerator.

        Args:
            openai_api_key (str): OpenAI API key.
            model (str): OpenAI model to use.
            max_tokens (int): Maximum tokens allowed for the response.
        """
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key
        self.model = model
        self.max_tokens = max_tokens  # Maximum tokens for the response

        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding(
            "gpt2"
        )  # GPT-3 models use the "gpt2" encoding

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: Number of tokens.
        """
        return len(self.encoding.encode(text))

    def generate(
        self, retrieved_texts: List[str], question: str, language: str = "fi"
    ) -> str:
        """
        Generates a response based on retrieved texts and the user's question.
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

            # Construct the optimized prompt
            system_prompt = f"""You are an expert assistant.
            Your task is to:
            1. Carefully analyze the provided document excerpts.
            2. Find the specific sections that answer the user's question.
            3. Provide a clear, structured answer that:
               - Directly addresses the question.
               - Cites specific sections when applicable.
               - Includes all relevant details and conditions.
               - Is organized with bullet points or numbering when listing multiple criteria.
            4. Only state facts.
            5. Answer in the same language as the question.

            Always maintain high accuracy and only use information from the provided excerpts."""

            user_prompt = f"""Relevant Collective Agreement Excerpts:
            {'-' * 40}
            {chr(10).join(retrieved_texts)}
            {'-' * 40}

            Question: {question}

            Please provide a detailed answer based only on the above excerpts. Include section numbers and quote relevant parts when applicable."""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,  # Keep this low for factual responses
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            logger.info("Response generated successfully.")
            return answer, question
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}", exc_info=True)
            return f"Sorry, there was an error processing your request: {str(e)}"
