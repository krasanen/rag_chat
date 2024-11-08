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

        Args:
            retrieved_texts (List[str]): List of relevant text chunks.
            question (str): User's question.
            language (str): Language code ('fi' for Finnish, 'en' for English).

        Returns:
            str: Generated answer from the model.
        """
        # Construct the prompt
        prompt = "Collective Agreement Chatbot\n"
        prompt += f"Language: {'Finnish' if language == 'fi' else 'English'}\n"
        prompt += "Give answers to question from these documents:\n"
        for text in retrieved_texts:
            prompt += text + "\n"
        prompt += f"Question: {question}\nAnswer:"

        # Count tokens in the prompt
        prompt_token_count = self.count_tokens(prompt)
        logger.info(f"Prompt token count: {prompt_token_count}")

        # Define model's max token limit
        model_max_tokens = 4096  # Adjust based on the model you're using

        # Calculate allowable tokens for the response
        response_max_tokens = (
            self.max_tokens - prompt_token_count - 50
        )  # Buffer of 50 tokens

        if response_max_tokens <= 0:
            logger.error("Prompt is too long for the model's maximum token limit.")
            return "Sorry, your request is too long to process."

        try:
            # Call OpenAI's API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=response_max_tokens,
                temperature=0.3,
                n=1,
                stop=None,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            logger.info("Response generated successfully.")
            return answer
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, there was an error processing your request."
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occurred. Please try again later."
