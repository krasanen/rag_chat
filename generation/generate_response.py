# generation/generate_response.py
import cohere
import os

class ResponseGenerator:
    def __init__(self, cohere_api_key):
        self.cohere_client = cohere.Client(cohere_api_key)
    
    def generate(self, retrieved_texts, question, language='fi'):
        prompt = "Collective Agreement Chatbot\n"
        prompt += f"Language: {'Finnish' if language == 'fi' else 'English'}\n"
        prompt += "Documents:\n"
        for text in retrieved_texts:
            prompt += text + "\n"
        prompt += f"Question: {question}\nAnswer:"
        
        response = self.cohere_client.generate(
            model='command-xlarge-nightly',
            prompt=prompt,
            max_tokens=1500,
            temperature=0.3,
            stop_sequences=["\n"],
        )
        return response.generations[0].text.strip()
