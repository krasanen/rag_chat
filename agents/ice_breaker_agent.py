import re
from typing import Optional, Dict, Any

class IceBreakerAgent:
    """
    Agent responsible for handling common ice breaker phrases
    """
    # Comprehensive list of ice breaker phrases in Finnish and other common languages
    ICE_BREAKER_PHRASES = {
        # Finnish
        'fi': [
            'terve', 'hei', 'moi', 'moikka', 'terve hei', 'päivää', 'hyvää päivää', 
            'mitä kuuluu', 'kuinka voit', 'hauska tavata', 'miten menee'
        ],
        # English
        'en': [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'what\'s up', 'how are you', 'nice to meet you'
        ],
        # Swedish
        'sv': [
            'hej', 'hallå', 'god dag', 'hur mår du', 'trevligt att träffas'
        ]
    }

    def __init__(self):
        """
        Initialize the ice breaker agent
        """
        # Compile regex patterns for efficient matching
        self.ice_breaker_patterns = {
            lang: [re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE) 
                   for phrase in phrases]
            for lang, phrases in self.ICE_BREAKER_PHRASES.items()
        }

    def is_ice_breaker(self, text: str) -> bool:
        """
        Check if the input text is an ice breaker phrase
        
        Args:
            text (str): Input text to check
        
        Returns:
            bool: True if the text is an ice breaker, False otherwise
        """
        # Check for ice breaker phrases in all languages
        for lang_patterns in self.ice_breaker_patterns.values():
            for pattern in lang_patterns:
                if pattern.search(text):
                    return True
        return False

    def generate_ice_breaker_response(self, text: str) -> Optional[str]:
        """
        Generate a friendly response for ice breaker phrases
        
        Args:
            text (str): Input ice breaker text
        
        Returns:
            Optional[str]: Friendly response or None
        """
        # Detect language (simple approach)
        text_lower = text.lower().strip()
        
        # Finnish responses
        if any(phrase in text_lower for phrase in self.ICE_BREAKER_PHRASES['fi']):
            responses = [
                "Terve! Mitä sinulle kuuluu?",
                "Hei! Hauska tavata!",
                "Päivää! Miten voin auttaa?",
            ]
        # English responses
        elif any(phrase in text_lower for phrase in self.ICE_BREAKER_PHRASES['en']):
            responses = [
                "Hi there! How can I help you today?",
                "Hello! What can I do for you?",
                "Greetings! How are you doing?",
            ]
        # Swedish responses
        elif any(phrase in text_lower for phrase in self.ICE_BREAKER_PHRASES['sv']):
            responses = [
                "Hej! Hur kan jag hjälpa dig?",
                "Hallå! Vad kan jag göra för dig?",
            ]
        else:
            # Fallback response
            responses = [
                "Hello! How can I assist you today?",
            ]
        
        # Import random here to avoid global import
        import random
        return random.choice(responses)

    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process the input text and generate an ice breaker response if applicable
        
        Args:
            input_text (str): Input text to process
        
        Returns:
            Dict with processing result
        """
        # Check if the input is an ice breaker
        if self.is_ice_breaker(input_text):
            return {
                'is_ice_breaker': True,
                'response': self.generate_ice_breaker_response(input_text)
            }
        
        # If not an ice breaker, return negative result
        return {
            'is_ice_breaker': False,
            'response': None
        }
