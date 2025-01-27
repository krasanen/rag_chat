import re
import os
import sqlite3
from typing import Optional, Dict, Any, List

class IceBreakerAgent:
    """
    Agent responsible for handling common ice breaker phrases from database
    """
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the ice breaker agent with database connection
        
        Args:
            db_path: Optional path to the ice breakers database
        """
        # Use default database path if not provided
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'database', 
                'ice_breakers.db'
            )
        
        # Ensure database exists
        if not os.path.exists(db_path):
            from database.init_ice_breaker_db import create_ice_breaker_database
            create_ice_breaker_database()
        
        # Connect to database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Precompile regex patterns for efficiency
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Compile regex patterns for all ice breaker phrases
        
        Returns:
            Dictionary of language-specific regex patterns
        """
        # Fetch all phrases from database
        self.cursor.execute("SELECT phrase, language FROM ice_breaker_phrases")
        phrases = self.cursor.fetchall()
        
        # Compile patterns
        patterns = {}
        for phrase, lang in phrases:
            if lang not in patterns:
                patterns[lang] = []
            patterns[lang].append(re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE))
        
        return patterns

    def is_ice_breaker(self, input_text: str) -> bool:
        """
        Check if the input text is a pure ice breaker phrase
        
        Args:
            input_text: Input text to check
        
        Returns:
            True if the input is a pure ice breaker phrase, False otherwise
        """
        # Normalize input
        normalized_text = input_text.strip().lower()
        
        # Split input into words
        words = normalized_text.split()
        
        # First word is the potential ice breaker
        first_word = words[0]
        
        # If there are additional words after the first word, it's not a pure ice breaker
        if len(words) > 1:
            return False
        
        # Check against compiled regex patterns for each language
        for lang, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(first_word):
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
        # Detect language and get category
        self.cursor.execute("""
            SELECT language, category FROM ice_breaker_phrases 
            WHERE ? LIKE '%' || phrase || '%'
        """, (text,))
        result = self.cursor.fetchone()
        
        if not result:
            return "Hello! How can I help you today?"
        
        language, category = result
        
        # Fetch predefined responses based on language and category
        self.cursor.execute("""
            SELECT response FROM ice_breaker_responses 
            WHERE language = ? AND category = ?
            ORDER BY RANDOM() LIMIT 1
        """, (language, category))
        
        response = self.cursor.fetchone()
        return response[0] if response else "Hello! How can I help you today?"

    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process the input text and generate an ice breaker response if applicable
        
        Args:
            input_text (str): Input text to process
        
        Returns:
            Dict with processing result
        """
        return {
            'response': self.generate_ice_breaker_response(input_text)
        }

    def __del__(self):
        """
        Close database connection when object is deleted
        """
        if hasattr(self, 'conn'):
            self.conn.close()
