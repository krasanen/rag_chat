import sqlite3
import os

def create_ice_breaker_database():
    """
    Create and initialize the ice breaker phrases database
    """
    # Ensure database directory exists
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    # Connect to SQLite database (creates if not exists)
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'ice_breakers.db'))
    cursor = conn.cursor()

    # Create table for ice breaker phrases
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ice_breaker_phrases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phrase TEXT NOT NULL,
        language TEXT NOT NULL,
        category TEXT DEFAULT 'general'
    )
    ''')

    # Create table for ice breaker responses
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ice_breaker_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        language TEXT NOT NULL,
        category TEXT NOT NULL,
        response TEXT NOT NULL
    )
    ''')

    # Initial set of ice breaker phrases
    initial_phrases = [
        # Finnish
        ('terve', 'fi', 'greeting'),
        ('hei', 'fi', 'greeting'),
        ('moi', 'fi', 'greeting'),
        ('moikka', 'fi', 'greeting'),
        ('terve hei', 'fi', 'greeting'),
        ('päivää', 'fi', 'greeting'),
        ('hyvää päivää', 'fi', 'greeting'),
        ('mitä kuuluu', 'fi', 'small_talk'),
        ('kuinka voit', 'fi', 'small_talk'),
        ('hauska tavata', 'fi', 'introduction'),

        # English
        ('hi', 'en', 'greeting'),
        ('hello', 'en', 'greeting'),
        ('hey', 'en', 'greeting'),
        ('greetings', 'en', 'greeting'),
        ('good morning', 'en', 'greeting'),
        ('good afternoon', 'en', 'greeting'),
        ('good evening', 'en', 'greeting'),
        ('what\'s up', 'en', 'small_talk'),
        ('how are you', 'en', 'small_talk'),
        ('nice to meet you', 'en', 'introduction'),

        # Swedish
        ('hej', 'sv', 'greeting'),
        ('hallå', 'sv', 'greeting'),
        ('god dag', 'sv', 'greeting'),
        ('hur mår du', 'sv', 'small_talk'),
        ('trevligt att träffas', 'sv', 'introduction')
    ]

    # Initial set of ice breaker responses
    initial_responses = [
        # Finnish Greeting Responses
        ('fi', 'greeting', 'Terve! Mitä sinulle kuuluu?'),
        ('fi', 'greeting', 'Hei! Hauska tavata!'),
        ('fi', 'greeting', 'Päivää! Miten voin auttaa?'),
        ('fi', 'small_talk', 'Kiva, että kysyit! Mitä kuuluu?'),
        ('fi', 'introduction', 'Hauska tavata! Miten voin auttaa?'),

        # English Greeting Responses
        ('en', 'greeting', 'Hi there! How can I help you today?'),
        ('en', 'greeting', 'Hello! What can I do for you?'),
        ('en', 'greeting', 'Greetings! How are you doing?'),
        ('en', 'small_talk', 'Great! How can I assist you?'),
        ('en', 'introduction', 'Nice to meet you! What brings you here?'),

        # Swedish Greeting Responses
        ('sv', 'greeting', 'Hej! Hur kan jag hjälpa dig?'),
        ('sv', 'greeting', 'Hallå! Vad kan jag göra för dig?'),
        ('sv', 'small_talk', 'Bra, tack! Hur kan jag hjälpa?'),
        ('sv', 'introduction', 'Trevligt att träffas! Vad kan jag hjälpa dig med?')
    ]

    # Insert phrases, ignore duplicates
    cursor.executemany('''
    INSERT OR IGNORE INTO ice_breaker_phrases (phrase, language, category) 
    VALUES (?, ?, ?)
    ''', initial_phrases)

    # Insert responses, ignore duplicates
    cursor.executemany('''
    INSERT OR IGNORE INTO ice_breaker_responses (language, category, response) 
    VALUES (?, ?, ?)
    ''', initial_responses)

    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_ice_breaker_database()
