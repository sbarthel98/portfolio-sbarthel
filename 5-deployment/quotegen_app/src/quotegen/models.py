import random
from collections import defaultdict
from quotegen.custom_logger import logger

class MarkovQuoteGenerator:
    def __init__(self):
        # This dictionary will hold the transitions
        # Format: {"word": ["next_word1", "next_word2"], ...}
        # It also includes special START and END tokens
        self.transition_dict = defaultdict(list)
        self.start_words = []
        self._START_TOKEN = "_START_"
        self._END_TOKEN = "_END_"

    def build_model(self, quotes: list[str]):
        """Builds the Markov transition dictionary from a list of quotes."""
        logger.info(f"Building Markov model from {len(quotes)} quotes...")
        for quote in quotes:
            words = quote.split()
            if len(words) < 2:
                continue
            
            # Add first word to start_words list
            self.start_words.append(words[0])
            
            # Add start token transition
            self.transition_dict[self._START_TOKEN].append(words[0])
            
            # Add word-to-word transitions
            for i in range(len(words) - 1):
                self.transition_dict[words[i]].append(words[i+1])
                
            # Add end token transition
            self.transition_dict[words[-1]].append(self._END_TOKEN)
            
        # Ensure start words are unique
        self.start_words = list(set(self.start_words))
        logger.info(f"Model built. {len(self.transition_dict)} unique words in transition map.")

    def generate_quote(self, max_length: int = 30) -> str:
        """Generates a single quote using the built model."""
        
        # Start with a random word from the start_words list
        if not self.start_words:
            logger.warning("Model has no start words. Is it trained?")
            return ""
            
        current_word = random.choice(self.start_words)
        quote = [current_word]
        
        for _ in range(max_length - 1):
            # Get possible next words
            next_words = self.transition_dict.get(current_word)
            
            # If no transition, or current word is not in dict, stop.
            if not next_words:
                break
                
            # Choose a random next word
            current_word = random.choice(next_words)
            
            # If we hit the end token, stop.
            if current_word == self._END_TOKEN:
                break
                
            quote.append(current_word)
            
        return " ".join(quote).capitalize()

    def sample_n(self, n: int, max_length: int = 30) -> list[str]:
        """Generates N unique quotes."""
        quotes = set()
        # Add a safety break to prevent infinite loops if uniqueness is hard
        for _ in range(n * 5): 
            if len(quotes) >= n:
                break
            quotes.add(self.generate_quote(max_length=max_length))
            
        return list(quotes)