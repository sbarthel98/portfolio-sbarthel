import random
from quotegen.custom_logger import logger

_START_TOKEN = "_START_"
_END_TOKEN = "_END_"

def _generate_quote(model: dict, max_length: int) -> str:
    """
    Generates a single quote using the loaded Markov model dictionary.
    """
    transitions = model.get("transitions", {})
    start_words = model.get("starts", [])

    if not start_words:
        logger.warning("Model dictionary has no start words.")
        return "Model is not trained yet."

    # Start with a random word
    current_word = random.choice(start_words)
    quote = [current_word]

    for _ in range(max_length - 1):
        next_words = transitions.get(current_word)
        
        if not next_words:
            break
            
        current_word = random.choice(next_words)
        
        if current_word == _END_TOKEN:
            break
            
        quote.append(current_word)
        
    return " ".join(quote).capitalize()


def sample_n(n: int, model: dict, max_length: int = 30) -> list[str]:
    """Generates N unique quotes."""
    quotes = set()
    # Add a safety break
    for _ in range(n * 5):
        if len(quotes) >= n:
            break
        quotes.add(_generate_quote(model, max_length))
    
    return list(quotes)