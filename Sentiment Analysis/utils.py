from nltk import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np


nltk.download('stopwords')

# Pre-compile regex patterns and initialize once (avoid recompilation)
_SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z\s]+")
_WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')
_STOP_WORDS = set(stopwords.words('english'))
_STEMMER = PorterStemmer()

def preprocess_text(text):
    """
    Preprocess the input text by removing special characters, converting to lowercase,
    removing stopwords, and stemming the words.
    The goal is to prepare the text for sentiment analysis.

    Args:
        text (str): The input text to preprocess.
    Returns:
        str: The preprocessed text.
    
    """
    # Use pre-compiled pattern (faster than recompiling each time)
    text = _SPECIAL_CHARS_PATTERN.sub(" ", text)
    
    # Extract words using pre-compiled pattern
    words = _WORD_PATTERN.findall(text)
    text = ' '.join(words).lower()
    
    # Filter stopwords and stem in one pass (avoid creating intermediate lists)
    words = [_STEMMER.stem(word) for word in text.split() if word not in _STOP_WORDS]
    
    return ' '.join(words)

def build_frequency_dict(texts, labels):
    """
    Build a frequency dictionary mapping (word, label) pairs to their frequency counts.
    Optimized for speed with efficient dict operations.

    Args:
        texts (list of str): List of preprocessed text documents.
        labels (list of int): Corresponding list of sentiment labels (0 or 1).
    
    Returns:
        dict: A dictionary with keys as (word, label) tuples and values as frequency counts.
    """
    frequency_dict = {}
    # Use setdefault for faster dict operations (single lookup instead of two)
    for text, label in zip(texts, labels):
        for word in text.split():
            pair = (word, label)
            frequency_dict[pair] = frequency_dict.get(pair, 0) + 1
    return frequency_dict

def extract_features(text, frequency_dict):
    """
    Extract sentiment features from raw text by:
    1. Preprocessing: removes special chars, stopwords, and applies stemming via preprocess_text()
    2. Counting positive/negative sentiment words from the frequency dictionary

    Args:
        text (str): Raw or preprocessed text (WILL BE PREPROCESSED automatically)
        frequency_dict (dict): The frequency dictionary mapping (word, label) tuples to counts.
                              Built from preprocessed training texts.
        
    Returns:
        np.array: A feature vector [bias, positive_count, negative_count]
                  where positive_count = sum of sentiment word frequencies for label=1
                  and negative_count = sum of sentiment word frequencies for label=0
        """
    # IMPORTANT: preprocess_text() is called here automatically
    # It: removes special chars, converts to lowercase, removes stopwords, applies stemming
    words = preprocess_text(text).split()
    
    # Count sentiment words by looking them up in frequency_dict
    # frequency_dict was built from preprocessed training texts
    positive_count = sum(frequency_dict.get((word, 1), 0) for word in words)
    negative_count = sum(frequency_dict.get((word, 0), 0) for word in words)
    
    return np.array([1, positive_count, negative_count])


"""print(preprocess_text("This is a sample text! With special characters #$% and stopwords."))
print(build_frequency_dict(
    ["I love this movie", "I hate this movie", "This movie is great", "This movie is terrible"],
    [1, 0, 1, 0]
))
print(extract_features("I love this great movie", {
    ('love', 1): 2,
    ('great', 1): 3,
    ('hate', 0): 1,
    ('terrible', 0): 2
}))"""    