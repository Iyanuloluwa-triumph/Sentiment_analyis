from nltk import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np


nltk.download('stopwords')
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
    text = re.sub(r"[^a-zA-Z\s]+", " ", text)  # Remove special characters or words with these characters
    
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    text = ' '.join(words) # Keep only words with alphabetic characters
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  # Stemming
    preprocessed_text = ' '.join(words)
    return preprocessed_text

#Gold standard is logistic regression with sentiment lexicon features

def build_frequency_dict(texts, labels):
    """
    Build a frequency dictionary mapping (word, label) pairs to their frequency counts.

    Args:
        texts (list of str): List of preprocessed text documents.
        labels (list of int): Corresponding list of sentiment labels (0 or 1).
    
    Returns:
        dict: A dictionary with keys as (word, label) tuples and values as frequency counts.
    """
     # in case labels are in a 2D array
    frequency_dict = {}
    for text, label in zip(texts, labels):
        words = text.split()
        for word in words:
            pair = (word, label)
            if pair in frequency_dict:
                frequency_dict[pair] += 1
            else:
                frequency_dict[pair] = 1
    return frequency_dict

def extract_features(text, frequency_dict):
    """
    Extract features from the input text based on the frequency dictionary.

    Args:
        text (str): The preprocessed text document.
        frequency_dict (dict): The frequency dictionary mapping (word, label) pairs to counts.
        
    Returns:
        np.array: A feature vector [bias, positive_count, negative_count].
        """
    words = preprocess_text(text).split()
    positive_count = 0
    negative_count = 0
    for word in words:
        positive_count += frequency_dict.get((word, 1), 0)
        negative_count += frequency_dict.get((word, 0), 0)
    feature_vector = np.array([1, positive_count, negative_count])
    return feature_vector


print(preprocess_text("This is a sample text! With special characters #$% and stopwords."))
print(build_frequency_dict(
    ["I love this movie", "I hate this movie", "This movie is great", "This movie is terrible"],
    [1, 0, 1, 0]
))
print(extract_features("I love this great movie", {
    ('love', 1): 2,
    ('great', 1): 3,
    ('hate', 0): 1,
    ('terrible', 0): 2
}))   