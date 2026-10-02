import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Safe download (only runs if missing)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean a single text string.
    Steps: lowercase -> remove URLs -> remove mentions/hashtags
           -> remove punctuation/numbers -> tokenize -> remove stopwords -> stem
    """
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove mentions (@user) and hashtag symbol (keep word)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    words = text.split()

    # Remove stopwords and stem
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)


if __name__ == "__main__":
    sample = "I LOVE this product!!! Check https://xyz.com @user #awesome"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))