import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')

def preprocess(text):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    texts = [preprocess(text) for text in data['text']]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, data['label'], vectorizer
