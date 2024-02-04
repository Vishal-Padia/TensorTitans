import pickle
import os
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Get the path to the models folder
current_dir = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(current_dir, 'models')

# Load the logistic regression model from the pickle file
model_path = os.path.join(models_folder, 'svm_imdb.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer
tfidf_path = os.path.join(models_folder, 'tfidf.pkl')

with open(tfidf_path, 'rb') as file:
    tfidf = pickle.load(file)

def predict_sentiment(text):
    # Transform the input text using the loaded TfidfVectorizer
    text_vector = tfidf.transform([text])

    # Use the loaded model to predict sentiment
    prediction = model.predict(text_vector)
    result = prediction[0]
    return result

if __name__ == '__main__':
    # Example usage:
    with open("../data/transcript.txt", "r") as file:
        transcript = file.read()
    input_text = transcript
    sentiment = predict_sentiment(input_text)
    print(f"Sentiment prediction: {sentiment}")
