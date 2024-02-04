import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pickle

# Function to load and prepare the dataset
def load_and_prepare_data():
    dataset = load_dataset("go_emotions", "raw")
    df = pd.DataFrame(dataset["train"])

    selected_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
        'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]
    labels = df[selected_labels].values
    texts = df['text'].tolist()

    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, mlb

# Function to preprocess the text
def preprocess_text(X_train, X_test):
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences, maxlen=100, padding='post', truncating='post')
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = pad_sequences(test_sequences, maxlen=100, padding='post', truncating='post')
    return train_padded, test_padded, tokenizer

# Function to build the model
def build_model(num_labels):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=16, input_length=100),
        LSTM(64),
        Dense(24, activation='relu'),
        Dense(num_labels, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    return model


def predict_sentiment(text, tokenizer, model, mlb):
    # Convert the input text to a sequence and pad it
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

    # Predict probabilities for each label
    prediction = model.predict(padded)

    # Convert probabilities to label indices based on a threshold (e.g., 0.5)
    # Ensure we're correctly identifying indices of predicted labels
    predicted_labels = [mlb.classes_[i] for i, prob in enumerate(prediction[0]) if prob > 0.5]

    return predicted_labels

# Main execution block
if __name__ == "__main__":
    # X_train, X_test, y_train, y_test, mlb = load_and_prepare_data()
    # train_padded, test_padded, tokenizer = preprocess_text(X_train, X_test)
    # model = build_model(len(mlb.classes_))
    # model.fit(train_padded, y_train, epochs=10, validation_data=(test_padded, y_test), batch_size=32)
    #
    # # Save the model, tokenizer, and mlb
    # model.save("my_model")
    # with open("my_models/tokenizer.pickle", "wb") as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("my_models/mlb.pickle", "wb") as handle:
    #     pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # To predict sentiment from new text after loading the model, tokenizer, and mlb:
    loaded_model = load_model("my_models/my_model")
    with open("my_models/tokenizer.pickle", "rb") as handle:
        loaded_tokenizer = pickle.load(handle)
    with open("my_models/mlb.pickle", "rb") as handle:
        loaded_mlb = pickle.load(handle)
    # Use loaded_model, loaded_tokenizer, and loaded_mlb for predictions as shown in previous examples.
    # Now you can use loaded_model, loaded_tokenizer, and loaded_mlb for predictions
    input_text = "This is not a fantastic day!"
    predicted_labels = predict_sentiment(input_text, loaded_tokenizer, loaded_model, loaded_mlb)
    print(f"Predicted sentiments: {predicted_labels}")

