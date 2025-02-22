import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

def build_model(input_length=100):
    model = Sequential([
        Embedding(input_dim=4, output_dim=8, input_length=input_length),
        SimpleRNN(32, return_sequences=True),
        SimpleRNN(32),
        Dense(4, activation="softmax")
    ])
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
