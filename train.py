from preprocessing import load_fasta, one_hot_encode
from model.py import build_model
import numpy as np

# Load and preprocess data
sequences = load_fasta("data/sequences.fasta")
X_train = np.array([one_hot_encode(seq[:100]) for seq in sequences])  # Trim to 100 bases
y_train = np.random.randint(0, 4, size=(len(X_train), 4))  # Dummy labels (replace with actual)

# Build and train the model
model = build_model(input_length=100)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save model
model.save("virus_model.h5")
