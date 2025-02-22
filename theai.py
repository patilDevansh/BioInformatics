import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import numpy as np

# Example DNA sequence (input)
sequence = "ATCGGA"

# Mapping nucleotides to indices
nucleotide_to_index = {"A": 0, "T": 1, "C": 2, "G": 3}

# Convert sequence to numerical nodes
nodes = torch.tensor([nucleotide_to_index[nt] for nt in sequence], dtype=torch.long)

# Create edges (connect adjacent nucleotides)
edges = torch.tensor([
    [i, i + 1] for i in range(len(sequence) - 1)
] + [
    [i + 1, i] for i in range(len(sequence) - 1)  # Bidirectional edges
], dtype=torch.long).t()

# Create a graph data object
data = Data(x=nodes.unsqueeze(1).float(), edge_index=edges)

# Define the GNN model
class MutationGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(MutationGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)  # Output 4 classes (A, T, C, G)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.fc(x)
        return x

# Initialize the model
num_features = 1  # One feature per nucleotide
hidden_dim = 16
num_classes = 4  # A, T, C, G (classification)
model = MutationGNN(num_features, hidden_dim, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Example target mutations (random for now, would come from real data)
target_mutations = torch.tensor([nucleotide_to_index["G"],  # A → G
                                 nucleotide_to_index["A"],  # T → A
                                 nucleotide_to_index["T"],  # C → T
                                 nucleotide_to_index["C"],  # G → C
                                 nucleotide_to_index["G"],  # G stays G
                                 nucleotide_to_index["T"]]) # A → T

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target_mutations)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Get predicted mutations
predicted_mutations = torch.argmax(model(data), dim=1)

# Convert predicted indices back to nucleotides
index_to_nucleotide = {v: k for k, v in nucleotide_to_index.items()}
predicted_nucleotides = [index_to_nucleotide[idx.item()] for idx in predicted_mutations]

print("Predicted Mutations:", predicted_nucleotides)
