from Bio import SeqIO
import torch
from torch_geometric.data import Data
import os
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

class DNASequenceProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.nucleotide_to_index = {"A": 0, "T": 1, "C": 2, "G": 3}
        self.index_to_nucleotide = {v: k for k, v in self.nucleotide_to_index.items()}
    
    def load_sequences(self):
        """ Reads the FASTA file and extracts DNA sequences. """
        sequences = []
        for record in SeqIO.parse(self.file_path, "fasta"):
            cleaned_seq = self.clean_sequence(str(record.seq))
            if cleaned_seq:
                sequences.append(cleaned_seq)
        return sequences

    def clean_sequence(self, sequence):
        """ Removes invalid characters (like N, gaps, etc.) and returns a clean DNA sequence. """
        valid_bases = set(self.nucleotide_to_index.keys())
        cleaned = ''.join([base for base in sequence if base in valid_bases])
        return cleaned if len(cleaned) > 5 else None  # Ignore very short sequences
    
    def sequence_to_graph(self, sequence):
        """ Converts a DNA sequence into a PyTorch Geometric graph. """
        nodes = torch.tensor([self.nucleotide_to_index[nt] for nt in sequence], dtype=torch.long)
        edges = torch.tensor(
            [[i, i + 1] for i in range(len(sequence) - 1)] + 
            [[i + 1, i] for i in range(len(sequence) - 1)], 
            dtype=torch.long
        ).t()  # Bi-directional edges
        
        return Data(x=nodes.unsqueeze(1).float(), edge_index=edges)
    
    def process(self):
        """ Loads sequences, cleans them, and converts them into graphs. """
        sequences = self.load_sequences()
        graphs = [self.sequence_to_graph(seq) for seq in sequences]
        return graphs

# Load and process data
file_path = "BioInformatics/Sequences.fasta"
processor = DNASequenceProcessor(file_path)
graph_data = processor.process()

print(f"Processed {len(graph_data)} sequences into graph structures.")

class MutationGNN(nn.Module):
    def __init__(self, num_features=1, hidden_dim=16, num_classes=4):
        super(MutationGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)  # Outputs (A, T, C, G)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.fc(x)
        return x

# Initialize model, loss, and optimizer
model = MutationGNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Example target mutations (random labels, replace with real mutations)
target_mutations = torch.randint(0, 4, (graph_data[0].x.size(0),))  # Random A, T, C, G labels

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(graph_data[0])  # Using the first sequence graph as input
    loss = criterion(output, target_mutations)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Get predicted mutations
predicted_mutations = torch.argmax(model(graph_data[0]), dim=1)
predicted_nucleotides = [processor.index_to_nucleotide[idx.item()] for idx in predicted_mutations]

print("Predicted Mutations:", predicted_nucleotides)
