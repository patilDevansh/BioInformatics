try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from Bio import SeqIO
    import pandas as pd
except ModuleNotFoundError as e:
    print("Error: Required modules not found. Please install them using 'pip install torch torch_geometric biopython pandas'")
    raise e

from collections import defaultdict
import random
import os

### 1️⃣ DATA PROCESSING: FASTA FILE LOADING & CONVERTING TO DATAFRAME ###

def load_fasta_to_dataframe(file_path):
    """Loads influenza sequences from a FASTA file and converts them to a pandas DataFrame."""
    data = []
    for record in SeqIO.parse(file_path, "fasta"):
        data.append([record.id, str(record.seq)])
    return pd.DataFrame(data, columns=["Sequence_ID", "Sequence"])

### 2️⃣ FEATURE ENGINEERING: CODON PROCESSING & MUTATION ESTIMATION ###

def sequence_to_codons(sequence):
    """Converts a nucleotide sequence into codon triplets."""
    return [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]

def calculate_mutation_probabilities(sequences):
    """Estimates mutation probabilities for each codon position."""
    codon_counts = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        codons = sequence_to_codons(seq)
        for i, codon in enumerate(codons):
            codon_counts[i][codon] += 1
    
    mutation_probs = {}
    for position, codon_freqs in codon_counts.items():
        most_common_codon = max(codon_freqs, key=codon_freqs.get)
        total_mutations = sum(codon_freqs.values()) - codon_freqs[most_common_codon]
        mutation_probs[position] = total_mutations / sum(codon_freqs.values())
    
    return mutation_probs

### 3️⃣ GRAPH DATA PREPARATION ###

def create_graph_from_sequence(sequence, mutation_probs):
    """Creates a graph structure from a sequence."""
    codons = sequence_to_codons(sequence)
    nodes = torch.tensor([mutation_probs.get(i, 0.0) for i in range(len(codons))], dtype=torch.float).unsqueeze(1)
    edges = torch.tensor([[i, i + 1] for i in range(len(codons) - 1)] +
                         [[i + 1, i] for i in range(len(codons) - 1)], dtype=torch.long).t()
    return Data(x=nodes, edge_index=edges)

### 4️⃣ GNN MODEL: MUTATION PREDICTION ###

class MutationGNN(nn.Module):
    def __init__(self, num_features=1, hidden_dim=16, embedding_dim=32):
        super(MutationGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.normalize(x, dim=1)  # Normalize embeddings

def contrastive_loss(embedding1, embedding2, temperature=0.5):
    cosine_sim = F.cosine_similarity(embedding1, embedding2)
    loss = -torch.log(torch.exp(cosine_sim / temperature) /
                      torch.sum(torch.exp(cosine_sim / temperature)))
    return loss

### 5️⃣ TRAINING & EVALUATION ###

def train_model(graph_data, num_epochs=100, learning_rate=0.01):
    model = MutationGNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(graph_data)
        loss = contrastive_loss(output, output)  # Self-supervised contrastive learning
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model

### 6️⃣ FULL PIPELINE EXECUTION ###

def main(fasta_files):
    sequences_df = pd.concat([load_fasta_to_dataframe(file) for file in fasta_files], ignore_index=True)
    mutation_probs = calculate_mutation_probabilities(sequences_df["Sequence"].tolist())
    graph_data = create_graph_from_sequence(sequences_df["Sequence"].iloc[0], mutation_probs)
    model = train_model(graph_data)
    return model

if __name__ == "__main__":
    fasta_files = ["BioInformatics/sequences_20250223_3139619.fasta", "BioInformatics/sequences_20250223_5860499.fasta"]
    trained_model = main(fasta_files)
