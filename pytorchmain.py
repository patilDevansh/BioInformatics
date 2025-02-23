import itertools
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

### DATASET CLEANING AND PROCESSING ###
# This processor now splits influenza gene sequences (HA/NA) into codons and creates a graph
# where each node is a codon represented as a one-hot vector over 64 possible codons.
class InfluenzaSequenceProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # Generate codon mapping for 64 codons (AAA, AAC, …)
        nucleotides = ['A', 'T', 'C', 'G']
        self.codons = [''.join(c) for c in itertools.product(nucleotides, repeat=3)]
        self.codon_to_index = {codon: i for i, codon in enumerate(self.codons)}
        self.index_to_codon = {i: codon for codon, i in self.codon_to_index.items()}
    
    def load_sequences(self):
        sequences = []
        for record in SeqIO.parse(self.file_path, "fasta"):
            cleaned_seq = self.clean_sequence(str(record.seq))
            # Optionally: filter sequences to HA/NA genes if metadata is available.
            if cleaned_seq and len(cleaned_seq) >= 9:  # ensure at least 3 codons
                sequences.append(cleaned_seq)
        return sequences

    def clean_sequence(self, sequence):
        # Remove invalid characters (e.g. N, gaps) and trim to a multiple of 3
        valid_bases = set(['A', 'T', 'C', 'G'])
        cleaned = ''.join([base for base in sequence if base in valid_bases])
        if len(cleaned) < 9:
            return None
        remainder = len(cleaned) % 3
        if remainder != 0:
            cleaned = cleaned[:-remainder]  # trim extra bases
        return cleaned
    
    def sequence_to_graph(self, sequence):
        # Split the cleaned sequence into codons
        codon_list = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        # Map each codon to its index; default to 0 if not found (should rarely happen)
        codon_indices = [self.codon_to_index.get(codon, 0) for codon in codon_list]
        
        # Create one-hot encoded features for each codon (feature dimension = 64)
        x = torch.nn.functional.one_hot(torch.tensor(codon_indices), num_classes=64).float()
        
        # Create sequential bidirectional edges between codon nodes
        num_nodes = len(codon_list)
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # TODO: Add extra edges for functionally related codons using receptor-binding/antigenic site annotations
        
        return Data(x=x, edge_index=edge_index)
    
    def process(self):
        sequences = self.load_sequences()
        graphs = [self.sequence_to_graph(seq) for seq in sequences]
        return graphs

# Load and process data from the influenza HA/NA FASTA file.
file_path = "BioInformatics/Sequences.fasta"  # update path as needed
processor = InfluenzaSequenceProcessor(file_path)
graph_data = processor.process()
print(f"Processed {len(graph_data)} influenza sequences into graph structures.")

### THE AI PART: Mutation Prediction using GNN and Contrastive Learning ###
# The model now includes two heads:
# 1. A mutation probability head (binary classification per codon)
# 2. A codon change prediction head (64-class classification per codon)
class MutationGNN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=16, num_codons=64):
        super(MutationGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Head for predicting mutation probability (using a sigmoid activation later)
        self.mutation_head = nn.Linear(hidden_dim, 1)
        # Head for predicting the codon change (64 possible classes)
        self.codon_change_head = nn.Linear(hidden_dim, num_codons)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # Mutation probability output (per codon node)
        mutation_prob = torch.sigmoid(self.mutation_head(x)).squeeze()  # shape: [num_nodes]
        # Codon change prediction logits (per codon node)
        codon_logits = self.codon_change_head(x)  # shape: [num_nodes, 64]
        return mutation_prob, codon_logits, x  # also return embeddings x for potential contrastive loss

# Placeholder for contrastive loss on the codon embeddings.
def contrastive_loss(embeddings):
    # In practice, this loss clusters similar mutation-prone regions.
    # Here we return zero as a placeholder.
    return torch.tensor(0.0)

# Initialize the model, loss functions, and optimizer.
model = MutationGNN(input_dim=64, hidden_dim=16, num_codons=64)
criterion_codon = nn.CrossEntropyLoss()   # For codon change prediction
criterion_mutation = nn.BCELoss()           # For binary mutation probability prediction
optimizer = optim.Adam(model.parameters(), lr=0.01)

# For demonstration, create random target labels for the first graph.
# For mutation probability: binary labels (0 = low mutation risk, 1 = high)
target_mutation = torch.randint(0, 2, (graph_data[0].x.size(0),)).float()
# For codon change: random codon indices (0 to 63)
target_codon = torch.randint(0, 64, (graph_data[0].x.size(0),))

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    mutation_prob, codon_logits, embeddings = model(graph_data[0])
    loss_mutation = criterion_mutation(mutation_prob, target_mutation)
    loss_codon = criterion_codon(codon_logits, target_codon)
    loss_contrastive = contrastive_loss(embeddings)
    
    total_loss = loss_mutation + loss_codon + loss_contrastive
    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

# Get predictions for the first graph.
model.eval()
with torch.no_grad():
    mutation_prob, codon_logits, _ = model(graph_data[0])
    
# Apply a threshold to mutation probability (0.5 as a simple cutoff).
predicted_mutation = (mutation_prob > 0.5).int()  # binary prediction per node
# For codon change, choose the class with highest logit.
predicted_codon_indices = torch.argmax(codon_logits, dim=1)
predicted_codons = [processor.index_to_codon[idx.item()] for idx in predicted_codon_indices]

print("Predicted Mutation Probabilities (binary):", predicted_mutation.tolist())
print("Predicted Codon Changes:", predicted_codons)
