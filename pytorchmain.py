"""
AI-Driven Influenza Mutation Prediction

This script implements a codon-based Graph Neural Network (GNN) pipeline to predict
mutation probabilities in influenza virus genes (HA/NA). The approach is based on the
"AI-Driven Influenza Mutation Prediction" project summary, which focuses on:

1) Preprocessing FASTA sequences into a codon-based graph representation.
2) Learning mutation-prone codon sites via GNN.
3) (Optional) Using contrastive learning to cluster similar mutation-prone embeddings.
4) Outputting two predictions per codon node:
   - Binary mutation probability (how likely a codon might mutate).
   - The predicted new codon (which codon it might change into).

Key Features:
------------
- Handles Influenza A, B, or C sequences.
- Optionally filters for HA/NA segments/subtypes only (e.g., H3N2, H1N1, B).
- Cleans sequences, ensuring only valid nucleotides (A,T,C,G) and a multiple of 3 length.
- Builds a bidirectional sequential adjacency for codon nodes.
- GNN with two heads (BCELoss for mutation, CrossEntropy for codon changes).
- Placeholder for contrastive loss to enhance embedding quality.
- Summarizes predictions in a user-friendly format.

Dependencies:
-------------
- Python 3.x
- Biopython (for SeqIO)
- PyTorch and PyTorch Geometric

Usage:
------
1) Place your influenza FASTA file(s) in a directory.
2) Update `file_path` with the path to your dataset.
3) Adjust settings (require_ha_na_only, min_codon_length, etc.) as needed.
4) Run the script (e.g., `python influenza_mutation_pipeline.py`).
"""

import re
import itertools
from Bio import SeqIO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


############################################
# 1. DATA PREPROCESSING & GRAPH CONVERSION #
############################################

class InfluenzaMutationProcessor:
    """
    Processor for influenza (A/B/C) FASTA sequences, focusing on
    (H#N#) subtypes or "Influenza B virus" by default (HA/NA).
    Converts sequences into codon-based graphs for GNN training.
    """
    def __init__(self, file_path, require_ha_na_only=True, min_codon_length=50):
        """
        :param file_path: Path to the FASTA file.
        :param require_ha_na_only: If True, filter for HA/NA (H#N#) or 'Influenza B virus' etc.
        :param min_codon_length: Min # of codons needed. E.g., 50 → 150 nucleotides.
        """
        self.file_path = file_path
        self.require_ha_na_only = require_ha_na_only
        self.min_codon_length = min_codon_length

        # Create 64 possible codons and index mappings
        nucleotides = ['A', 'T', 'C', 'G']
        self.codons = [''.join(c) for c in itertools.product(nucleotides, repeat=3)]
        self.codon_to_index = {codon: i for i, codon in enumerate(self.codons)}
        self.index_to_codon = {i: codon for codon, i in self.codon_to_index.items()}

    def is_ha_na_segment(self, header):
        """
        Rough check to see if the header contains (H#N#) or mentions 'Influenza B virus'.
        Adjust logic if you want to include Influenza C or skip type checks entirely.
        """
        match_subtype = re.search(r'\(H\d+N\d+\)', header)  # e.g. (H3N2)
        has_b = "influenza b virus" in header.lower()
        has_c = "influenza c virus" in header.lower()

        if self.require_ha_na_only:
            # Keep if there's an H#N# pattern or it's labeled Influenza B virus
            return bool(match_subtype) or has_b or has_c
        else:
            # Accept everything if not strictly requiring HA/NA
            return True

    def clean_sequence(self, seq_str):
        """
        Removes invalid characters (e.g. N, R, Y, -).
        Truncates to a multiple of 3. Returns None if too short.
        """
        valid_bases = {'A', 'T', 'C', 'G'}
        cleaned = ''.join([base for base in seq_str.upper() if base in valid_bases])
        remainder = len(cleaned) % 3
        if remainder != 0:
            cleaned = cleaned[:-remainder]
        if len(cleaned) < (3 * self.min_codon_length):
            return None
        return cleaned

    def sequence_to_graph(self, sequence):
        """
        Converts a DNA sequence into a PyTorch Geometric graph object.
        Each codon is a node (64-dimensional one-hot).
        Edges represent bidirectional adjacency between consecutive codons.
        """
        # Split into codons
        codon_list = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        # Map each codon to an index
        codon_indices = [self.codon_to_index.get(codon, 0) for codon in codon_list]

        # Node features: one-hot for each codon
        x = F.one_hot(torch.tensor(codon_indices), num_classes=64).float()

        # Edges: connect consecutive codons in both directions
        edges = []
        for i in range(len(codon_list) - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def process(self):
        """
        Parses the FASTA, filters sequences, and returns a list of Data() graphs.
        """
        graphs = []
        for record in SeqIO.parse(self.file_path, "fasta"):
            if not self.is_ha_na_segment(record.description):
                continue
            seq_str = str(record.seq)
            cleaned_seq = self.clean_sequence(seq_str)
            if cleaned_seq:
                graph = self.sequence_to_graph(cleaned_seq)
                graphs.append(graph)
        print(f"Processed {len(graphs)} sequences from {self.file_path}.")
        return graphs


#####################################################
# 2. GNN MODEL: BINARY MUTATION + 64-CODON PREDICTION
#####################################################

class InfluenzaMutationGNN(nn.Module):
    """
    A GNN model with:
    - 2 GCNConv layers
    - Head #1: Binary mutation probability
    - Head #2: Codon-class (64) prediction
    """
    def __init__(self, hidden_dim=32, num_codons=64):
        super().__init__()
        self.conv1 = GCNConv(num_codons, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Head 1: Probability of mutation (binary)
        self.mutation_head = nn.Linear(hidden_dim, 1)
        # Head 2: Which codon it might mutate into (64-class)
        self.codon_change_head = nn.Linear(hidden_dim, num_codons)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        mutation_prob = torch.sigmoid(self.mutation_head(x)).squeeze(-1)  # [num_nodes]
        codon_logits = self.codon_change_head(x)                           # [num_nodes, 64]
        return mutation_prob, codon_logits, x


###########################################
# 3. CONTRASTIVE LOSS (PLACEHOLDER SAMPLE)
###########################################

def contrastive_loss_placeholder(embeddings):
    """
    Placeholder for an optional contrastive objective, e.g., to cluster
    codons with similar mutation propensities.
    """
    return torch.tensor(0.0, requires_grad=True)


############################################################
# 4. TRAINING EXAMPLE + HELPER FOR USER-FRIENDLY PREDICTIONS
############################################################

def train_influenza_gnn(graphs, epochs=50, lr=1e-3):
    """
    Simple example training loop. Demonstrates how you'd combine:
      - BCELoss for mutation probability
      - CrossEntropyLoss for codon changes
      - Contrastive placeholder
    Using the first graph only; extend as needed.
    """
    if not graphs:
        print("No graphs to train on. Exiting.")
        return None

    model = InfluenzaMutationGNN(hidden_dim=32, num_codons=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion_mutation = nn.BCELoss()
    criterion_codon = nn.CrossEntropyLoss()

    # For demo, we just train on the first graph
    data_sample = graphs[0]
    # Random placeholders for labels
    mutation_labels = torch.randint(0, 2, (data_sample.x.size(0),)).float()
    codon_labels = torch.randint(0, 64, (data_sample.x.size(0),))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        mutation_prob, codon_logits, embeddings = model(data_sample)
        loss_mutation = criterion_mutation(mutation_prob, mutation_labels)
        loss_codon = criterion_codon(codon_logits, codon_labels)
        loss_contrastive = contrastive_loss_placeholder(embeddings)

        total_loss = loss_mutation + loss_codon + loss_contrastive
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | "
                  f"MutLoss={loss_mutation:.4f}, "
                  f"CodonLoss={loss_codon:.4f}, "
                  f"Contrastive={loss_contrastive:.4f}, "
                  f"Total={total_loss:.4f}")

    return model


def summarize_predictions(graph_data, model, processor, threshold=0.5):
    """
    Returns a list of dicts with user-friendly summaries of the model's predictions:
      - Graph index
      - Codon index
      - Original codon
      - Mutation probability
      - Mutated? (bool, threshold-based)
      - Predicted new codon (if mutated)
    """
    model.eval()
    summaries = []

    with torch.no_grad():
        for graph_idx, data in enumerate(graph_data):
            mutation_prob, codon_logits, _ = model(data)

            # Original codon from one-hot
            original_indices = torch.argmax(data.x, dim=1)
            original_codons = [processor.index_to_codon[idx.item()] for idx in original_indices]

            # Binary mutation decision
            mutated_bool = (mutation_prob > threshold).bool()

            # Most likely codon for mutated sites
            predicted_new_codons_idx = torch.argmax(codon_logits, dim=1)
            predicted_new_codons = [processor.index_to_codon[idx.item()] 
                                    for idx in predicted_new_codons_idx]

            for i, orig_codon in enumerate(original_codons):
                record = {
                    "graph_id": graph_idx,
                    "codon_index": i,
                    "original_codon": orig_codon,
                    "mutation_probability": float(mutation_prob[i].item()),
                    "is_mutated": bool(mutated_bool[i].item()),
                    "predicted_new_codon": (predicted_new_codons[i]
                                            if mutated_bool[i]
                                            else "No significant mutation")
                }
                summaries.append(record)

    return summaries


####################
# USAGE EXAMPLE
####################
if __name__ == "__main__":
    # Example usage:
    # 1. Prepare data
    file_path = "BioInformatics/sequences_20250223_3350210.fasta"  # Replace with your own
    processor = InfluenzaMutationProcessor(file_path, require_ha_na_only=True, min_codon_length=50)
    graphs = processor.process()

    # 2. Train a model (demo with random labels)
    model = train_influenza_gnn(graphs, epochs=20, lr=1e-3)

    # 3. Summarize predictions
    if model is not None:
        results = summarize_predictions(graphs, model, processor, threshold=0.5)
        for row in results[:10]:  # Show first 10 predictions
            print(row)
