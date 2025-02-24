#!/usr/bin/env python3
"""
Influenza Mutation Prediction using Codon-based Graph Neural Networks (GNNs)
---------------------------------------------------------------------------
This script demonstrates a pipeline that:
1. Reads metadata from a CSV file.
2. Reads influenza virus sequences from a FASTA file.
3. Normalizes and filters metadata to include only rows whose (base) accessions are found in the FASTA file.
4. Parses collection dates (handling partial dates).
5. Selects a baseline sequence (e.g., the earliest collection date).
6. (Demo) Performs pairwise alignment to compare the baseline with another sequence.
7. Converts a nucleotide sequence into a codon‐based graph (nodes are one‐hot encoded codons,
   and edges connect sequential codon nodes).
8. Trains a simple two‐layer GCN to predict mutation probabilities (dummy labels in this demo).

Note: This is a simplified demo pipeline. In a production scenario, you would
extend the pairwise alignment, label generation, and training routines.
"""

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

##############################
# Step 1: Load and Normalize FASTA Sequences
##############################
def load_fasta(fasta_file):
    """
    Load FASTA records and return a dictionary mapping normalized accession to record.
    We remove version numbers by splitting on '.'.
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    fasta_dict = {}
    for record in records:
        # Extract base accession by splitting on dot, e.g., "NC_026431.1" -> "NC_026431"
        base_accession = record.id.split('.')[0]
        fasta_dict[base_accession] = record
    return fasta_dict

##############################
# Step 2: Load Metadata CSV
##############################
def load_metadata(metadata_file):
    """
    Load metadata CSV into a DataFrame.
    """
    metadata = pd.read_csv(metadata_file, low_memory=False)
    return metadata

##############################
# Step 3: Normalize & Filter Metadata
##############################
def filter_metadata(metadata, fasta_dict):
    """
    Normalize metadata accessions (by splitting on '.') and filter to include only
    rows where the base accession exists in the FASTA file.
    """
    # Create a new column with normalized (base) accession codes.
    metadata["Accession_base"] = metadata["Accession"].astype(str).str.split('.').str[0]
    # Filter metadata rows whose normalized accession is in our FASTA dict
    filtered = metadata[metadata["Accession_base"].isin(fasta_dict.keys())].copy()
    if filtered.empty:
        raise ValueError("No metadata rows have matching accessions in the FASTA file.")
    return filtered

##############################
# Step 4: Parse Collection Dates
##############################
def parse_dates(metadata):
    """
    Parse the 'Collection_Date' column into datetime objects.
    Unparseable dates become NaT and are dropped.
    """
    metadata["ParsedDate"] = pd.to_datetime(metadata["Collection_Date"], errors="coerce")
    metadata = metadata.dropna(subset=["ParsedDate"])
    return metadata

##############################
# Step 5: Select Baseline Sequence
##############################
def select_baseline(metadata):
    """
    Select the baseline sequence from metadata (e.g., the one with the earliest parsed date).
    Returns the base accession and its row.
    """
    baseline_row = metadata.loc[metadata["ParsedDate"].idxmin()]
    baseline_accession = baseline_row["Accession_base"]
    return baseline_accession, baseline_row

##############################
# Step 6: Pairwise Alignment (Demo)
##############################
def align_sequences(seq1, seq2):
    """
    Perform a global pairwise alignment between two sequences using Bio.Align.PairwiseAligner.
    Returns the best alignment (for demonstration purposes).
    """
    aligner = PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    return best_alignment

##############################
# Step 7: Build a Codon-Based Graph
##############################
def sequence_to_graph(sequence):
    """
    Convert a nucleotide sequence into a codon graph.
    
    Steps:
      - Remove non-ATCG characters.
      - Trim sequence to a multiple of 3.
      - Split into codons.
      - One-hot encode each codon using a fixed ordering of 64 codons.
      - Create bidirectional edges connecting sequential codons.
    
    Returns a PyTorch Geometric Data object.
    """
    sequence = "".join([n for n in sequence.upper() if n in "ATCG"])
    remainder = len(sequence) % 3
    if remainder != 0:
        sequence = sequence[:-remainder]
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    
    # Fixed list of 64 codons
    codon_list = [a + b + c for a in "ATCG" for b in "ATCG" for c in "ATCG"]
    codon_to_idx = {codon: idx for idx, codon in enumerate(codon_list)}
    
    node_features = []
    for codon in codons:
        one_hot = [0] * 64
        idx = codon_to_idx.get(codon)
        if idx is not None:
            one_hot[idx] = 1
        node_features.append(one_hot)
    
    # Create sequential bidirectional edges
    edge_index = []
    for i in range(len(codons) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

##############################
# Step 8: Define the GNN Model
##############################
class MutationGCN(nn.Module):
    """
    A simple two-layer Graph Convolutional Network (GCN) that outputs a mutation probability per node.
    """
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(MutationGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.sigmoid(x)
        return x

##############################
# Main Pipeline
##############################
def main():
    fasta_file = "sequences.fasta"      # Update with your FASTA file path
    metadata_file = "metadata.csv"      # Update with your metadata CSV path
    
    # Load and normalize FASTA sequences
    fasta_dict = load_fasta(fasta_file)
    print(f"Loaded {len(fasta_dict)} sequences from FASTA (normalized accessions).")
    
    # Load metadata
    metadata = load_metadata(metadata_file)
    print(f"Loaded {len(metadata)} metadata rows.")
    
    # Normalize and filter metadata by matching FASTA accessions
    metadata = filter_metadata(metadata, fasta_dict)
    print(f"{len(metadata)} metadata rows remain after filtering by FASTA accessions.")
    
    # Parse collection dates
    metadata = parse_dates(metadata)
    print(f"{len(metadata)} metadata rows remain after parsing dates.")
    
    # Select baseline sequence (earliest date)
    baseline_accession, baseline_row = select_baseline(metadata)
    if baseline_accession not in fasta_dict:
        raise ValueError(f"Baseline accession {baseline_accession} not found in FASTA file.")
    print(f"Using baseline accession: {baseline_accession}")
    
    # Get the baseline sequence
    baseline_record = fasta_dict[baseline_accession]
    baseline_seq = str(baseline_record.seq)
    
    # For demonstration, process one other sequence (other than the baseline)
    for accession in metadata["Accession_base"].unique():
        if accession == baseline_accession:
            continue
        if accession not in fasta_dict:
            continue
        record = fasta_dict[accession]
        seq = str(record.seq)
        print(f"\nProcessing sequence: {accession}")
        
        # Pairwise alignment (demo)
        alignment = align_sequences(baseline_seq, seq)
        print("Performed pairwise alignment (demo).")
        
        # Convert sequence to codon-based graph
        graph = sequence_to_graph(seq)
        print(f"Constructed graph with {graph.num_nodes} nodes.")
        
        # For demo purposes, assign dummy mutation labels (all zeros)
        num_nodes = graph.num_nodes
        graph.y = torch.zeros((num_nodes, 1), dtype=torch.float)
        
        # Process only one sequence for this demo
        break
    
    # Create a dummy dataset and DataLoader
    dataset = [graph]
    loader = DataLoader(dataset, batch_size=1)
    
    # Define the GNN model
    model = MutationGCN(in_channels=64, hidden_channels=32, num_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # Dummy training loop
    model.train()
    print("\nStarting training loop...")
    for epoch in range(10):
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(),"Model.pt")
    
    print("\nDemo pipeline completed.")

if __name__ == "__main__":
    main()