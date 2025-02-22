from Bio import SeqIO
import numpy as np

def load_fasta(file_path):
    return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 
               'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence])

if __name__ == "__main__":
    sequences = load_fasta("data/sequences.fasta")
    encoded_seq = one_hot_encode(sequences[0])
    print(f"First encoded sequence shape: {encoded_seq.shape}")
