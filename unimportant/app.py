from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Define the GNN model
class MutationGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(MutationGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, data, threshold=0.5):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        probabilities = self.sigmoid(x)  # Output probabilities
        predictions = (probabilities > threshold).float()  # Apply threshold
        return predictions, probabilities

# Load the pre-trained model
def load_model(model_path="Model.pt"):
    model = MutationGCN(in_channels=64, hidden_channels=32, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Convert sequence to codon-based graph
def sequence_to_graph(sequence):
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

# Get mutation positions from predictions
def get_mutation_positions(predictions):
    mutation_positions = torch.nonzero(predictions.squeeze()).squeeze().tolist()
    if isinstance(mutation_positions, int):  # Handle single mutation case
        mutation_positions = [mutation_positions]
    return mutation_positions

# Load the model
model = load_model()

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    gene_sequence = data.get('sequence')

    if not gene_sequence:
        return jsonify({"error": "No gene sequence provided."}), 400

    try:
        print(f"Received gene sequence: {gene_sequence}")
        graph = sequence_to_graph(gene_sequence)
        model.eval()
        with torch.no_grad():
            predictions, probabilities = model(graph)  # Get binary predictions
            mutation_positions = get_mutation_positions(predictions)  # Get mutation positions

        print(f"Predicted mutations at positions: {mutation_positions}")
        return jsonify({"mutations": mutation_positions})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)