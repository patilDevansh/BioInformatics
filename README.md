# Influenza Mutation Prediction using Codon-based Graph Neural Networks (GNNs)

## 📌 Overview

This project implements a pipeline to analyze **influenza virus sequences** and predict potential mutations using a **Codon-based Graph Neural Network (GNN)**. The pipeline:

- Reads metadata from a CSV file and sequences from a FASTA file.
- Normalizes and filters metadata based on available sequences.
- Parses collection dates and selects a baseline sequence (earliest date).
- Performs **pairwise sequence alignment** (demo).
- Converts nucleotide sequences into **codon-based graphs** (nodes are one-hot encoded codons with sequential edges).
- Trains a **Graph Convolutional Network (GCN)** to predict mutation probabilities (dummy labels in this demo).

🚀 This is a **demo pipeline**. In a production scenario, you may refine the alignment, label generation, and training processes.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/Influenza-GNN.git
cd Influenza-GNN
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### Required Packages:
biopython (for sequence handling and alignment)
torch and torch_geometric (for GNN implementation)
pandas (for metadata processing)
If torch_geometric fails to install, refer to: PyG Installation Guide

## 🚀 Usage
### 1️⃣ Prepare Your Input Data:
FASTA File (sequences.fasta): Contains virus sequences.
Metadata CSV (metadata.csv): Should include an "Accession" column and "Collection_Date".
### 2️⃣ Run the Pipeline:
```bash
python main.py
```
### 3️⃣ Pipeline Steps:
✅ Load & normalize sequences
✅ Filter metadata based on available sequences
✅ Parse collection dates & select baseline sequence
✅ Perform pairwise sequence alignment (demo)
✅ Convert sequences to codon-based graphs
✅ Train a simple two-layer GNN model

### 4️⃣ Output:
Trained GNN Model (Model.pt): Can be used for inference.
Printed Logs: Displays alignment results, training progress, and mutation probabilities.


## ⚙️ Configuration
- Update file paths inside main.py:
```python
fasta_file = "sequences.fasta"
metadata_file = "metadata.csv"
```
- Modify GNN hyperparameters:
```python
model = MutationGCN(in_channels=64, hidden_channels=32, num_classes=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
```
- Define mutation labels (dummy labels used in this demo):
```python
graph.y = torch.zeros((num_nodes, 1), dtype=torch.float)
```

## 🛠️ Model Loading and Inference
To load a pre-trained model (Model.pt):
```python
from main import load_model
model = load_model("Model.pt")
```

## 🎯 Contributing
We welcome contributions! To contribute:

1. Fork the repository and create a new branch:
```bash
git checkout -b feature-name
```
2. Make changes & commit:
```bash
git commit -m "Added feature X"
```
3. Push to GitHub & create a pull request.

## 🐞 Reporting Issues
- If you encounter any bugs, open an issue with a detailed description.

## 📜 License
This project is licensed under the MIT License.

## 📧 Contact
For questions, reach out via:

📧 Email: duashmita@gmail.com
🔗 GitHub Issues: Open an Issue
Happy coding! 🚀