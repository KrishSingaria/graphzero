import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import graphzero as gz
import numpy as np
from torch.utils.data import DataLoader, Dataset

# --- 1. CONFIGURATION & DATA GENERATION ---
NUM_NODES = 50_000
NUM_EDGES = 200_000
FEATURE_DIM = 32
NUM_CLASSES = 10
FANOUT_K = 5
BATCH_SIZE = 1024

def generate_synthetic_data():
    """Generates synthetic CSVs if they don't exist yet."""
    if os.path.exists("dataset/edges.csv"): return
    os.makedirs("dataset", exist_ok=True)
    
    print("Generating synthetic dataset (CSVs)...")
    # Edges
    src = np.random.randint(0, NUM_NODES, NUM_EDGES)
    dst = np.random.randint(0, NUM_NODES, NUM_EDGES)
    with open("dataset/edges.csv", "w") as f:
        for s, d in zip(src, dst): f.write(f"{s},{d}\n")
            
    # Features (Float32)
    with open("dataset/features.csv", "w") as f:
        for i in range(NUM_NODES):
            feats = ",".join([f"{np.random.randn():.4f}" for _ in range(FEATURE_DIM)])
            f.write(f"{i},{feats}\n")
            
    # Labels (Int64)
    with open("dataset/labels.csv", "w") as f:
        for i in range(NUM_NODES):
            f.write(f"{i},{np.random.randint(0, NUM_CLASSES)}\n")

generate_synthetic_data()

# --- 2. GRAPHZERO CONVERSION (CSV -> Binary) ---
print("\nConverting CSVs to GraphZero formats...")
if not os.path.exists("graph.gl"):
    gz.convert_csv_to_gl("dataset/edges.csv", "graph.gl", directed=True)
if not os.path.exists("features.gd"):
    gz.convert_csv_to_gd("dataset/features.csv", "features.gd", dtype=gz.DataType.FLOAT32)
if not os.path.exists("labels.gd"):
    gz.convert_csv_to_gd("dataset/labels.csv", "labels.gd", dtype=gz.DataType.INT64)

# --- 3. ZERO-COPY MOUNTING ---
print("\nMounting Zero-Copy Engines...")
g = gz.Graph("graph.gl")
fs_feats = gz.FeatureStore("features.gd")
fs_labels = gz.FeatureStore("labels.gd")

print(f"Graph Mounted. Nodes: {g.num_nodes:,} | Edges: {g.num_edges:,}")

# Instantly map SSD data to PyTorch (RAM used: 0 Bytes)
X = torch.from_numpy(fs_feats.get_tensor())
Y = torch.from_numpy(fs_labels.get_tensor()).squeeze() # Squeeze (N, 1) to (N,)

print(f"Feature Tensor: {X.shape} ({X.dtype})")
print(f"Label Tensor:   {Y.shape} ({Y.dtype})")


# --- 4. PYTORCH DATALOADER & COLLATOR ---
class TargetNodeDataset(Dataset):
    def __len__(self): return NUM_NODES
    def __getitem__(self, idx): return idx

def collate_neighborhoods(batch_nodes):
    targets = [int(n) for n in batch_nodes]
    # Fast C++ neighbor sampling (Releases GIL)
    neighbors = g.batch_random_fanout(targets, FANOUT_K)
    return torch.tensor(targets, dtype=torch.long), torch.tensor(neighbors, dtype=torch.long)

loader = DataLoader(
    TargetNodeDataset(), batch_size=BATCH_SIZE, 
    collate_fn=collate_neighborhoods, shuffle=True
)


# --- 5. THE GRAPHSAGE MODEL ---
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, target_nodes, neighbor_nodes):
        # OS Page Fault Magic: 
        # PyTorch indexes the mapped SSD tensor, pulling only required 4KB blocks
        target_feats = X[target_nodes]
        neighbor_feats = X[neighbor_nodes] 
        
        # Mean pool the neighbors' features
        agg_neighbor_feats = neighbor_feats.mean(dim=1) 
        
        # Concat [Target || Aggregated] and pass through NN
        combined = torch.cat([target_feats, agg_neighbor_feats], dim=1)
        return self.classifier(self.relu(self.fc(combined)))


# --- 6. TRAINING LOOP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(FEATURE_DIM, 64, NUM_CLASSES).to(device)
X, Y = X.to(device), Y.to(device) # Move memory mappings to GPU buffer
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("\n🚀 Starting GraphSAGE Training...")
t0 = time.time()

for epoch in range(3):
    total_loss = 0
    for targets, neighbors in loader:
        targets, neighbors = targets.to(device), neighbors.to(device)
        
        optimizer.zero_grad()
        logits = model(targets, neighbors) 
        loss = criterion(logits, Y[targets]) # Fetch actual labels from .gd mapping
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/3 | Avg Loss: {total_loss/len(loader):.4f}")

print(f"✅ Training Complete in {time.time() - t0:.2f} seconds.")