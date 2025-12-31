import torch
import torch.nn as nn
import torch.optim as optim
import graphzero as gz
import numpy as np
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
GRAPH_PATH = "papers100M.gl" # The beast
EMBEDDING_DIM = 128
WALK_LENGTH = 20
WALKS_PER_EPOCH = 100_000 # Number of starts per batch
BATCH_SIZE = 1024
EPOCHS = 5

print(f"Initializing GraphZero Engine on {GRAPH_PATH}...")
g = gz.Graph(GRAPH_PATH)
print(f"   Nodes: {g.num_nodes:,} | Edges: {g.num_edges:,}")

# --- 1. THE DATASET (Powered by GraphZero) ---
class GraphZeroWalkDataset(Dataset):
    """
    Generates random walks on-the-fly using C++ engine.
    """
    def __init__(self, graph_engine, num_walks, walk_len):
        self.g = graph_engine
        self.num_walks = num_walks
        self.walk_len = walk_len
        
    def __len__(self):
        # In a real scenario, this might be num_nodes
        # For this demo, we define an arbitrary epoch size
        return self.num_walks

    def __getitem__(self, idx):
        # We don't generate single walks (too slow).
        # We let the DataLoader batch them, then call C++ in the collate_fn.
        # So we just return a random start node here.
        return np.random.randint(0, self.g.num_nodes)

# --- 2. CUSTOM COLLATE FUNCTION (The Secret Sauce) ---
def collate_walks(batch_start_nodes):
    """
    This is where the magic happens.
    Instead of Python looping, we give the whole batch of start nodes 
    to C++ and get back the massive walk matrix instantly.
    """
    # 1. Convert batch to list of uint64 for C++
    start_nodes = [int(x) for x in batch_start_nodes]
    
    # 2. Call C++ Engine (Releases GIL, runs OpenMP)
    # Result is a flat list: [walk1_step1, walk1_step2... walk2_step1...]
    flat_walks = g.batch_random_walk_uniform(start_nodes, WALK_LENGTH)
    
    # 3. Reshape for PyTorch (Batch Size, Walk Length)
    walks_tensor = torch.tensor(flat_walks, dtype=torch.long)
    walks_tensor = walks_tensor.view(len(start_nodes), WALK_LENGTH)
    
    return walks_tensor

# --- CONFIGURATION ADJUSTMENT ---
# We map 204M nodes -> 1M unique embeddings to save RAM
HASH_SIZE = 1_000_000  
# RAM Usage: 1M * 128 * 4 bytes = ~512 MB (Very safe)

# --- 3. THE MODEL (Hashed Skip-Gram) ---
class Node2Vec(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        # INSTEAD OF: self.in_embed = nn.Embedding(num_nodes, embed_dim)
        # WE USE:
        self.in_embed = nn.Embedding(HASH_SIZE, embed_dim)
        self.out_embed = nn.Embedding(HASH_SIZE, embed_dim)
        
    def forward(self, target, context):
        # Hashing Trick: Map massive ID -> Small ID
        # In a real app, you'd use a better hash, but modulo is fine for a demo
        t_hashed = target % HASH_SIZE
        c_hashed = context % HASH_SIZE
        
        v_in = self.in_embed(t_hashed)
        v_out = self.out_embed(c_hashed)
        
        return torch.sum(v_in * v_out, dim=1)

# --- 4. TRAINING LOOP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Node2Vec(g.num_nodes, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# PyTorch DataLoader wraps our C++ engine
loader = DataLoader(
    GraphZeroWalkDataset(g, WALKS_PER_EPOCH, WALK_LENGTH), 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_walks, # <--- Connects PyTorch to GraphZero
    num_workers=0 # Windows needs 0, Linux can use more
)

print("\nStarting Training...")

for epoch in range(EPOCHS):
    total_loss = 0
    
    for batch_walks in loader:
        # batch_walks shape: [1024, 20]
        batch_walks = batch_walks.to(device)
        
        # Simple Positive Pair generation: (Current, Next)
        # Real implementations use sliding windows, simplified here for brevity
        target = batch_walks[:, :-1].flatten()
        context = batch_walks[:, 1:].flatten()
        
        optimizer.zero_grad()
        loss = -model(target, context).mean() # Dummy loss for demo
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(loader):.4f}")

print("✅ Training Complete.")