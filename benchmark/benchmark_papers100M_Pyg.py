import time
import psutil
import os
import torch
from ogb.nodeproppred import PygNodePropPredDataset

print("==================================================")
print(" PYG BENCHMARK: The 'Control' Experiment")
print(" Warning: This script may freeze your laptop.")
print("==================================================")

process = psutil.Process(os.getpid())
def get_ram_usage():
    return process.memory_info().rss / (1024 ** 3)

print(f"Initial RAM: {get_ram_usage():.4f} GB")

# 1. Measure Loading Time
print("\n[Step 1] Attempting to load ogbn-papers100M with PyG...")
t0 = time.time()

try:
    # This tries to load the processed .pt file into RAM
    # If this line finishes in < 60 seconds, I will be shocked.
    dataset = PygNodePropPredDataset(name='ogbn-papers100M',root='dataset')
    data = dataset[0] # The actual graph object
    
    t_load = time.time() - t0
    print(f"✅ Loaded! (Miraculously)")
    print(f"⏱️ Load Time: {t_load:.4f} s")
    print(f"💾 RAM Usage: {get_ram_usage():.4f} GB")

except Exception as e:
    print(f"\n❌ CRASHED as expected: {e}")
    print(f"💾 RAM at Crash: {get_ram_usage():.4f} GB")
    exit(1)

# 2. Random Walk Benchmark (If we survived loading)
print("\n[Step 2] Attempting Random Walks (ClusterGCN style)...")
# PyG doesn't have a direct "random walk" sampler on CPU that is easy to invoke
# without a DataLoader, so we will just try to access the edge_index 
# to simulate 'touching' the memory.

try:
    t0 = time.time()
    # Simulate reading 1M random edges
    num_edges = data.edge_index.shape[1]
    indices = torch.randint(0, num_edges, (1_000_000,))
    
    # Force read
    subset = data.edge_index[:, indices] 
    
    t_bench = time.time() - t0
    print(f"✅ Access Test Complete in {t_bench:.4f} s")

except Exception as e:
    print(f"❌ Failed during access: {e}")