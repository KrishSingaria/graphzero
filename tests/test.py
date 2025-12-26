import graphzero as gz
import torch
import numpy as np
import time

print("CSV to .gl conversion test")
gz.convert_csv_to_gl("dummy.csv","dummy.gl",directed = True)

print("Loading Graph...")
g = gz.Graph("graph-100T.gl") # graph

# 1. Create a batch of start nodes (e.g., 10,000 nodes)
start_nodes = list(range(10000)) # Simple list of 0..9999

print(f"Sampling {len(start_nodes)} walks of length 10...")

print(f"node 0: deg {g.get_degree(0)}")

# 2. Call C++ (The Heavy Lifting)
t0 = time.time()
walks_np = g.batch_random_walk_uniform(start_nodes,10)
t1 = time.time()

print(f"C++ Time: {t1 - t0:.4f} seconds")

nodes = [0, 1, 2]


# 1. Standard (Defaults used: p=1, q=1)
w1 = g.batch_random_walk(nodes, 10)

# 2. Positional Arguments (p=2.0, q=0.5)
w2 = g.batch_random_walk(nodes, 10, 2.0, 0.5)

# 3. Named Arguments (Pythonic!)
w3 = g.batch_random_walk(nodes, 10, q=0.5, p=2.0)

print("All modes working!")


# 3. Convert to PyTorch (Zero Copy)
# .as_tensor() or .from_numpy() are both zero-copy for compatible types
walks_tensor = torch.from_numpy(walks_np)

print("\n--- PyTorch Tensor ---")
print(f"Shape: {walks_tensor.shape}")
print(f"Dtype: {walks_tensor.dtype}")
print(f"Device: {walks_tensor.device}")
print(walks_tensor[0:5]) # Print first 5 walks

# 4. Verify Memory
# Modifying the tensor should crash if memory was freed (safety check)
walks_tensor[0][0] = 99999
print("Modification successful (Memory is alive).")
# time.sleep(100) # for inferencing