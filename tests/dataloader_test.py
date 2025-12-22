import graphzero as gz
import torch
from torch.utils.data import Dataset, DataLoader
import time

# to change number of threads for OMP 
import os 
os.environ["OMP_NUM_THREADS"] = "2" # per process 2 threads only

# 1. Define a PyTorch Dataset Wrapper
class GraphDataset(Dataset):
    def __init__(self, graph_file, num_nodes):
        self.graph = gz.Graph(graph_file)
        self.nodes = list(range(num_nodes))

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        # The Worker Process calls this!
        # It needs its own copy of 'self.graph'
        node_id = self.nodes[idx]
        
        # Let's verify we can read neighbors
        deg = self.graph.get_degree(node_id)
        return node_id, deg
    
    def random_walk(self,start_nodes,walk_len):
        walks_np = self.graph.batch_random_walk_uniform(start_nodes,walk_len)
        return walks_np 

def run_test():
    print("--- Testing Multiprocessing ---")

    start_nodes = list(range(10000)) # Simple list of 0..9999
    # 2. Create Dataset
    # Assume 100000 items
    dataset = GraphDataset("/home/krish/graphzero/graph-100T.gl", num_nodes=100000)

    # 3. Create DataLoader with MULTIPLE WORKERS
    # num_workers=4 means 4 NEW processes are spawned.
    # They MUST pickle 'dataset' to start.
    loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)

    print("DataLoader created. Starting iteration...")
    
    start = time.time()
    for batch_idx, (nodes, degrees) in enumerate(loader):
        if batch_idx == 0:
            print(f"Batch 0 received! Nodes: {nodes[:5]}")
            print(f"Degrees: {degrees[:5]}")

    end = time.time()

    # 2. Call C++ (The Heavy Lifting)
    t0 = time.time()
    walks_np = dataset.random_walk(start_nodes,10)
    t1 = time.time()

    print(f"C++ Time: {t1 - t0:.4f} seconds")

    print(f"✅ Success! Iterated {len(dataset)} items in {end-start:.4f}s with 4 workers.")

if __name__ == "__main__":
    run_test()