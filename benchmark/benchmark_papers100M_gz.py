import graphzero as gz
import time
import psutil
import os
import numpy as np

# --- CONFIGURATION ---
GRAPH_PATH = "papers100M.gl"
NUM_WALKS = 1_000_000   # Number of starting nodes
WALK_LENGTH = 10      # Steps per walk
IS_DIRECTED = True    # Set based on how you converted it

process = psutil.Process(os.getpid())

def get_ram_usage():
    """Returns current RAM usage in GB"""
    return process.memory_info().rss / (1024 ** 3)

def print_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

print_header("GRAPHZERO ENGINE BENCHMARK (111M Nodes)")
print(f"System: Windows x64 | PID: {os.getpid()}")
print(f"Initial RAM: {get_ram_usage():.4f} GB")

# ---------------------------------------------------------
# TEST 1: LOADING (The "Zero-Copy" Test)
# ---------------------------------------------------------
print_header("TEST 1: Instant Loading")

t0 = time.time()
try:
    g = gz.Graph(GRAPH_PATH)
except Exception as e:
    print(f"❌ Error loading graph: {e}")
    exit(1)
t_load = time.time() - t0

print(f"✅ Graph Loaded successfully!")
print(f"⏱️  Time:       {t_load:.6f} seconds (Target: < 0.1s)")
print(f"💾 RAM Usage:  {get_ram_usage():.4f} GB (Should be near zero change)")

# ---------------------------------------------------------
# TEST 2: WARMUP (Fighting the Windows Cold Start)
# ---------------------------------------------------------
print_header("TEST 2: Cache Warmup")
print("Doing a small run to wake up the Disk I/O...")

start_nodes = np.random.randint(0, 100000, 1000).astype(np.uint64).tolist()
_ = g.batch_random_walk_uniform(start_nodes, WALK_LENGTH)

print("✅ Warmup Complete.")

# ---------------------------------------------------------
# TEST 3: THROUGHPUT (The Real Power)
# ---------------------------------------------------------
print_header(f"TEST 3: High-Throughput Random Walks")
print(f"Configuration: {NUM_WALKS:,} walkers x {WALK_LENGTH} steps")
print("Running...")

# Generate random start nodes across the WHOLE graph
# (Using high numbers proves we aren't just caching the first 1MB)
max_node_id = g.num_nodes # nodes for Papers100M
start_nodes = np.random.randint(0, max_node_id, NUM_WALKS).astype(np.uint64).tolist()

t0 = time.time()
# The heavy lifting C++ call
walks = g.batch_random_walk_uniform(start_nodes, WALK_LENGTH)
t_bench = time.time() - t0

# ---------------------------------------------------------
# RESULTS ANALYSIS
# ---------------------------------------------------------
total_steps = NUM_WALKS * WALK_LENGTH
throughput = total_steps / t_bench

print_header("🏆 BENCHMARK RESULTS")
print(f"⏱️ Total Time:\t{t_bench:.4f} s")
print(f"🚀 Throughput:\t{throughput:,.0f} steps/sec")
print(f"💾 Peak RAM:\t{get_ram_usage():.4f} GB")
print("-" * 50)

if throughput > 1_000_000:
    print("🌟 RATING: EXCELLENT (Million+ steps/sec on Laptop)")
elif throughput > 100_000:
    print("👍 RATING: GOOD (Consumer Hardware Standard)")
else:
    print("⚠️ RATING: SLOW (Check Disk I/O or Debug Mode)")

print("-" * 50)
print("GraphZero v0.1 - Ready for Deep Learning.")