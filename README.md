# GraphZero

**High-Performance, Zero-Copy Graph Engine for Massive Datasets on Consumer Hardware.**

GraphZero is a C++ graph processing engine with lightweight Python bindings designed to solve the **"Memory Wall"** in Graph Neural Networks (GNNs). It allows you to load and sample **100 Million+ node graphs** (like `ogbn-papers100M`) on a standard 16GB RAM laptop—something standard libraries like PyTorch Geometric (PyG) or DGL cannot do.


## ⚡ The Problem

GNN datasets can be  massive. `ogbn-papers100M` contains **111 Million nodes** and **1.6 Billion edges**.

* **Standard approach (PyG/NetworkX):** Tries to load the entire graph structure into **RAM**.
* **The Result:** `MemoryError` (OOM) on consumer hardware. You need 64GB+ **RAM** servers just to *load* the data.

## 🛠️ The Solution:

![GraphZero Architecture](benchmark/images/graphzero.png)

GraphZero abandons the "Load-to-RAM" model. Instead, it uses a custom **Zero-Copy Architecture**:

* **Memory Mapping (`mmap`):** The graph stays on disk. The OS only loads the specific "hot" pages needed for computation into RAM.
* **Compressed CSR:** A custom binary format (`.gl`) that compresses raw edges by **~60%** (30GB CSV  13GB Binary).
* **Parallel Sampling:** OpenMP-accelerated random walks that saturate NVMe SSD throughput.

## 🏆 Benchmarks: GraphZero vs. PyTorch Geometric

**Task:** Load `ogbn-papers100M` (56GB Raw) and perform random walks.
**Hardware:** Windows Laptop (16GB RAM, NVMe SSD).

| Metric | GraphZero (v0.1) | PyTorch Geometric |
| --- | --- | --- |
| **Load Time** | **0.000000 s** ⚡ | **FAILED** (Crash) ❌ |
| **Peak RAM Usage** | **~5.1 GB** (OS Cache) | **>24.1 GB** (Required) |
| **Throughput** | **1,264,000 steps/s** | N/A |
| **Status** | ✅ **Success** | ❌ **OOM Error** |

### Proof of Performance

> *Left: GraphZero loading instantly and utilizing OS Page Cache. Right: PyG crashing with `Unable to allocate 24.1 GiB`.*

<p float="left ">
<img src="benchmark/images/gz_bench.png" width="45%" />
<img src="benchmark/images/py_crash.png" width="45%" />
</p>

---

## 📦 Installation

GraphZero is available on PyPI (Pre-Alpha):

```bash
pip install graphzero

```

*Requirements: Python 3.8+, C++17 Compiler (MSVC/GCC), OpenMP.*

---

## 🚀 Quick Start

### 1. Convert Your Data

GraphZero uses a high-efficiency binary format (`.gl`). Convert your generic CSV edges list once.

```python
import graphzero as gz

# Converts raw CSV (src, dst) to memory-mapped binary
# Handles 100M+ edges easily on minimal RAM
gz.convert_csv_to_gl(
    input_csv="dataset/edges.csv", 
    output_bin="graph.gl", 
    directed=True
)

```

### 2. High-Speed Sampling

Once converted, the graph is instantly accessible.

```python
import graphzero as gz
import numpy as np

# 1. Zero-Copy Load (Instant)
g = gz.Graph("graph.gl")

# 2. Define Start Nodes (e.g., 1000 random nodes)
start_nodes = np.random.randint(0, g.num_nodes, 1000).astype(np.uint64)

# 3. Parallel Random Walk (node2vec / DeepWalk style)
# Returns: List of walks (flat or list-of-lists)
walks = g.batch_random_walk_uniform(
    start_nodes=start_nodes, 
    walk_length=10
)

print(f"Generated {len(walks)} steps instantly.")

```


## ⚙️ Under the Hood

GraphZero is built for **Systems & GNN** enthusiasts.

* **Core:** C++20 with `nanobind` for Python bindings.
* **Parallelism:** Uses `#pragma omp` with thread-local RNGs to prevent false sharing and lock contention.
* **IO:** Direct `CreateFileMapping` (Windows) and `mmap` (Linux) calls with alignment optimization (4KB/2MB pages).


## 🗺️ Roadmap

* **v0.1 (Current):** Topology-only support. Uniform Random Walks.
* **v0.2:** Columnar Feature Store (mmap support for Node Features ).
* **v0.3:** Weighted Edges & SIMD (AVX2) Neighbor Intersection.
* **v0.4:** Dynamic Updates (LSM-Tree based mutable graphs).
* **v0.5:** Pinned Memory Allocator for faster CPU  GPU transfer.


## 📄 License

MIT License. Created by **Krish Singaria** (IIT Mandi).