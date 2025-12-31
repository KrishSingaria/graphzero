import numpy as np
import pandas as pd
import zipfile
import os
import time
from tqdm import tqdm

# --- CONFIG ---
# Path to your downloaded .npz
NPZ_PATH = "benchmark/data.npz" 
# Where to extract the raw binary temporary file
RAW_NPY_FILE = "benchmark/temp_edge_index.npy"
# Final output
OUTPUT_CSV = "benchmark/papers100M_edges.csv"

def extract_and_convert():
    t0 = time.time()
    
    # STEP 1: Extract the .npy file from the .npz archive
    # This turns "Compressed Data" (RAM heavy) into "Disk Data" (Storage heavy)
    if not os.path.exists(RAW_NPY_FILE):
        print(f"[1/3] Extracting edge_index from {NPZ_PATH}...")
        try:
            with zipfile.ZipFile(NPZ_PATH, 'r') as archive:
                # The file inside is usually named 'edge_index.npy'
                # Let's verify the name
                file_list = archive.namelist()
                target_file = next((f for f in file_list if 'edge_index' in f), None)
                
                if not target_file:
                    raise ValueError(f"Could not find edge_index in {file_list}")
                
                print(f"   Found internal file: {target_file}")
                print("   Unzipping to disk (this will take a few minutes)...")
                
                # Extract specific file
                with archive.open(target_file) as source, open(RAW_NPY_FILE, "wb") as target:
                    # Stream copy to avoid loading into RAM
                    while True:
                        chunk = source.read(1024 * 1024 * 64) # 64MB chunks
                        if not chunk: break
                        target.write(chunk)
                        
        except zipfile.BadZipFile:
            print("\n❌ CRITICAL ERROR: The data.npz file is corrupted.")
            print("You must delete it and re-download using the 'Option 2 (cURL)' method.")
            return
    else:
        print(f"[1/3] Found extracted {RAW_NPY_FILE}, skipping extraction.")

    # STEP 2: Memory Map the Raw File
    print(f"[2/3] Memory mapping raw file...")
    # Now this works because it's a real file on disk!
    edge_index = np.load(RAW_NPY_FILE, mmap_mode='r')
    
    # Shape is usually (2, N) -> 2 rows (src, dst), N columns
    num_edges = edge_index.shape[1]
    print(f"   Shape: {edge_index.shape}")
    print(f"   Total Edges: {num_edges:,}")

    # STEP 3: Write to CSV in chunks
    print(f"[3/3] Converting to {OUTPUT_CSV}...")
    chunk_size = 5_000_000 # 5M edges per chunk
    
    with open(OUTPUT_CSV, "w") as f:
        # We assume OGB format is Row 0 = Source, Row 1 = Dest
        # We iterate through columns
        for i in tqdm(range(0, num_edges, chunk_size), unit="chunk"):
            end = min(i + chunk_size, num_edges)
            
            # This read is fast because it pulls from disk only what is needed
            # We transpose .T to get [[src, dst], [src, dst]...]
            chunk = edge_index[:, i:end].T 
            
            df = pd.DataFrame(chunk)
            df.to_csv(f, header=False, index=False)

    total_time = time.time() - t0
    print(f"\n✅ Success! CSV created in {total_time:.2f}s")
    print(f"You can now delete {RAW_NPY_FILE} to free up 24GB.")

if __name__ == "__main__":
    extract_and_convert()