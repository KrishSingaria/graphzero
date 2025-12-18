#ifndef CSRFILEGEN_H
#define CSRFILEGEN_H
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <random>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include "MemoryMap.hpp"

void generateBinary(std::vector<size_t>& nnzRow,std::vector<size_t>& colPtr,const char* pathFileName){
    int fd = open(pathFileName,O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if(fd == -1){
        throw std::runtime_error("File open failed while generating binaray");
    }

    GraphHeader graphData; 
    graphData.sizeofnnzRow = nnzRow.size() * sizeof(size_t);
    graphData.sizeofcolPtr = colPtr.size() * sizeof(size_t);

    graphData.offset_nnz = sizeof(GraphHeader);
    graphData.offset_col = sizeof(GraphHeader) + graphData.sizeofnnzRow;

    if(write(fd,&graphData,sizeof(GraphHeader)) == -1){
        close(fd);
        throw std::runtime_error("could not write Graph Header");  
    }
    // Write everything at once for nnzRow
    if (write(fd, nnzRow.data(), graphData.sizeofnnzRow) == -1) {
        close(fd);
        throw std::runtime_error("could not write nnzRow");
    }
    // Write everything at once for colPtr
    if (write(fd, colPtr.data(), graphData.sizeofcolPtr) == -1) {
        close(fd);
        throw std::runtime_error("could not write colPtr");
    }
    
    close(fd);
    return;
}

// for bigger graphs Direct generation 
// Generates massive graphs directly to disk using mmap (OOM-Proof)
void generateLargeGraph(size_t NUM_NODES, float PROB, const char* pathFileName) {
    std::cout << "Generating graph with " << NUM_NODES << " nodes (Direct-to-Disk mode)..." << std::endl;

    // ---------------------------------------------------------
    // 1. SETUP LOCAL RNG (Deterministic)
    // ---------------------------------------------------------
    // We use a local engine instead of global 'RNG' to ensure we can 
    // reset the seed exactly for Pass 2.
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // ---------------------------------------------------------
    // 2. PASS 1: SIMULATION (Count Degrees)
    // ---------------------------------------------------------
    std::cout << "[Pass 1] Counting degrees..." << std::endl;
    
    // RAM Usage: NUM_NODES * 8 bytes (Tiny. 100k nodes = 0.8 MB)
    std::vector<size_t> degrees(NUM_NODES, 0);
    size_t total_edges_count = 0; // This will be size of colPtr

    for (size_t i = 0; i < NUM_NODES; ++i) {
        for (size_t j = i + 1; j < NUM_NODES; ++j) {
            if (dist(gen) <= PROB) {
                degrees[i]++;
                degrees[j]++;
                total_edges_count += 2; // Undirected: (i,j) and (j,i)
            }
        }
    }

    // ---------------------------------------------------------
    // 3. PREPARE FILE & HEADER
    // ---------------------------------------------------------
    GraphHeader graphData;
    graphData.sizeofnnzRow = (NUM_NODES + 1) * sizeof(size_t);
    graphData.sizeofcolPtr = total_edges_count * sizeof(size_t);
    graphData.offset_nnz = sizeof(GraphHeader);
    graphData.offset_col = sizeof(GraphHeader) + graphData.sizeofnnzRow;

    size_t fileSize = sizeof(GraphHeader) + graphData.sizeofnnzRow + graphData.sizeofcolPtr;

    std::cout << "[Disk] Allocating " << fileSize / (1024 * 1024) << " MB..." << std::endl;

    // Open and Resize File
    int fd = open(pathFileName, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1) throw std::runtime_error("Failed to open file");
    
    if (ftruncate(fd, fileSize) == -1) {
        close(fd);
        throw std::runtime_error("Failed to resize file (Disk full?)");
    }

    // Map File to Memory
    char* map_addr = (char*)mmap(nullptr, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap failed");
    }

    // ---------------------------------------------------------
    // 4. WRITE HEADER & nnzRow (Prefix Sum)
    // ---------------------------------------------------------
    // Define Pointers into the file map
    GraphHeader* header_ptr = (GraphHeader*)map_addr;
    size_t* nnzRow_ptr = (size_t*)(map_addr + graphData.offset_nnz);
    size_t* colPtr_ptr = (size_t*)(map_addr + graphData.offset_col);

    // Write Header
    *header_ptr = graphData; 

    // Calculate & Write nnzRow directly to file
    // We also keep 'current_offset' in RAM to know where to write neighbors in Pass 2
    std::vector<size_t> current_write_pos(NUM_NODES); 
    size_t running_sum = 0;

    nnzRow_ptr[0] = 0;
    for (size_t i = 0; i < NUM_NODES; ++i) {
        current_write_pos[i] = running_sum; // Start of i's neighbors in colPtr
        running_sum += degrees[i];
        nnzRow_ptr[i + 1] = running_sum;    // End of i's neighbors
    }

    // ---------------------------------------------------------
    // 5. PASS 2: EXECUTION (Write Edges)
    // ---------------------------------------------------------
    std::cout << "[Pass 2] Writing edges..." << std::endl;
    
    // RESET GENERATOR TO EXACT SAME SEED
    gen.seed(42); 

    for (size_t i = 0; i < NUM_NODES; ++i) {
        for (size_t j = i + 1; j < NUM_NODES; ++j) {
            if (dist(gen) <= PROB) {
                // Edge exists: Write to file via pointers
                
                // Add j to i's list
                size_t pos_i = current_write_pos[i]++;
                colPtr_ptr[pos_i] = j;

                // Add i to j's list
                size_t pos_j = current_write_pos[j]++;
                colPtr_ptr[pos_j] = i;
            }
        }
        // Optional Progress bar for huge graphs
        if (i % 1000 == 0) std::cout << "\rProgress: " << (size_t)((float)i/NUM_NODES * 100) << "%" << std::flush;
    }
    std::cout << std::endl;

    // ---------------------------------------------------------
    // 6. SORT NEIGHBORS 
    // ---------------------------------------------------------
    // CSR usually expects sorted neighbors for faster intersection.
    // Since we are mmapped, we can sort strictly in-place on disk!
    std::cout << "[Post-Process] Sorting neighbor lists..." << std::endl;
    for(size_t i=0; i<NUM_NODES; ++i) {
        size_t start = nnzRow_ptr[i];
        size_t end = nnzRow_ptr[i+1];
        std::sort(colPtr_ptr + start, colPtr_ptr + end);
    }

    // ---------------------------------------------------------
    // 7. CLEANUP
    // ---------------------------------------------------------
    if (msync(map_addr, fileSize, MS_SYNC) == -1) {
        std::cerr << "Warning: msync failed" << std::endl;
    }
    munmap(map_addr, fileSize);
    close(fd);
    std::cout << "Success! Graph saved to " << pathFileName << std::endl;
}

#endif