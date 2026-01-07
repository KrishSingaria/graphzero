#ifndef CSRFILEGEN_H
#define CSRFILEGEN_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>      // For std::ofstream
#include <random>       // For RNG
#include <algorithm>    // For std::sort
#include <cstdint>      // For uint64_t
#include <cstring>      // For memset, strchr
#include <charconv>     // For std::from_chars (Modern C++17)

// --- PLATFORM DEPENDENT INCLUDES ---
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>   // For mmap (Large Graph)
    #include <unistd.h>     // For ftruncate, close
    #include <fcntl.h>      // For open constants
    #include <omp.h>        // Linux OpenMP
#endif

#include "MemoryMap.hpp"

// Helper for alignment
inline uint64_t align64(uint64_t val) {
    return (val + 63) & ~63; 
}

// ---------------------------------------------------------
// 1. SMALL/MEDIUM GENERATOR (Use std::ofstream)
// ---------------------------------------------------------
// No OS-specific changes needed here; std::ofstream is cross-platform.
void generateBinary(std::vector<size_t>& nnzRow, std::vector<size_t>& colPtr, const char* pathFileName){
    
    // 1. Prepare Header
    GraphHeader graphData; 
    graphData.sizeofnnzRow = nnzRow.size() * sizeof(size_t);
    graphData.sizeofcolPtr = colPtr.size() * sizeof(size_t);
    graphData.num_nodes = nnzRow.size() - 1;
    graphData.num_edges = colPtr.size();
    graphData.flags = 0; 

    // 2. Calculate Offsets (With Alignment)
    graphData.offset_nnz = sizeof(GraphHeader); // Header is 64 bytes
    
    uint64_t end_of_nnz = graphData.offset_nnz + graphData.sizeofnnzRow;
    graphData.offset_col = align64(end_of_nnz); // Align the start of next array

    // 3. Open File (Standard C++)
    std::ofstream outfile(pathFileName, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("File open failed: " + std::string(pathFileName));
    }

    // 4. Write Header
    outfile.write((char*)&graphData, sizeof(GraphHeader));

    // 5. Write nnzRow
    outfile.write((char*)nnzRow.data(), graphData.sizeofnnzRow);

    // 6. Write Padding (Zeros)
    size_t padding_needed = graphData.offset_col - end_of_nnz;
    if (padding_needed > 0) {
        std::vector<char> zeros(padding_needed, 0);
        outfile.write(zeros.data(), padding_needed);
    }

    // 7. Write colPtr (Now perfectly aligned)
    outfile.write((char*)colPtr.data(), graphData.sizeofcolPtr);
    
    outfile.close();
    std::cout << "Successfully generated " << pathFileName << " with 64-byte alignment." << std::endl;
}


// ---------------------------------------------------------
// 2. LARGE GRAPH GENERATOR (Use mmap + Alignment)
// ---------------------------------------------------------
void generateLargeGraph(size_t NUM_NODES, float PROB, const char* pathFileName) {
    std::cout << "Generating graph with " << NUM_NODES << " nodes (Direct-to-Disk mode)..." << std::endl;

    // --- SETUP RNG ---
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // --- PASS 1: Count Degrees ---
    std::cout << "[Pass 1] Counting degrees..." << std::endl;
    std::vector<size_t> degrees(NUM_NODES, 0);
    size_t total_edges_count = 0;

    for (size_t i = 0; i < NUM_NODES; ++i) {
        for (size_t j = i + 1; j < NUM_NODES; ++j) {
            if (dist(gen) <= PROB) {
                degrees[i]++;
                degrees[j]++;
                total_edges_count += 2; 
            }
        }
    }

    // --- PREPARE HEADER & OFFSETS ---
    GraphHeader graphData;
    graphData.sizeofnnzRow = (NUM_NODES + 1) * sizeof(size_t);
    graphData.sizeofcolPtr = total_edges_count * sizeof(size_t);
    graphData.num_nodes = NUM_NODES;
    graphData.num_edges = total_edges_count;
    graphData.flags = 0;

    // ALIGNMENT LOGIC
    graphData.offset_nnz = sizeof(GraphHeader);
    uint64_t end_of_nnz = graphData.offset_nnz + graphData.sizeofnnzRow;
    graphData.offset_col = align64(end_of_nnz); // Round up to next 64 bytes

    // Total File Size must include the gap/padding
    size_t fileSize = graphData.offset_col + graphData.sizeofcolPtr;

    std::cout << "[Disk] Allocating " << fileSize / (1024 * 1024) << " MB..." << std::endl;

    char* map_addr = nullptr;

    // --- OS SPECIFIC FILE CREATION & MAPPING ---
    #ifdef _WIN32
        // Windows Implementation
        HANDLE hFile = CreateFileA(pathFileName, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to create file (Windows)");

        // Resize file
        LARGE_INTEGER li;
        li.QuadPart = fileSize;
        if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) { CloseHandle(hFile); throw std::runtime_error("Failed to set file pointer"); }
        if (!SetEndOfFile(hFile)) { CloseHandle(hFile); throw std::runtime_error("Failed to resize file"); }

        // Create Mapping
        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
        if (hMap == NULL) { CloseHandle(hFile); throw std::runtime_error("Failed to create mapping"); }

        // Map View
        map_addr = (char*)MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, 0);
        if (map_addr == NULL) { CloseHandle(hMap); CloseHandle(hFile); throw std::runtime_error("MapViewOfFile failed"); }
    #else
        // Linux Implementation
        int fd = open(pathFileName, O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd == -1) throw std::runtime_error("Failed to open file");
        
        // ftruncate fills new space with zeros
        if (ftruncate(fd, fileSize) == -1) {
            close(fd);
            throw std::runtime_error("Failed to resize file (Disk full?)");
        }

        map_addr = (char*)mmap(nullptr, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map_addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed");
        }
    #endif

    // --- WRITE HEADER ---
    GraphHeader* header_ptr = (GraphHeader*)map_addr;
    *header_ptr = graphData; 

    // --- SET POINTERS (Using Aligned Offsets) ---
    size_t* nnzRow_ptr = (size_t*)(map_addr + graphData.offset_nnz);
    size_t* colPtr_ptr = (size_t*)(map_addr + graphData.offset_col); 

    // --- WRITE nnzRow (Prefix Sum) ---
    std::vector<size_t> current_write_pos(NUM_NODES); 
    size_t running_sum = 0;

    nnzRow_ptr[0] = 0;
    for (size_t i = 0; i < NUM_NODES; ++i) {
        current_write_pos[i] = running_sum; 
        running_sum += degrees[i];
        nnzRow_ptr[i + 1] = running_sum;    
    }

    // --- PASS 2: WRITE EDGES ---
    std::cout << "[Pass 2] Writing edges..." << std::endl;
    gen.seed(42); // Reset RNG

    for (size_t i = 0; i < NUM_NODES; ++i) {
        for (size_t j = i + 1; j < NUM_NODES; ++j) {
            if (dist(gen) <= PROB) {
                size_t pos_i = current_write_pos[i]++;
                colPtr_ptr[pos_i] = j;

                size_t pos_j = current_write_pos[j]++;
                colPtr_ptr[pos_j] = i;
            }
        }
        if (i % 1000 == 0) std::cout << "\rProgress: " << (size_t)((float)i/NUM_NODES * 100) << "%" << std::flush;
    }
    std::cout << std::endl;

    // --- SORTING ---
    std::cout << "[Post-Process] Sorting neighbor lists..." << std::endl;
    // On Windows MSVC, OpenMP requires special flags, stick to standard sort for compatibility
    // #pragma omp parallel for // Uncomment if using OpenMP on Windows
    for(size_t i=0; i<NUM_NODES; ++i) {
        size_t start = nnzRow_ptr[i];
        size_t end = nnzRow_ptr[i+1];
        std::sort(colPtr_ptr + start, colPtr_ptr + end);
    }

    // --- CLEANUP ---
    #ifdef _WIN32
        UnmapViewOfFile(map_addr);
        CloseHandle(hMap);
        CloseHandle(hFile);
    #else
        msync(map_addr, fileSize, MS_SYNC);
        munmap(map_addr, fileSize);
        close(fd);
    #endif
    
    std::cout << "Success! Aligned Graph saved to " << pathFileName << std::endl;
}

// ===============================
// edges csv to .gl file generator  
// ===============================

// Modern C++ Fast CSV Line Parser
void parse_line(char* line, uint64_t& u, uint64_t& v) {
    // Find comma
    char* comma = strchr(line, ',');
    if (!comma) return;
    
    // C++17 std::from_chars (Faster than atoll)
    std::from_chars(line, comma, u);
    std::from_chars(comma + 1, line + strlen(line), v);
}

void convert_csv(const std::string& csv_path, const std::string& out_path, bool directed) {
    std::cout << "Starting Conversion: " << csv_path << std::endl;

    // --- PASS 1: DISCOVERY ---
    std::cout << "[Pass 1] Scanning for Max Node ID and Degrees..." << std::endl;
    
    // Standard C I/O is cross-platform and fast enough
    FILE* f = fopen(csv_path.c_str(), "r");
    if (!f) throw std::runtime_error("Could not open CSV");

    char buffer[1024];
    uint64_t max_node = 0;
    uint64_t edge_count = 0;
    
    // Skip header if exists
    if (fgets(buffer, sizeof(buffer), f)) {
        if (!isdigit(buffer[0])) {
            std::cout << "Skipping header: " << buffer;
        } else {
            rewind(f); 
        }
    }

    std::vector<uint64_t> degrees;
    
    while (fgets(buffer, sizeof(buffer), f)) {
        uint64_t u, v;
        parse_line(buffer, u, v);
        max_node = std::max(max_node,(std::max)(u,v));
        if (u >= degrees.size() || v >= degrees.size()) {
            size_t new_max = (std::max)(u, v) + 1;
            if (new_max > degrees.size()) {
                degrees.resize((std::max)(new_max, degrees.size() * 2), 0);
            }
        }
        
        degrees[u]++;
        if (!directed) degrees[v]++;
        
        edge_count++;
        if (edge_count % 1000000 == 0) std::cout << "\rScanned " << edge_count << " edges..." << std::flush;
    }
    
    degrees.resize(max_node + 1); 

    uint64_t num_nodes = degrees.size();
    
    std::cout << "\nFound Nodes: " << num_nodes << ", Edges: " << edge_count 
              << (directed ? " (Directed)" : " (Undirected)") << std::endl;

    // --- PREPARE OUTPUT FILE ---
    uint64_t total_written_edges = directed ? edge_count : edge_count * 2;
    
    GraphHeader header;
    header.num_nodes = num_nodes;
    header.num_edges = total_written_edges;
    header.sizeofnnzRow = (num_nodes + 1) * sizeof(uint64_t);
    header.sizeofcolPtr = total_written_edges * sizeof(uint64_t);
    header.flags = 0;

    header.offset_nnz = sizeof(GraphHeader);
    uint64_t end_nnz = header.offset_nnz + header.sizeofnnzRow;
    header.offset_col = align64(end_nnz);

    size_t file_size = header.offset_col + header.sizeofcolPtr;

    std::cout << "[Disk] Creating " << file_size / (1024*1024) << "MB binary file..." << std::endl;
    
    char* map_addr = nullptr;

    // --- OS SPECIFIC FILE CREATION ---
    #ifdef _WIN32
        HANDLE hFile = CreateFileA(out_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) { fclose(f); throw std::runtime_error("Failed to create file"); }

        LARGE_INTEGER li;
        li.QuadPart = file_size;
        if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) { CloseHandle(hFile); fclose(f); throw std::runtime_error("Failed to set file pointer"); }
        if (!SetEndOfFile(hFile)) { CloseHandle(hFile); fclose(f); throw std::runtime_error("Failed to resize file"); }

        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
        if (hMap == NULL) { CloseHandle(hFile); fclose(f); throw std::runtime_error("Failed to create mapping"); }

        map_addr = (char*)MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, 0);
        if (map_addr == NULL) { CloseHandle(hMap); CloseHandle(hFile); fclose(f); throw std::runtime_error("MapViewOfFile failed"); }
    #else
        int fd = open(out_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd == -1) { fclose(f); throw std::runtime_error("Failed to open file"); }

        if (ftruncate(fd, file_size) == -1) {
            close(fd); fclose(f);
            throw std::runtime_error("Resize failed");
        }
        
        map_addr = (char*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map_addr == MAP_FAILED) {
            close(fd); fclose(f);
            throw std::runtime_error("MMap failed");
        }
    #endif

    // --- WRITE HEADER & POINTERS ---
    memcpy(map_addr, &header, sizeof(header));
    
    uint64_t* indptr = (uint64_t*)(map_addr + header.offset_nnz);
    uint64_t* indices = (uint64_t*)(map_addr + header.offset_col);

    // Build Prefix Sum (Indptr)
    std::vector<uint64_t> write_offsets(num_nodes);
    uint64_t running_sum = 0;
    
    indptr[0] = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        write_offsets[i] = running_sum;
        running_sum += degrees[i];
        indptr[i+1] = running_sum;
    }

    // --- PASS 2: PLACEMENT ---
    std::cout << "[Pass 2] Writing edges to disk..." << std::endl;
    rewind(f); 
    
    if (fgets(buffer, sizeof(buffer), f) && isdigit(buffer[0])) rewind(f);

    size_t processed = 0;
    while (fgets(buffer, sizeof(buffer), f)) {
        uint64_t u, v;
        parse_line(buffer, u, v);

        // Place u -> v
        uint64_t offset = write_offsets[u]++;
        indices[offset] = v;

        if (!directed) {
            // Place v -> u
            offset = write_offsets[v]++;
            indices[offset] = u;
        }
        processed++;
        if (processed % 1000000 == 0) std::cout << "\rWritten " << processed << " edges..." << std::flush;
    }
    
    fclose(f);
    std::cout << "\n[Post-Process] Sorting neighbor lists..." << std::endl;
    
    // Using standard sort for portability (OpenMP requires flags)
    #pragma omp parallel for
    for (signed long long i = 0; i < num_nodes; ++i) {
        std::sort(indices + indptr[i], indices + indptr[i+1]);
    }

    // Cleanup
    #ifdef _WIN32
        UnmapViewOfFile(map_addr);
        CloseHandle(hMap);
        CloseHandle(hFile);
    #else
        msync(map_addr, file_size, MS_SYNC);
        munmap(map_addr, file_size);
        close(fd);
    #endif

    std::cout << "Conversion Complete: " << out_path << std::endl;
}

#endif