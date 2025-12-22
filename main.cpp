#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h> // Parallel processing
#include "Graphlite.hpp"

// Compilation: 
// g++ -std=c++20 -O3 -march=native -fopenmp ./main.cpp -o bench

int main() {
    std::cout << "Loading Graph..." << std::endl;
    Graphlite graph("graph.gl");
    
    // Setup Benchmark Parameters
    const size_t NUM_NODES = 100000; // Must match your generated graph size
    const size_t WALK_LENGTH = 10;
    const size_t WALKS_PER_NODE = 1; 
    
    std::cout << "Starting Benchmark on " << omp_get_max_threads() << " threads..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    // The Parallel Loop
    // #pragma omp parallel for schedule(dynamic)
    #pragma omp parallel for
    for (size_t i = 0; i < NUM_NODES; ++i) {
        // Run walk for every node
        // std::cout<<"Loop "<<i<<std::endl;
        std::vector<size_t> walk = graph.randomWalk(i, WALK_LENGTH, 1.0f, 1.0f);
        
        // Prevent compiler from optimizing away the code (Do something with result)
        if (walk.size() != WALK_LENGTH) {
            std::cerr << "Error! on this node " << i << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // Report
    double total_steps = NUM_NODES * WALKS_PER_NODE * WALK_LENGTH;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Throughput: " << (total_steps / diff.count()) / 1000000.0 << " M steps/sec" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    return 0;
}