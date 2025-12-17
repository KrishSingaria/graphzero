#ifndef GRAPHLITE_H
#define GRAPHLITE_H
#include "ThreadLocalRNG.hpp"
#include "CSR.hpp"
#include "MemoryMap.hpp"
#include "AliasTable.hpp"
#include <vector>
#include <span>
#include <unordered_map>

class Graphlite
{
private:
    CSR* storage;
public:
    Graphlite(const char* graphPath);
    ~Graphlite();
    
    std::vector<size_t> fySampling(size_t nodeId, int k);
    std::vector<size_t> ReservoirSampling(size_t nodeId, int k);
};

inline Graphlite::Graphlite(const char* graphPath){
    storage = new CSR(graphPath);
}

inline Graphlite::~Graphlite(){
    delete storage;
}

// uses Fisher-Yates Shuffling Method,[Could be bad for Memory SO NOT USING IT] 
inline std::vector<size_t> Graphlite::fySampling(size_t nodeId, int k){
    if(k < 1) return {};

    std::span<size_t> neighbours = storage->get_edges(nodeId);
    size_t deg = neighbours.size();

    if(deg <= (size_t)k){
        return std::vector<size_t>(neighbours.begin(),neighbours.end());
    }
       
    // selection first k neighbours
    std::vector<size_t> result(neighbours.begin(),neighbours.end());
    // Fisher-Yates shuffle first K elements 
    for(size_t i = 0; i < k; i++){
        size_t j = RNG.rand_int(i,deg-1);
        std::swap(result[i],result[j]);
    }

    result.resize(k);
    return result; 
}

// use Reservoir Sampling Method
inline std::vector<size_t> Graphlite::ReservoirSampling(size_t nodeId, int k){
    if(k < 1) return {};

    std::span<size_t> neighbours = storage->get_edges(nodeId);
    size_t deg = neighbours.size();

    if(deg <= (size_t)k){
        return std::vector<size_t>(neighbours.begin(),neighbours.end());
    }
    
    // selection k neighbours
    std::vector<size_t> result(neighbours.begin(),neighbours.begin() + k); // first k elements
    
    // Resevoir Sampling K elements 
    for(size_t i = k; i < deg; i++){
        size_t j = RNG.rand_int(0,i);
        if(j < k){
            result[j] = neighbours[i];
        }
    }

    return result;
}
#endif
