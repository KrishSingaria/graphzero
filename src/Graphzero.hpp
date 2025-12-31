#ifndef GRAPHZERO_H
#define GRAPHZERO_H
#include "ThreadLocalRNG.hpp"
#include "CSR.hpp"
#include "MemoryMap.hpp"
#include "AliasTable.hpp"
#include <omp.h> 
#include <vector>
#include <span>
#include <unordered_map>
#include <algorithm>

class Graphzero
{
private:
    CSR* storage;
    size_t MAXLRUSIZE = 10000;
public:
    std::string filename;
    size_t num_nodes;
    size_t num_edges;

    Graphzero(const char* filename);
    ~Graphzero();
    
    bool isNeighbor(size_t u, size_t v);
    size_t node2vec_step(size_t curr, size_t prev, float p, float q, const AliasTable& table);

    std::vector<size_t> fySampling(size_t nodeId, int k);
    std::vector<size_t> ReservoirSampling(size_t nodeId, int k);
    std::vector<size_t> randomWalk(size_t start_node, size_t length, float p, float q);
    
    std::vector<size_t> batchRandomWalk(const std::vector<size_t>& startNodes, size_t walkLength, float p, float q);
    std::vector<size_t> batchRandomUniformWalk(const std::vector<size_t>& startNodes, size_t walkLength);
    std::vector<size_t> batchRandomFanout(const std::vector<size_t>& startNodes, size_t K);

    CSR* get_storage(){
        return storage;
    }
};

inline Graphzero::Graphzero(const char* filename){
    this->filename = filename;
    storage = new CSR(filename);
    num_nodes = storage->get_num_nodes();
    num_edges = storage->get_num_edges();
}

inline Graphzero::~Graphzero(){
    delete storage;
}

// uses Fisher-Yates Shuffling Method,[Could be bad for Memory SO NOT USING IT] 
inline std::vector<size_t> Graphzero::fySampling(size_t nodeId, int k){
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
inline std::vector<size_t> Graphzero::ReservoirSampling(size_t nodeId, int k){
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

// is v neighbor of u ? 
inline bool Graphzero::isNeighbor(size_t u, size_t v){
    
    auto edges = storage->get_edges(u);
    for(auto&& i: edges){
        if(i == v) return true;
    }
    return false;
}

// return next step in node2vec algo
inline size_t Graphzero::node2vec_step(size_t curr, size_t prev, float p, float q, const AliasTable& table){
    // Rejection sampling 
    float maxBias = (std::max)({1.0f,1.0f/p,1.0f/q}); // for windows max

    while (true)
    {
        size_t neighbour = storage->get_edges(curr)[table.sample()];

        float bias = 0.0f;

        if(neighbour == prev){
            bias = 1.0f / p;
        }else if(isNeighbor(prev,neighbour)){
            bias = 1.0f;
        }else{
            bias = 1.0f / q;
        }

        if(bias>= maxBias || RNG.rand() < (bias / maxBias)){
            return neighbour;
        }// else run loop again
    }
    
}

inline std::vector<size_t> Graphzero::randomWalk(size_t start_node, size_t length, float p, float q){

    static thread_local LRUTable lruCache(MAXLRUSIZE);

    size_t next,curr = start_node,prev;

    std::vector<size_t> walk;
    walk.reserve(length);
    walk.push_back(curr);

    for (size_t i = 1; i < length; i++)
    {
        size_t degree = storage->get_degree(curr);
        if (degree == 0) break; // Dead end

        auto table = lruCache.get_alias_table(curr,storage->get_degree(curr));
        if(i == 1){
            next = storage->get_edges(curr)[table.sample()];
        }else {
            next = node2vec_step(curr,prev,p,q,table);
        }
        prev = curr;
        curr = next;
        walk.push_back(next);
    }

    return walk;
}

//keep p = 1.0f and q = 1.0f for default values.
inline std::vector<size_t> Graphzero::batchRandomWalk(const std::vector<size_t>& startNodes, size_t walkLength, float p, float q){
    std::vector<size_t> results(walkLength*startNodes.size());

    // set only for random walks 
    storage->set_access_pattern(true);

    #pragma omp parallel for
    for(signed long long i = 0; i < startNodes.size(); i++){
        std::vector<size_t> walk = randomWalk(startNodes[i],walkLength,p,q);
        
        // thread safe
        size_t offset = i*walkLength;
        for(int j = 0; j < walk.size(); j++){
            results[j+offset] = walk[j];
        }
    }

    // reset
    storage->set_access_pattern(false);
    return results;
}

inline std::vector<size_t> Graphzero::batchRandomUniformWalk(const std::vector<size_t>& startNodes, size_t walkLength){
    std::vector<size_t> results(walkLength*startNodes.size());
    
    // set only for random walks 
    storage->set_access_pattern(true);

    #pragma omp parallel for
    for(signed long long i = 0; i < startNodes.size(); i++){
        // walking here 
        size_t offset = i*walkLength;
        size_t curr = startNodes[i], next;
        results[offset] = curr;
        for(size_t j = 1; j < walkLength; ++j){
            auto edges = storage->get_edges(curr);

            if(edges.size() == 0){
                results[offset+j] = curr;
                continue;
            }

            next = edges[RNG.rand_int(0,edges.size()-1)];
            results[offset + j] = next;
            curr = next;
        }   
    }

    // reset
    storage->set_access_pattern(false);
    return results;
}


inline std::vector<size_t> Graphzero::batchRandomFanout(const std::vector<size_t>& startNodes, size_t K){
    std::vector<size_t> results(K * startNodes.size(), 0);

    // set only for sampling access pattern
    storage->set_access_pattern(true);

    #pragma omp parallel for
    for (signed long long i = 0; i < (signed long long)startNodes.size(); ++i) {
        std::vector<size_t> samples = ReservoirSampling(startNodes[i], (int)K);

        // thread safe write into results
        size_t offset = (size_t)i * K;
        for (size_t j = 0; j < samples.size(); ++j) {
            results[offset + j] = samples[j];
        }
    }

    // reset
    storage->set_access_pattern(false);
    return results;
}
#endif
