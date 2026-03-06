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
    int64_t MAXLRUSIZE = 10000;
public:
    std::string filename;
    int64_t num_nodes;
    int64_t num_edges;
    bool has_weights;

    Graphzero(const char* filename);
    ~Graphzero();
    
    bool isNeighbor(int64_t u, int64_t v);
    int64_t node2vec_step(int64_t curr, int64_t prev, float p, float q, const AliasTable& table);

    std::vector<int64_t> fySampling(int64_t nodeId, int k);
    void ReservoirSampling(int64_t nodeId, int k, int64_t* result);
    void randomWalk(int64_t start_node, int64_t length, float p, float q,int64_t* walk);
    
    std::vector<int64_t>* batchRandomWalk(const std::vector<int64_t>& startNodes, int64_t walkLength, float p, float q);
    std::vector<int64_t>* batchRandomUniformWalk(const std::vector<int64_t>& startNodes, int64_t walkLength);
    std::vector<int64_t>* batchRandomFanout(const std::vector<int64_t>& startNodes, int64_t K);

    CSR* get_storage(){
        return storage;
    }
};

inline Graphzero::Graphzero(const char* filename){
    this->filename = filename;
    storage = new CSR(filename);
    num_nodes = storage->get_num_nodes();
    num_edges = storage->get_num_edges();
    has_weights = storage->has_weights;
}

inline Graphzero::~Graphzero(){
    delete storage;
}

// uses Fisher-Yates Shuffling Method,[Could be bad for Memory SO NOT USING IT] 
inline std::vector<int64_t> Graphzero::fySampling(int64_t nodeId, int k){
    if(k < 1) return {};

    std::span<int64_t> neighbours = storage->get_edges(nodeId);
    int64_t deg = neighbours.size();

    if(deg <= (int64_t)k){
        return std::vector<int64_t>(neighbours.begin(),neighbours.end());
    }
       
    // selection first k neighbours
    std::vector<int64_t> result(neighbours.begin(),neighbours.end());
    // Fisher-Yates shuffle first K elements 
    for(int64_t i = 0; i < k; i++){
        int64_t j = RNG.rand_int(i,deg-1);
        std::swap(result[i],result[j]);
    }

    result.resize(k);
    return result; 
}

// use Reservoir Sampling Method, without weigths
inline void Graphzero::ReservoirSampling(int64_t nodeId, int k, int64_t* result){
    if(k < 1) return;

    std::span<int64_t> neighbours = storage->get_edges(nodeId);
    int64_t deg = neighbours.size();

    if(deg <= (int64_t)k){
        for(int i = 0; i < deg; i++){
            result[i] = neighbours[i];
        }
        return;
    }
    
    // selection k neighbours, first k elements
    for(int i = 0; i < k; i++){
        result[i] = neighbours[i];
    }
    
    // Resevoir Sampling K elements 
    for(int64_t i = k; i < deg; i++){
        int64_t j = RNG.rand_int(0,i);
        if(j < k){
            result[j] = neighbours[i];
        }
    }
}

// is v neighbor of u ? 
inline bool Graphzero::isNeighbor(int64_t u, int64_t v){
    
    auto edges = storage->get_edges(u);
    for(auto&& i: edges){
        if(i == v) return true;
    }
    return false;
}

// return next step in node2vec algo
inline int64_t Graphzero::node2vec_step(int64_t curr, int64_t prev, float p, float q, const AliasTable& table){
    // Rejection sampling 
    float maxBias = (std::max)({1.0f,1.0f/p,1.0f/q}); // for windows max

    while (true)
    {
        int64_t neighbour = storage->get_edges(curr)[table.sample()];

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

inline void Graphzero::randomWalk(int64_t start_node, int64_t length, float p, float q, int64_t* walk){

    auto weightFunc = [this](int64_t nodeID){
        if(this->has_weights) return this->storage->get_weights(nodeID);
        else{
            return std::span<float>(); // lrutable detect if empty
        }
    };

    static thread_local LRUTable lruCache(MAXLRUSIZE,weightFunc);

    int64_t next,curr = start_node,prev;

    walk[0] = start_node;

    for (int64_t i = 1; i <= length; i++)
    {
        int64_t degree = storage->get_degree(curr);
        if (degree == 0){ // Dead end
            walk[i] = curr;
            continue;
        }; 

        auto table = lruCache.get_alias_table(curr,this->storage->get_degree(curr));
        if(i == 1){
            next = storage->get_edges(curr)[table.sample()];
        }else {
            next = node2vec_step(curr,prev,p,q,table);
        }
        prev = curr;
        curr = next;
        walk[i] = next;
    }
}

//keep p = 1.0f and q = 1.0f for default values.
inline std::vector<int64_t>* Graphzero::batchRandomWalk(const std::vector<int64_t>& startNodes, int64_t walkLength, float p, float q){
    std::vector<int64_t>* results = new std::vector<int64_t>((walkLength+1)*startNodes.size());

    // set only for random walks 
    storage->set_access_pattern(true);

    #pragma omp parallel for
    for(signed long long i = 0; i < startNodes.size(); i++){
        
        // thread safe
        int64_t offset = i*(walkLength+1);
        
        randomWalk(startNodes[i],walkLength,p,q, results->data() + offset);
    }

    // reset
    storage->set_access_pattern(false);
    return results;
}

inline std::vector<int64_t>* Graphzero::batchRandomUniformWalk(const std::vector<int64_t>& startNodes, int64_t walkLength){
    std::vector<int64_t>* results = new std::vector<int64_t>((walkLength + 1)*startNodes.size());
    
    // set only for random walks 
    storage->set_access_pattern(true);

    #pragma omp parallel for
    for(signed long long i = 0; i < startNodes.size(); i++){
        // walking here 
        int64_t offset = (int64_t)(i*(walkLength+1));
        int64_t curr = startNodes[i], next;
        (*results)[offset] = curr;
        for(int64_t j = 1; j < walkLength; ++j){
            auto edges = storage->get_edges(curr);

            if(edges.size() == 0){
                (*results)[offset+j] = curr;
                continue;
            }

            next = edges[RNG.rand_int(0,edges.size()-1)];
            (*results)[offset + j] = next;
            curr = next;
        }   
    }

    // reset
    storage->set_access_pattern(false);
    return results;
}


inline std::vector<int64_t>* Graphzero::batchRandomFanout(const std::vector<int64_t>& startNodes, int64_t K){
    std::vector<int64_t>* results = new std::vector<int64_t>(K * startNodes.size());

    // set only for sampling access pattern
    storage->set_access_pattern(true);

    #pragma omp parallel for
    for (signed long long i = 0; i < (signed long long)startNodes.size(); ++i) {

        // thread safe write into results
        int64_t offset = i * K;
        this->ReservoirSampling(startNodes[i], (int)K, results->data() + offset);
    }

    // reset
    storage->set_access_pattern(false);
    return results;
}
#endif
