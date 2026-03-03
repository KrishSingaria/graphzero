#ifndef ALIASTABLE_H
#define ALIASTABLE_H
#include <vector>
#include <span>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <stack>
#include <list>
#include "ThreadLocalRNG.hpp"

class AliasTable
{
private:
    int64_t N = 0;
    std::vector<float> prob;
    std::vector<int> alias;
public:
    AliasTable() = default;
    AliasTable(const std::vector<float>& weights); // N buckets; N size cache

    int64_t sample() const;

    bool empty() const { return prob.empty(); }
};

inline AliasTable::AliasTable(const std::vector<float>& weights)
{
    if(weights.empty()) return;

    N = weights.size(); // size/ number of buckets
    prob.resize(N);
    alias.resize(N);

    double sum = std::accumulate(weights.begin(), weights.end() ,0.0);
    if(sum == 0) throw std::runtime_error("Weights sum can not be zero");

    double scale = N / sum;

    // Vose's Two stack algo
    std::stack<int64_t> underfull,overfull; // stores index i
    
    for(int64_t i = 0; i < N; ++i){
        prob[i] = static_cast<float>(weights[i] * scale);
        if(prob[i] < 1.0f){
            underfull.push(i);
        }else{
            overfull.push(i);
        }
    }

    while(!underfull.empty() && !overfull.empty()){
        int64_t idxUf = underfull.top(); underfull.pop();
        int64_t idxOf = overfull.top() ; overfull.pop();

        alias[idxUf] = idxOf;
        // prob[idxUf] + X(taken from prob[idxOf]) = 1.0f
        prob[idxOf] -= 1.0f - prob[idxUf];
        

        if(prob[idxOf] < 1.0f){
            underfull.push(idxOf);
        }else{
            overfull.push(idxOf);
        }
    }

    while (!underfull.empty())
    {
        int64_t top = underfull.top(); underfull.pop();
        prob[top] = 1.0f;
    }
    
    while (!overfull.empty())
    {
        int64_t top = overfull.top(); overfull.pop();
        prob[top] = 1.0f;
    }
}

inline int64_t AliasTable::sample() const{
    if(N == 0) return 0; 

    float uniformAtRandom = N*RNG.rand();
    int64_t intPart = static_cast<int64_t>(uniformAtRandom);
    float decimalPart = uniformAtRandom - static_cast<float>(intPart);
    
    if(decimalPart < prob[intPart]){
        return intPart;
    }else return alias[intPart];
}

// LRUCache table that handles AliasTable
class LRUTable
{
private:
    int64_t MAXCAPACITY;
    std::function<std::span<float>(int64_t)> getWeights;
    
    // Cache entry defination 
    using CacheEntry = std::pair<AliasTable, std::list<int64_t>::iterator>;
    std::unordered_map<int64_t,CacheEntry> isNodePresent;
    
    //LRU queue
    std::list<int64_t> lst;
public:
    LRUTable(int64_t maxCapacity, std::function<std::span<float>(int64_t)> weightFunc) : MAXCAPACITY(maxCapacity), getWeights(weightFunc) {}

    const AliasTable& get_alias_table(int64_t nodeId);
};

inline const AliasTable& LRUTable::get_alias_table(int64_t nodeID){
    auto it = isNodePresent.find(nodeID);
    if(it != isNodePresent.end()){ // cache HIT
        
        lst.splice(lst.begin(), lst, it->second.second);
        
        return it->second.first;
    }
    
    // cache Miss
    
    // Weights
    std::span<float> span_w = getWeights(nodeID);
    std::vector<float> weights(span_w.begin(),span_w.end());
    
    AliasTable newTable(weights);

    if(lst.size() >= MAXCAPACITY){
        int64_t lruNode = lst.back(); 
        lst.pop_back();
        isNodePresent.erase(lruNode);
    }

    lst.push_front(nodeID);
    isNodePresent[nodeID] = {std::move(newTable), lst.begin()};
    
    return isNodePresent[nodeID].first;
}

#endif 