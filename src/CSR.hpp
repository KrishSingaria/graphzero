#ifndef CSR_H
#define CSR_H
#include "MemoryMap.hpp"
#include "csrFilegen.hpp"
#include <span>
#include <cstddef>

class CSR{
    // int64_t is the type used for every data!!
private:
    MemoryMap* graphMap = nullptr;

    // defined as nnzRow[i] = nnzRow[i-1] + no of non zero row entries of ith row
    int64_t* nnzRow; 
    int64_t  sizeofnnzRow; // size in bytes
    
    int64_t* colPtr;
    int64_t  sizeofcolPtr;   // size in bytes
    
    float* weightPtr;      // same size a colPtr
    
    int64_t flags;
    int64_t  num_nodes;  // num of nodes
    int64_t  num_edges;  // num of edges
public:
    bool has_weights;   // if the graph has weights or not, 
    
    CSR(const char* graphPath);
    ~CSR();

    int64_t get_degree(int64_t nodeId);
    std::span<int64_t> get_edges(int64_t nodeId);
    std::span<float> get_weights(int64_t nodeId);

    void set_access_pattern(bool isRandom);
    
    // accessors
    
    int64_t* get_nnzRow(){
        return nnzRow;
    }
    int64_t* get_colPtr(){
        return colPtr;
    }
    float* get_weightsPtr(){
        return weightPtr;
    }
    int64_t  get_num_nodes(){
        return num_nodes;
    }
    int64_t  get_num_edges(){
        return num_edges;
    }
};


inline CSR::CSR(const char* graphPath){
    // constructor
    this->graphMap = new MemoryMap(graphPath);

    // get the first number to check if they are magic numbers 
    GraphHeader header = reinterpret_cast<GraphHeader*>(this->graphMap->get_data())[0]; 

    if(header.MAGIC_NUM != MAGIC_NUM){
        throw std::runtime_error("Magic number of files DOESN'T match with actual magic number");
    }
    this->sizeofnnzRow = header.sizeofnnzRow;
    this->sizeofcolPtr = header.sizeofcolPtr;
    this->num_nodes = header.num_nodes;
    this->num_edges = header.num_edges;
    this->flags = header.flags;
    this->has_weights = (flags & 1) != 0; 

    this->nnzRow = reinterpret_cast<int64_t*>(this->graphMap->get_data()) + header.offset_nnz/sizeof(int64_t);
    this->colPtr = reinterpret_cast<int64_t*>(this->graphMap->get_data()) + header.offset_col/sizeof(int64_t);
    
    if(has_weights){
        char* base = static_cast<char*>(this->graphMap->get_data()); // 1byte shifts
        uint64_t offset_weights = align64(header.offset_col + this->sizeofcolPtr);
        this->weightPtr = reinterpret_cast<float*>(base + offset_weights);
    }else this->weightPtr = nullptr;  
}

inline CSR::~CSR(){
    //destructor 
    delete this->graphMap;
    this->nnzRow = nullptr;
    this->colPtr = nullptr;
    this->weightPtr = nullptr;
}

inline int64_t CSR::get_degree(int64_t nodeId){
    // return degree of nodeId, how many conections it have
    if(nodeId >= num_nodes) return 0;
    return this->nnzRow[nodeId+1] - this->nnzRow[nodeId];
}
inline std::span<int64_t> CSR::get_edges(int64_t nodeId){
    // return the edges of nodeId
    if(nodeId>= num_nodes){
        throw std::runtime_error("NodeId is greater than number of nodes.");
    }
    int64_t* p =&this->colPtr[this->nnzRow[nodeId]];
    int64_t  d = this->get_degree(nodeId);
    return std::span<int64_t>(p,d);
}

inline std::span<float> CSR::get_weights(int64_t nodeId){
    // return the weights of nodeId, only valid if has_weights is true
    if(nodeId>= num_nodes){
        throw std::runtime_error("NodeId is greater than number of nodes.");
    }
    if(!has_weights){
        throw std::runtime_error("Graph does not have weights");
    }
    float* p =&this->weightPtr[this->nnzRow[nodeId]];
    int64_t d = this->get_degree(nodeId);
    return std::span<float>(p,d);
}

inline void CSR::set_access_pattern(bool isRandom){
    #ifdef __linux__
    if(isRandom){
        madvise(this->nnzRow,this->sizeofnnzRow,MADV_RANDOM);
        madvise(this->colPtr,this->sizeofcolPtr,MADV_RANDOM);
    }else{
        madvise(this->nnzRow,this->sizeofcolPtr,MADV_SEQUENTIAL);
        madvise(this->colPtr,this->sizeofcolPtr,MADV_SEQUENTIAL);
    }
    #endif // linux only, no mac/windows
}

#endif