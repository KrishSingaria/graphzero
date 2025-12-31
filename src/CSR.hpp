#ifndef CSR_H
#define CSR_H
#include "MemoryMap.hpp"
#include <span>
#include <cstddef>

class CSR{
    // size_t is the type used for every data!!
private:
    MemoryMap* graphMap = nullptr;

    // defined as nnzRow[i] = nnzRow[i-1] + no of non zero row entries of ith row
    size_t* nnzRow; 
    size_t  sizeofnnzRow; // size in bytes
    
    size_t* colPtr;
    size_t  sizeofcolPtr; // size in bytes
    
    size_t flags;
    size_t  num_nodes; // num of nodes
    size_t  num_edges; // num of edges
public:
    
    CSR(const char* graphPath);
    ~CSR();

    size_t get_degree(size_t nodeId);
    std::span<size_t> get_edges(size_t nodeId);

    void set_access_pattern(bool isRandom);
    
    // accessors
    
    size_t* get_nnzRow(){
        return nnzRow;
    }
    size_t* get_colPtr(){
        return colPtr;
    }
    size_t  get_num_nodes(){
        return num_nodes;
    }
    size_t  get_num_edges(){
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

    this->nnzRow = reinterpret_cast<size_t*>(this->graphMap->get_data()) + header.offset_nnz/sizeof(size_t);
    this->colPtr = reinterpret_cast<size_t*>(this->graphMap->get_data()) + header.offset_col/sizeof(size_t);
}

inline CSR::~CSR(){
    //destructor 
    delete this->graphMap;
    this->nnzRow = nullptr;
    this->colPtr = nullptr;
}

inline size_t CSR::get_degree(size_t nodeId){
    // return degree of nodeId, how many conections it have 
    return this->nnzRow[nodeId+1] - this->nnzRow[nodeId];
}
inline std::span<size_t> CSR::get_edges(size_t nodeId){
    // return the edges of nodeId
    size_t* p =&this->colPtr[this->nnzRow[nodeId]];
    size_t  d = this->get_degree(nodeId);
    return std::span<size_t>(p,d);
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