#include "CSR.hpp"
#include "MemoryMap.hpp"

CSR::CSR(const char* nnzPath,const char* colPath){
    // constructor
    this->nnzRowMaped = new MemoryMap(nnzPath);
    this->colPtrMaped = new MemoryMap(colPath);
    
    this->nnzRow = reinterpret_cast<size_t*>(this->nnzRowMaped->get_data());
    this->colPtr = reinterpret_cast<size_t*>(this->colPtrMaped->get_data());
}

CSR::~CSR(){
    //destructor 
    delete this->nnzRowMaped;  
    delete this->colPtrMaped;
    this->nnzRow = nullptr;
    this->colPtr = nullptr;
}

size_t* CSR::get_nnzRow(){
    // return nnzRow pointer
    return this->nnzRow;
}
size_t* CSR::get_colPtr(){
    // return colPtr pointer
    return this->colPtr;
}

size_t CSR::get_degree(size_t nodeId){
    // return degree of nodeId, how many conections it have 
    return this->nnzRow[nodeId+1] - this->nnzRow[nodeId];
}

std::span<size_t> CSR::get_edges(size_t nodeId){
    // return the edges of nodeId
    size_t* p =&this->colPtr[this->nnzRow[nodeId]];
    size_t  d = this->get_degree(nodeId);
    return std::span<size_t>(p,d);
}