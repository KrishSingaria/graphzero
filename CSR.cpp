#include "CSR.hpp"
#include "MemoryMap.hpp"

CSR::CSR(const char* graphPath){
    // constructor
    this->graphMap = new MemoryMap(graphPath);
    
    // this->nnzRow = reinterpret_cast<size_t*>(this->nnzRowMaped->get_data());
    // this->colPtr = reinterpret_cast<size_t*>(this->colPtrMaped->get_data());

    // get the first number to check if they are magic numbers 
    GraphHeader header = reinterpret_cast<GraphHeader*>(this->graphMap->get_data())[0]; 

    if(header.MAGIC_NUM != MAGIC_NUM){
        throw std::runtime_error("Magic number of files DOESN'T match with actual magic number");
    }
    this->sizeofnnzRow = header.sizeofnnzRow;
    this->sizeofcolPtr = header.sizeofcolPtr;

    this->nnzRow = reinterpret_cast<size_t*>(this->graphMap->get_data()) + header.offset_nnz/sizeof(size_t);
    this->colPtr = reinterpret_cast<size_t*>(this->graphMap->get_data()) + header.offset_col/sizeof(size_t);
}

CSR::~CSR(){
    //destructor 
    delete this->graphMap;
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

void CSR::set_access_pattern(bool isRandom){

    if(isRandom){
        madvise(this->nnzRow,this->sizeofnnzRow,MADV_RANDOM);
        madvise(this->colPtr,this->sizeofcolPtr,MADV_RANDOM);
    }else{
        madvise(this->nnzRow,this->sizeofcolPtr,MADV_SEQUENTIAL);
        madvise(this->colPtr,this->sizeofcolPtr,MADV_SEQUENTIAL);
    }
}