#ifndef CSR_H
#define CSR_H
#include "MemoryMap.hpp"
#include <span>
#include <cstddef>

class CSR{
    // size_t is the type used for every data!!
private:
    MemoryMap* nnzRowMaped = nullptr;
    MemoryMap* colPtrMaped = nullptr;

    // defined as nnzRow[i] = nnzRow[i-1] + no of non zero row entries of ith row
    size_t* nnzRow; 
    size_t* colPtr;
public:
    CSR(const char* nnzPath,const char* colPath);
    ~CSR();

    size_t get_degree(size_t nodeId);
    std::span<size_t> get_edges(size_t nodeId);

    // accessors

    size_t* get_nnzRow();
    size_t* get_colPtr();
};
#endif