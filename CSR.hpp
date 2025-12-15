#ifndef CSR_H
#define CSR_H
#include "MemoryMap.hpp"
#include <span>
#include <cstddef>
#include <sys/mman.h>

class CSR{
    // size_t is the type used for every data!!
private:
    MemoryMap* graphMap = nullptr;

    // defined as nnzRow[i] = nnzRow[i-1] + no of non zero row entries of ith row
    size_t* nnzRow; 
    size_t  sizeofnnzRow; // size in bytes

    size_t* colPtr;
    size_t  sizeofcolPtr; // size in bytes
public:
    CSR(const char* graphPath);
    ~CSR();

    size_t get_degree(size_t nodeId);
    std::span<size_t> get_edges(size_t nodeId);

    void set_access_pattern(bool isRandom);

    // accessors

    size_t* get_nnzRow();
    size_t* get_colPtr();
};
#endif