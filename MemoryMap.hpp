#ifndef MEMORYMAP_H
#define MEMORYMAP_H
#include <string>
#include <cstddef>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>

// only here
const size_t MAGIC_NUM = 8388354976772092519; // 'graphlit' converted in size_t

struct GraphHeader {
    uint64_t MAGIC_NUM 
    = 8388354976772092519;      // 'graphlit' converted in uint64_t
    uint64_t sizeofnnzRow;      // Needed to know size of nnzRow in Bytes (N+1)
    uint64_t sizeofcolPtr;      // Needed to know size of col_indices (M)
    uint64_t offset_nnz;        // Byte offset where nnzRow start
    uint64_t offset_col;        // Byte offset where colPtr start
};

class MemoryMap
{
private:
    int fd; // file descriptor
    size_t length;
    struct stat st;
    void* mappedptr;
public:
    // constructor accquires, no flags currently 
    MemoryMap(const char* path);
    // it releases
    ~MemoryMap();

    // accessors
    void* get_data();
    size_t get_size();
};

#endif