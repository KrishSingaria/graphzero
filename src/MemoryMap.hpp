#ifndef MEMORYMAP_H
#define MEMORYMAP_H
#include <string>
#include <cstddef>
#include <stdexcept>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

// only here
const uint64_t MAGIC_NUM = 8388354976772092519; // 'graphlit' converted in size_t

struct GraphHeader {
    uint64_t MAGIC_NUM 
    = 8388354976772092519;      // 'graphlit' converted in uint64_t
    uint64_t sizeofnnzRow;      // Needed to know size of nnzRow in Bytes (N+1)
    uint64_t sizeofcolPtr;      // Needed to know size of col_indices (M)
    uint64_t offset_nnz;        // Byte offset where nnzRow start
    uint64_t offset_col;        // Byte offset where colPtr start
    uint64_t num_nodes;         // Explicit count (N)
    uint64_t num_edges;         // Explicit count (M)
    uint64_t flags;     // flags later user
}; // 64 byte header perfect AVX-512 alignment

class MemoryMap
{
private:
    #ifndef _WIN32
    int fd; // file descriptor
    struct stat st;
    #endif

    size_t length;
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

inline MemoryMap::MemoryMap(const char* path){
    // acquires resource/bin file on the Path given
    #ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if(hFile == INVALID_HANDLE_VALUE) { 
        CloseHandle(hFile);
        throw std::runtime_error("File open failed"); 
    }
    
    LARGE_INTEGER fsize;
    if((GetFileSizeEx(hFile,&fsize)) == 0){
        CloseHandle(hFile);
        throw std::runtime_error("can not get file size ");
    }
    length = fsize.QuadPart;
    
    HANDLE hMap = CreateFileMappingA(hFile,NULL,PAGE_READONLY, 0,0,NULL);
    if(hMap == NULL) { 
        CloseHandle(hMap);
        CloseHandle(hFile);
        throw std:: runtime_error("File mapping failed"); 
    }

    mappedptr = MapViewOfFile(hMap,FILE_MAP_READ, 0,0,0);
    if(mappedptr == NULL) {
        throw std::runtime_error("mappedptr is NULL, could not map");
    }

    CloseHandle(hMap);
    CloseHandle(hFile);
    #else // linux/mac
    if((fd = open(path,O_RDONLY)) == -1){
        throw std::runtime_error("File open failed");
    }

    if(fstat(fd,&st) == -1){
        close(fd);
        throw std::runtime_error("File open failed");
    }
    length = st.st_size; // size in bytes

    if ((mappedptr = mmap(NULL,length,PROT_READ,MAP_SHARED,fd,0) ) == MAP_FAILED) {
        close(fd); // Clean up the fd we just opened
        throw std::runtime_error("mmap failed");
    }

    // Only use Huge Pages on Linux. macOS don't support this flag.
    #ifdef __linux__
        madvise(mappedptr, length, MADV_HUGEPAGE);
    #endif

    #endif 

}

inline MemoryMap::~MemoryMap(){
    // release resource, destory itself
    #ifdef _WIN32
    if (mappedptr != nullptr) {
        UnmapViewOfFile(mappedptr);
        mappedptr = nullptr;
    }
    #else// linux
    if(mappedptr != MAP_FAILED && mappedptr != nullptr){
        munmap(mappedptr,length);
    }
    
    if(fd != -1){
        close(fd);
    }
    fd = -1;
    
    length = 0;
    #endif 
}
inline void* MemoryMap::get_data(){
    // get data pointer 
    return mappedptr;
}

inline size_t MemoryMap::get_size(){
    // get the length in bytes
    return length;
}

#endif