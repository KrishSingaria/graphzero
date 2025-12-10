#ifndef MEMORYMAP_H
#define MEMORYMAP_H
#include <string>
#include <cstddef>
#include <sys/stat.h>

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