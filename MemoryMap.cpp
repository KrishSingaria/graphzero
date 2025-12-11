#include "MemoryMap.hpp"

MemoryMap::MemoryMap(const char* path){
    // acquires resource/bin file on the Path given
    
    if((this->fd = open(path,O_RDONLY)) == -1){
        throw std::runtime_error("File open failed");
    }

    if(fstat(fd,&this->st) == -1){
        close(this->fd);
        throw std::runtime_error("File open failed");
    }
    this->length = st.st_size; // size in bytes

    if ((this->mappedptr = mmap(NULL,this->length,PROT_READ,MAP_SHARED,this->fd,0) ) == MAP_FAILED) {
        close(fd); // Clean up the fd we just opened
        throw std::runtime_error("mmap failed");
    }
}

MemoryMap::~MemoryMap(){
    // release resource, destory itself

    if(this->mappedptr != MAP_FAILED && this->mappedptr != nullptr){
        munmap(this->mappedptr,this->length);
    }
    
    if(this->fd != -1){
        close(this->fd);
    }

    this->fd = -1;
    this->length = 0;   
}
void* MemoryMap::get_data(){
    // get data pointer 
    return this->mappedptr;
}

size_t MemoryMap::get_size(){
    // get the length in bytes
    return this->length;
}