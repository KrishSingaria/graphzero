#include "csrFilegen.hpp"

void generateBinary(std::vector<size_t> data,const char* pathFileName){
    int fd = open(pathFileName,O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if(fd == -1){
        throw std::runtime_error("File open failed while generating binaray");
    }

    // Write everything at once
    size_t total_bytes = data.size() * sizeof(size_t);
    if (write(fd, data.data(), total_bytes) == -1) {
        close(fd);
        throw std::runtime_error("could not write data");
    }
    
    close(fd);
    return;
}