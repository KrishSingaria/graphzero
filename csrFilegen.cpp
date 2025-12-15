#include "csrFilegen.hpp"

void generateBinary(std::vector<size_t> nnzRow,std::vector<size_t> colPtr,const char* pathFileName){
    int fd = open(pathFileName,O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if(fd == -1){
        throw std::runtime_error("File open failed while generating binaray");
    }

    GraphHeader graphData; 
    graphData.sizeofnnzRow = nnzRow.size() * sizeof(size_t);
    graphData.sizeofcolPtr = colPtr.size() * sizeof(size_t);

    graphData.offset_nnz = sizeof(GraphHeader);
    graphData.offset_col = sizeof(GraphHeader) + graphData.sizeofnnzRow;

    if(write(fd,&graphData,sizeof(GraphHeader)) == -1){
        close(fd);
        throw std::runtime_error("could not write Graph Header");  
    }
    // Write everything at once for nnzRow
    if (write(fd, nnzRow.data(), graphData.sizeofnnzRow) == -1) {
        close(fd);
        throw std::runtime_error("could not write nnzRow");
    }
    // Write everything at once for colPtr
    if (write(fd, colPtr.data(), graphData.sizeofcolPtr) == -1) {
        close(fd);
        throw std::runtime_error("could not write colPtr");
    }
    
    close(fd);
    return;
}