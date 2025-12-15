#ifndef CSRFILEGEN_H
#define CSRFILEGEN_H
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include "MemoryMap.hpp"

void generateBinary(std::vector<size_t> nnzRow,std::vector<size_t> colPtr,const char* pathFileName);
#endif