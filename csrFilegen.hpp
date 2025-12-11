#ifndef CSRFILEGEN_H
#define CSRFILEGEN_H
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

void generateBinary(std::vector<size_t> data,const char* pathFileName);
#endif