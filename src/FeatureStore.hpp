#ifndef FEATURE_STORE_H
#define FEATURE_STORE_H
#include <cstdint>
#include <stdexcept>
#include "MemoryMap.hpp"
#include <span>

enum class DataType : int32_t { 
    INT32   = 0, 
    INT64   = 1, 
    FLOAT32 = 2, 
    FLOAT64 = 3
};

#pragma pack(push,1) // to complier to use 1-byte alignment(no padding)
struct FeatureHeader
{
    char magic[8];          // 8bytes "GZDATA26"
    uint32_t flags;         // 4bytes
    DataType dtype;         // 4bytes
    uint64_t num_nodes;     // 8bytes
    uint64_t feature_dim;   // 8bytes
}; // 32 bytes aligned 
#pragma pack(pop) // restore default compiler alignment

class FeatureStore {
private:
    MemoryMap* fileMap;
    struct FeatureHeader header;
    char* data_ptr;
public:
    int64_t num_nodes;
    int64_t feature_dim;
    std::string filename;

    FeatureStore(const char* filename);
    ~FeatureStore();

    DataType get_dtype() const;
    char* get_data_ptr() const; // 1byte pointer

    template <typename T>
    std::span<T> get_data(int64_t nodeId);
};

inline FeatureStore::FeatureStore(const char* filename){
    this->filename = std::string(filename);
    this->fileMap = new MemoryMap(filename);
    this->header = reinterpret_cast<FeatureHeader*>(this->fileMap->get_data())[0];
    if (std::string_view(header.magic, 8) != "GZDATA26") {
        throw std::runtime_error("Corrupted or invalid .gd file! Magic bytes mismatch.");
    }
    this->num_nodes = this->header.num_nodes;
    this->feature_dim = this->header.feature_dim;
    this->data_ptr = reinterpret_cast<char*>(this->fileMap->get_data()) + sizeof(FeatureHeader);
}

inline FeatureStore::~FeatureStore(){
    this->data_ptr = nullptr;
    delete this->fileMap;
}

inline DataType FeatureStore::get_dtype() const{
    return this->header.dtype;
}

inline char* FeatureStore::get_data_ptr() const {
    return this->data_ptr;
}

template <typename T>
inline std::span<T> FeatureStore::get_data(int64_t nodeId){
    if(nodeId >= num_nodes){
        throw std::runtime_error("NodeId is greater than number of nodes.");
    }
    int64_t offset = nodeId * feature_dim; // start of data of nodeId
    T* p = reinterpret_cast<T*>(this->data_ptr) + offset;
    return std::span<T>(p,this->feature_dim);
}
#endif