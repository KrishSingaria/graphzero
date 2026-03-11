#ifndef FEATURE_FILEGEN_H
#define FEATURE_FILEGEN_H

#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <charconv>
#include <algorithm>
#include "FeatureStore.hpp"
#include <type_traits>
#include <omp.h>

// PLATFORM DEPENDENT INCLUDES
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>   // For mmap (Large Graph)
    #include <unistd.h>     // For ftruncate, close
    #include <fcntl.h>      // For open constants
    #include <omp.h>        // Linux OpenMP
#endif

void scan_csv_pass1(const std::string& filepath, uint64_t& num_nodes,uint64_t& feature_dim){
    // open file 
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    // find maxnode id and feature dim in CSV in one pass
    std::string line;
    uint64_t max_node_id = 0;
    uint64_t max_feature_dim = 0;
    bool feature_dim_found = false;
    while (std::getline(file, line)) {
        if(line.empty()) continue; // skip empty lines
        std::string_view sv(line);
        // find first comma (node id)
        size_t comma_pos = sv.find(',');
        if (comma_pos == std::string_view::npos) {
            continue; // skip malformed lines   
        }
        // parse node id
        uint64_t node_id;
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + comma_pos, node_id);
        if (ec != std::errc()) {
            continue; // skip malformed node id
        }
        max_node_id = std::max(max_node_id, node_id);
        // count remaining commas to determine feature dim
        if(!feature_dim_found) {
            size_t feature_count = std::count(sv.begin() + comma_pos + 1, sv.end(), ',') + 1; // f1,f2,f3 -> 2 + 1 features
            max_feature_dim = feature_count;
            feature_dim_found = true;
        }
    }
    file.close();

    num_nodes = max_node_id + 1; 
    feature_dim = max_feature_dim;
}

template <typename T>
void parse_and_write_features(const std::string& csv_path, char* map_addr, uint64_t feature_dim) {
    // Cast the payload area to the specific type T
    T* feature_data_ptr = reinterpret_cast<T*>(map_addr + sizeof(FeatureHeader));
    
    std::ifstream file(csv_path);
    std::string line;

    while (std::getline(file, line)) {
        if(line.empty()) continue;
        std::string_view sv(line);
        size_t comma_pos = sv.find(',');
        if (comma_pos == std::string_view::npos) continue;

        uint64_t node_id;
        std::from_chars(sv.data(), sv.data() + comma_pos, node_id);

        size_t feature_start = comma_pos + 1;
        for (uint64_t f = 0; f < feature_dim; ++f) {
            size_t next_comma = sv.find(',', feature_start);
            std::string_view feature_str = (next_comma == std::string_view::npos) ? sv.substr(feature_start) : sv.substr(feature_start, next_comma - feature_start);
            
            // Parse the correct type T
            T feature_value;
            #if defined(__APPLE__)
                // Apple's libc++ doesn't support float from_chars yet, so we safely fallback
                if constexpr (std::is_same_v<T, float>) {
                    feature_value = std::stof(std::string(feature_str));
                } else if constexpr (std::is_same_v<T, double>) {
                    feature_value = std::stod(std::string(feature_str));
                } else {
                    // Apple does support it for integers!
                    std::from_chars(feature_str.data(), feature_str.data() + feature_str.size(), feature_value);
                }
            #else
                // Windows and Linux get the ultra-fast C++17 standard parsing
                std::from_chars(feature_str.data(), feature_str.data() + feature_str.size(), feature_value);
            #endif
            
            feature_data_ptr[node_id * feature_dim + f] = feature_value;
            
            if (next_comma == std::string_view::npos) break;
            feature_start = next_comma + 1;
        }
    }
    file.close();
}

inline void convert_csv_to_binary(const std::string& csv_path, const std::string& out_path,DataType dtype) {
    // This function would implement the logic to read the CSV and write the binary file
    // using the determined num_nodes and feature_dim. It would follow a similar pattern to the CSR conversion,
    // but instead of building a graph structure, it would build a feature matrix.
    uint64_t num_nodes, feature_dim;
    scan_csv_pass1(csv_path, num_nodes, feature_dim);
    std::cout << "CSV Analysis Complete. Num Nodes: " << num_nodes << ", Feature Dim: " << feature_dim << std::endl;

    FeatureHeader header;
    std::memcpy(header.magic, "GZDATA26", 8); // Magic String
    header.flags = 0;  
    header.dtype = dtype;
    header.num_nodes = num_nodes;
    header.feature_dim = feature_dim;

    char* map_addr = nullptr;
    uint64_t element_size = 0;
    switch (dtype){
        case DataType::INT32: element_size = sizeof(int32_t); break;
        case DataType::INT64: element_size = sizeof(int64_t); break;
        case DataType::FLOAT32: element_size = sizeof(float); break;
        case DataType::FLOAT64: element_size = sizeof(double); break;
        default: throw std::runtime_error("Unsupported data type");
    }
    uint64_t feature_size = num_nodes * feature_dim * element_size;
    uint64_t file_size = sizeof(FeatureHeader) + feature_size;
    std::cout << "Creating binary file of size " << file_size / (1024*1024) << "MB..." << std::endl;

    // OS SPECIFIC FILE CREATION
    #ifdef _WIN32
        HANDLE hFile = CreateFileA(out_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {  throw std::runtime_error("Failed to create file"); }

        LARGE_INTEGER li;
        li.QuadPart = file_size;
        if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) { CloseHandle(hFile);  throw std::runtime_error("Failed to set file pointer"); }
        if (!SetEndOfFile(hFile)) { CloseHandle(hFile);  throw std::runtime_error("Failed to resize file"); }

        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
        if (hMap == NULL) { CloseHandle(hFile);  throw std::runtime_error("Failed to create mapping"); }

        map_addr = (char*)MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, 0);
        if (map_addr == NULL) { CloseHandle(hMap); CloseHandle(hFile);  throw std::runtime_error("MapViewOfFile failed"); }
    #else
        int fd = open(out_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd == -1) {  throw std::runtime_error("Failed to open file"); }

        if (ftruncate(fd, file_size) == -1) {
            close(fd); 
            throw std::runtime_error("Resize failed");
        }
        
        map_addr = (char*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map_addr == MAP_FAILED) {
            close(fd); 
            throw std::runtime_error("MMap failed");
        }
    #endif

    // write header
    std::memcpy(map_addr, &header, sizeof(header));
    // write features
    switch(dtype){
        case DataType::INT32: 
            parse_and_write_features<int32_t>(csv_path, map_addr, feature_dim); break;
        case DataType::INT64: 
            parse_and_write_features<int64_t>(csv_path, map_addr, feature_dim); break;
        case DataType::FLOAT32: 
            parse_and_write_features<float>(csv_path, map_addr, feature_dim); break;
        case DataType::FLOAT64: 
            parse_and_write_features<double>(csv_path, map_addr, feature_dim); break;
        default: throw std::runtime_error("Unsupported data type");
    }

    #ifdef _WIN32
        UnmapViewOfFile(map_addr);
        CloseHandle(hMap);
        CloseHandle(hFile);
    #else
        msync(map_addr, file_size, MS_SYNC);
        munmap(map_addr, file_size);
        close(fd);
    #endif
}
#endif