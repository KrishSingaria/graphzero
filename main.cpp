// test code 
#include <iostream>
#include <vector>
#include <span>

#include "MemoryMap.hpp"
#include "CSR.hpp"
#include "csrFilegen.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    vector<size_t> nnzRowData = {0,2,5,7,11,13,14};
    vector<size_t> colPtrData = {1,3,0,2,3,1,3,0,1,2,4,3,5,4};
    
    generateBinary(nnzRowData,colPtrData,"./graph.gl");

    cout<<"FILE GEN COMPLETE\n";

    CSR storage("./graph.gl");
    
    // for performance  
    // first loop 
    for(int i = 0; i < 1e6; i++){
        size_t deg = 0;
        for(size_t nodeId = 0; nodeId < 6; nodeId++){
            deg += storage.get_degree(nodeId);
            auto edges = storage.get_edges(nodeId);
        }
    }

    cout<<"---FIRST RUN COMPLETED---\n";
    // second loop Cache should be used 
    for(int i = 0; i < 1e6; i++){
        size_t deg = 0;
        for(size_t nodeId = 0; nodeId < 6; nodeId++){
            deg += storage.get_degree(nodeId);
            auto edges = storage.get_edges(nodeId);
        }
    }
    // single run
    // auto edges = storage.get_edges(2);
    // for(auto &&i: edges){
    //     cout<<i<<" ";
    // }
    // cout<<"\n";
    
    return 0;
}