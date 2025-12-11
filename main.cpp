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
    
    generateBinary(nnzRowData,"./nnzRow.bin");
    generateBinary(colPtrData,"./colPtr.bin");

    cout<<"FILE GEN COMPLETE\n";
    sleep(2);

    CSR storage("./nnzRow.bin","./colPtr.bin");

    span<size_t> edges = storage.get_edges(5); 
    cout<<"Edges for 5:\n";
    for(auto& i: edges){
        cout<<i<<" ";
    }
    cout<<"\n";
    
    return 0;
}