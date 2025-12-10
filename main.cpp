// test code 
#include <iostream>
#include "MemoryMap.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    cout<<"Test for memory map"<<endl;
    MemoryMap file("/home/krish/graphlite/test.bin");
    cout<<"file initiated"<<endl;
    
    int* arrData = (int*)file.get_data();
    int length = file.get_size()/sizeof(arrData[0]);

    cout<<"Array Data in binary file"<<endl;
    for (size_t i = 0; i < length; i++)
    {
        cout<<i<<" "<<arrData[i]<<"\n";
    }
    cout<<endl;

    return 0;
}
