#include "csrFilegen.hpp"

int main(int argc, char const *argv[])
{
    generateLargeGraph(100000,0.05f,"graph-100T.gl");
    return 0;
}
