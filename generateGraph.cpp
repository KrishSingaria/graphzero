#include "src/csrFilegen.hpp"

int main(int argc, char const *argv[])
{
    generateLargeGraph(100000,0.05f,"graph-100T.gl");
    return 0;
}

// csv_to_gl
// int main(int argc, char** argv) {
//     if (argc < 3) {
//         std::cerr << "Usage: ./converter <input.csv> <output.gl> [directed=0]" << std::endl;
//         return 1;
//     }
//     bool directed = (argc > 3 && std::string(argv[3]) == "1");
//     convert_csv(argv[1], argv[2], directed);
//     return 0;
// }