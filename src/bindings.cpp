#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include "Graphzero.hpp"
#include "csrFilegen.hpp"
#include <vector>
namespace nb = nanobind;


NB_MODULE(graphzero,m) {
    m.doc() = "graphzero: High-performance C++ Graph Engine";

    nb::class_<Graphzero>(m, "Graph")
        .def(nb::init<const char*>(), // constructor
R"doc(Graph Class contains the graphfile and its relevant functions and methods.
It holds the mmap / zero-copy memory.
Args:
    filename (str): either absolute path or relative path (depends on the current working directory).
Returns:
    Graph class instance.
)doc",
            nb::arg("filename")
        ) 
        .def_rw("num_nodes",&Graphzero::num_nodes)
        .def_rw("num_edges",&Graphzero::num_edges)


        .def("get_degree", [](Graphzero &self, size_t node_id) {
            return self.get_storage()->get_degree(node_id);
        },
R"doc(Get the degree of a node.
Args:
    node_id (int)
Returns:
    degree (int)
)doc",
            nb::arg("node_id")
        )
        
        .def("get_neighbours", [](Graphzero &self, size_t node_id) {
            auto edges = self.get_storage()->get_edges(node_id);

            // Return a zero-copy view into the underlying edge buffer and keep
            // the Graph object alive as the owner.
            return nb::ndarray<nb::numpy, size_t, nb::shape<1>>(
                const_cast<size_t*>(edges.data()), // pointer to data
                { edges.size() },                 // shape
                nb::cast(self)                    // owner: keep Graph instance alive
            );
        },
            nb::keep_alive<0,1>(),
R"doc(Returns the neighbours of a node.
Args:
    node_id (int)
Returns:
    1-D numpy ndarray of neighbour node IDs (dtype: platform-size integer).
)doc",
            nb::arg("node_id")
        )
        
        .def("batch_random_walk", [](Graphzero &self, const std::vector<size_t>& startNodes, size_t walkLength, float p, float q) {
            auto tmp = self.batchRandomWalk(startNodes, walkLength, p, q);
            // convert to int64 for Python-facing ndarray (explicit copy)
            std::vector<int64_t>* walkData = new std::vector<int64_t>(tmp.begin(), tmp.end());

            // owner capsule will free the vector when Python releases it
            nb::capsule owner(walkData, [](void* p) noexcept {
                delete static_cast<std::vector<int64_t>*>(p);
            });

            return nb::ndarray<nb::numpy, int64_t, nb::shape<2>>(
                reinterpret_cast<int64_t*>(walkData->data()),
                {startNodes.size(),walkLength },
                owner
            );
        },
R"doc(Performs 2nd-order random walks (Node2Vec style).

Args:
    start_nodes (list): list of starting nodes.
    walk_length (int): how long walks should be (e.g. 10).
    p (float): Return parameter; Low = keeps walk local (BFS-like).
    q (float): In-out parameter; Low = explores far away (DFS-like).

Returns:
    ndarray of shape (len(start_nodes), walk_length) with dtype np.int64.
)doc",
            // DEFINING ARGUMENTS & DEFAULTS
            nb::arg("start_nodes"),
            nb::arg("walk_length"),
            nb::arg("p") = 1.0f,  // Default p=1.0
            nb::arg("q") = 1.0f   // Default q=1.0
        )

        .def("batch_random_walk_uniform", [](Graphzero &self, const std::vector<size_t>& startNodes, size_t walkLength) {
            auto tmp = self.batchRandomUniformWalk(startNodes, walkLength);
            std::vector<int64_t>* walkData = new std::vector<int64_t>(tmp.begin(), tmp.end());

            nb::capsule owner(walkData, [](void* p) noexcept {
                delete static_cast<std::vector<int64_t>*>(p);
            });

            return nb::ndarray<nb::numpy, int64_t, nb::shape<2>>(
                reinterpret_cast<int64_t*>(walkData->data()),
                {startNodes.size(),walkLength },
                owner
            );
        },
R"doc(Performs uniform random walks.

Args:
    start_nodes (list): list of starting nodes.
    walk_length (int): how long walks should be (e.g. 10).

Returns:
    ndarray of shape (len(start_nodes), walk_length) with dtype np.int64.
)doc",
            // DEFINING ARGUMENTS & DEFAULTS
            nb::arg("start_nodes"),
            nb::arg("walk_length")
        )

        .def("batch_random_fanout", [](Graphzero &self, const std::vector<size_t>& startNodes, size_t K) {
            auto tmp = self.batchRandomFanout(startNodes, K);
            std::vector<int64_t>* walkData = new std::vector<int64_t>(tmp.begin(), tmp.end());

            nb::capsule owner(walkData, [](void* p) noexcept {
                delete static_cast<std::vector<int64_t>*>(p);
            });

            return nb::ndarray<nb::numpy, int64_t, nb::shape<2>>(
                reinterpret_cast<int64_t*>(walkData->data()),
                {startNodes.size(),K },
                owner
            );
        },
R"doc(Performs uniform random fanout sampling.

Args:
    start_nodes (list): list of starting nodes.
    K (int): how many neighbours to sample.

Returns:
    ndarray of shape (len(start_nodes), K) with dtype np.int64.
)doc",
            // DEFINING ARGUMENTS & DEFAULTS
            nb::arg("start_nodes"),
            nb::arg("K")
        )
        
        .def("sample_neighbours", [](Graphzero &self, size_t startNode, size_t K) {
            auto tmp = self.ReservoirSampling(startNode, K);
            std::vector<int64_t>* walkData = new std::vector<int64_t>(tmp.begin(), tmp.end());

            nb::capsule owner(walkData, [](void* p) noexcept {
                delete static_cast<std::vector<int64_t>*>(p);
            });

            return nb::ndarray<nb::numpy, int64_t, nb::shape<1>>(
                reinterpret_cast<int64_t*>(walkData->data()),
                {walkData->size() },
                owner
            );
        },
R"doc(Performs uniform random neighbour sampling for a node.

Args:
    start_node (int): node id to sample from.
    K (int): how many neighbours to sample.

Returns:
    1-D ndarray with up to K neighbour ids (dtype np.int64).
)doc",
            // DEFINING ARGUMENTS & DEFAULTS
            nb::arg("start_node"),
            nb::arg("K")
        )

        // serialization (Pack)
        .def("__getstate__", [](const Graphzero &g){

            return nb::make_tuple(g.filename); // only filename required to rebuild the object
        })
        // deserialization (unpack)
        .def("__setstate__",[](nb::tuple &t){
            
            if (t.size() != 1) 
                throw std::runtime_error("Invalid state!");
            
            std::string filename = nb::cast<std::string>(t[0]);

            // create new c++ object using the filename
            return new Graphzero(filename.c_str());
        })
        ;
    
    // convert csv to gl 
    m.def("convert_csv_to_gl", &convert_csv,
        "Convert a CSV edge list to GraphZero binary format (.gl)",
        nb::arg("csv_path"), 
        nb::arg("out_path"), 
        nb::arg("directed") = false,
        nb::call_guard<nb::gil_scoped_release>());
}
