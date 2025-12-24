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

    nb::class_<Graphzero>(m,"Graph")
        .def(nb::init<const char*>()) //constructor 
        
        .def("get_degree",&Graphzero::get_degree,"Get the degree of a node",nb::arg("node_id"))
        
        .def("get_neighbours",[](Graphzero &self,size_t node_id) {
            
            auto edges = self.get_storage()->get_edges(node_id);

            return nb::ndarray<nb::numpy, size_t, nb::shape<1> >(
                edges.data(),       //pointer to data 
                {edges.size() }    // shape , size of array
                // nb::cast(self)      // Owner 
            );
        }, 
            nb::keep_alive<0,1>(),
            "Returns the neighbours of a node",
            nb::arg("node_id")
        )
        
        .def("batch_random_walk",[](Graphzero &self, const std::vector<size_t>& startNodes, size_t walkLength,float p, float q){

            std::vector<size_t>* walkData = new std::vector<size_t>(self.batchRandomWalk(startNodes,walkLength,p,q));

            // small python object that is owner of return array/vector
            nb::capsule owner(walkData,[](void* p) noexcept{
                delete (std::vector<size_t> *) p;
            });

            return nb::ndarray<nb::numpy, int64_t, nb::shape<2> >(
                reinterpret_cast<int64_t*>(walkData->data()),
                {startNodes.size(),walkLength},
                owner
            );
        },
            // DEFINING ARGUMENTS & DEFAULTS
            nb::arg("start_nodes"), 
            nb::arg("walk_length"), 
            nb::arg("p") = 1.0f,  // Default p=1.0
            nb::arg("q") = 1.0f   // Default q=1.0
        )

        .def("batch_random_walk_uniform",[](Graphzero &self, const std::vector<size_t>& startNodes, size_t walkLength){

            //uniform sampling 
            std::vector<size_t>* walkData = new std::vector<size_t>(self.batchRandomUniformWalk(startNodes,walkLength));

            // small python object that is owner of return array/vector
            nb::capsule owner(walkData,[](void* p) noexcept{
                delete (std::vector<size_t> *) p;
            });

            return nb::ndarray<nb::numpy, int64_t, nb::shape<2> >(
                reinterpret_cast<int64_t*>(walkData->data()),
                {startNodes.size(),walkLength},
                owner
            );
        },
            // DEFINING ARGUMENTS & DEFAULTS
            nb::arg("start_nodes"), 
            nb::arg("walk_length")
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
