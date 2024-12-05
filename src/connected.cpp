#include "seagullmesh.hpp"
#include "util.hpp"
#include <boost/property_map/property_map.hpp>

#include <CGAL/boost/graph/Face_filtered_graph.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

typedef CGAL::Face_filtered_graph<Mesh3>         FilteredMesh;

typedef Mesh3::Property_map<F, bool>             FaceBool;
typedef Mesh3::Property_map<F, F::size_type>     FacePatchId;

namespace PMP = CGAL::Polygon_mesh_processing;

void init_connected(py::module &m) {
    py::module sub = m.def_submodule("connected");


    sub
        .def("label_connected_components", [](const Mesh3& mesh, FacePatchId& face_patch) {
            // Returns the number of connected components
            return PMP::connected_components(mesh, face_patch);
        })
        .def("label_selected_face_patches", [](Mesh3& mesh, const std::vector<F> faces, FacePatchId& face_patch) {
            FilteredMesh filtered(mesh, faces);

            std::map<F, F::size_type> filtered_face_patch;
            boost::associative_property_map< std::map<F, F::size_type> > filtered_face_patch_map(filtered_face_patch);
            auto n_components = PMP::connected_components(filtered, filtered_face_patch_map);

            for (auto const& [f, i] : filtered_face_patch) {
                face_patch[f] = i + 1;
            }

            return n_components;
        })
    ;
}