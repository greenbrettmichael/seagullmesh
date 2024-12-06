#include "seagullmesh.hpp"
#include "util.hpp"
#include <boost/property_map/property_map.hpp>

#include <CGAL/boost/graph/Face_filtered_graph.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

typedef CGAL::Face_filtered_graph<Mesh3>        FilteredMesh;

typedef Mesh3::Property_map<F, bool>            FaceBool;
typedef Mesh3::Property_map<E, bool>            EdgeBool;

// I think CGAL's happy with any int type here as long as it's convertible to std::size but I'm not sure
typedef F::size_type                            FaceIdx;
typedef Mesh3::Property_map<F, FaceIdx>         FacePatchMap;

namespace PMP = CGAL::Polygon_mesh_processing;

void init_connected(py::module &m) {
    py::module sub = m.def_submodule("connected");

    sub
        .def("label_connected_components", [](const Mesh3& mesh, FacePatchMap& face_patch, EdgeBool& edge_is_constrained) {
            // Returns the number of connected components
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained);
            return PMP::connected_components(mesh, face_patch, params);
        })
        .def("label_selected_face_patches", [](Mesh3& mesh, const std::vector<F> faces, FacePatchMap& face_patch) {
            FilteredMesh filtered(mesh, faces);

            std::map<F, F::size_type> filtered_face_patch;
            boost::associative_property_map< std::map<F, F::size_type> > filtered_face_patch_map(filtered_face_patch);
            auto n_components = PMP::connected_components(filtered, filtered_face_patch_map);

            for (auto const& [f, i] : filtered_face_patch) {
                face_patch[f] = i + 1;
            }
            return n_components;
        })
        .def("keep_connected_components", [](Mesh3& mesh, const std::vector<F::size_type>& components_to_keep, const FacePatchMap& components) {
            PMP::keep_connected_components(mesh, components_to_keep, components);
        })
        .def("connected_component", [](const Mesh3& mesh, F seed_face, EdgeBool& edge_is_constrained) {
            std::vector<F> out;
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained);
            PMP::connected_component(seed_face, mesh, std::back_inserter(out), params);
            return out;
        })
        .def("remove_connected_faces", [](Mesh3& mesh, const std::vector<F>& faces) {
            PMP::remove_connected_components(mesh, faces);
        })
        .def("remove_connected_face_patches", [](Mesh3& mesh, const std::vector<FaceIdx>& components_to_remove, const FacePatchMap& components) {
            PMP::remove_connected_components(mesh, components_to_remove, components);
        })
    ;
}