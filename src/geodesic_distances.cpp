#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>

void init_geodesic_distances(py::module &m) {
    m.def_submodule("geodesic_distances")
        .def("estimate_geodesic_distances", [](const Mesh3& mesh, Mesh3::Property_map<V, double>& distances, V source) {
            CGAL::Heat_method_3::estimate_geodesic_distances(mesh, distances, source);
        })
        .def("estimate_geodesic_distances", [](
                const Mesh3& mesh, Mesh3::Property_map<V, double>& distances, const Indices<V>& sources) {
            CGAL::Heat_method_3::estimate_geodesic_distances(mesh, distances, sources.to_vector());
        })
    ;
}