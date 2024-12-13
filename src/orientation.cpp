#include "seagullmesh.hpp"

#include <CGAL/Polygon_mesh_processing/orientation.h>

namespace PMP = CGAL::Polygon_mesh_processing;

void init_orientation(py::module &m) {
    m.def_submodule("orientation")
        .def("reverse_face_orientations", [](Mesh3& mesh, const Indices<F>& faces) {
            PMP::reverse_face_orientations(faces.to_vector(), mesh);
        })
        .def("reverse_face_orientations", [](Mesh3& mesh) {
            PMP::reverse_face_orientations(mesh);
        })
    ;
}