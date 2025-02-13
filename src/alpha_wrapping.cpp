#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/alpha_wrap_3.h>


void init_alpha_wrapping(py::module &m) {
    m.def_submodule("alpha_wrapping")
        .def("wrap_mesh", [](const Mesh3& mesh, const double alpha, const double offset) {
            Mesh3 out;
            CGAL::alpha_wrap_3(mesh, alpha, offset, out);
            return out;
        })
        .def("wrap_points", [](const py::array_t<double>& points, const double alpha, const double offset) {
            std::vector<Point3> pts = array_to_points_3(points);
            Mesh3 out;
            CGAL::alpha_wrap_3(pts, alpha, offset, out);
            return out;
        })
        .def("wrap_faces", [](
                const py::array_t<double>& points,
                const std::vector<std::vector<size_t>>& faces,
                const double alpha,
                const double offset
            ) {
                std::vector<Point3> pts = array_to_points_3(points);
                Mesh3 out;
                CGAL::alpha_wrap_3(pts, faces, alpha, offset, out);
                return out;
            }
        )
    ;
}