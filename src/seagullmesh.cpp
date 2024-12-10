#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_mesh(py::module&);
void init_properties(py::module&);
void init_io(py::module&);
void init_geodesic_distances(py::module&);

void init_connected(py::module&);
void init_meshing(py::module&);
void init_corefine(py::module&);
void init_locate(py::module&);
void init_parametrize(py::module&);
void init_triangulate(py::module&);
void init_border(py::module&);
void init_simplification(py::module&);
void init_skeletonization(py::module&);
void init_poisson_reconstruct(py::module&);
void init_alpha_wrapping(py::module&);

PYBIND11_MODULE(_seagullmesh, m) {
    m.doc() = "";
    init_mesh(m);
//    init_properties(m);
//    init_io(m);
//    init_geodesic_distances(m);
//    init_locate(m);
//    init_connected(m);
//    init_border(m);
//    init_skeletonization(m);
//    init_simplification(m);
//    init_parametrize(m);
//    init_poisson_reconstruct(m);
//    init_alpha_wrapping(m);

    init_meshing(m);
//    init_corefine(m);
//    init_triangulate(m);


}
