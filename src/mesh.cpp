#include "seagullmesh.hpp"
#include "util.hpp"

#include <pybind11/functional.h>

#include <CGAL/boost/graph/generators.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>

namespace PMP = CGAL::Polygon_mesh_processing;

template<typename T>
void define_simple_type_3(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<double, double, double>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__getitem__", [](const T& t, size_t i) {return t[i];})
        .def("__iter__", [](const T& t) { return py::make_iterator(t.cartesian_begin(), t.cartesian_end()); }, py::keep_alive<0, 1>())
    ;
}

template<typename T>
void define_simple_type_2(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<double, double>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__getitem__", [](const T& t, size_t i) {return t[i];})
        .def("__iter__", [](const T& t) { return py::make_iterator(t.cartesian_begin(), t.cartesian_end()); }, py::keep_alive<0, 1>())
    ;
}

// Used for defining Vertex/Vertices, Face/Faces, etc.
template<typename T>
void define_indices(py::module &m, std::string idx_name, std::string idxs_name) {
    using size_type = typename T::size_type;

    // Vertex, Face, Edge, Halfedge
    py::class_<T>(m, idx_name.c_str())
        .def(py::init<size_type>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__hash__", [](const T& idx) {
            // Surface_mesh/include/CGAL/Surface_mesh/Surface_mesh.h Line 307 -- just returns the uint32
            return hash_value(idx);
        })
        .def("to_int", [](const T& idx) { return size_type(idx); })
    ;

    // Vertices, Faces, Edges, Halfedges
    py::class_<Indices<T>>(m, idxs_name.c_str())
        .def(py::init< py::array_t<size_type> >() )
        .def(py::init< const std::vector<T>& >() )
        .def("from_indices", [](const std::vector<T>& idxs) { return Indices<T>(idxs); })
        .def_property_readonly("indices", [](const Indices<T>& idxs) { return idxs.get_indices(); })
        .def("__len__", [](const Indices<T>& idxs) { return idxs.size(); })
        .def("is_removed", [](const Indices<T>& idxs, const Mesh3& mesh) {
            return idxs.template map_to_array_of_scalars<bool>([&mesh](T idx) { return mesh.is_removed(idx); });
        })
        .def("is_valid", [](const Indices<T>& idxs, const Mesh3& mesh) {
            return idxs.template map_to_array_of_scalars<bool>([&mesh](T idx) { return mesh.is_valid(idx); });
        })
    ;

    // Also provide a module-level free function
    m.def("make_indices", [](const std::vector<T>& idxs){ return Indices<T>(idxs); });
}


void init_mesh(py::module &m) {
    py::module sub = m.def_submodule("mesh");

    define_simple_type_2<Point2>(sub, "Point2");
    define_simple_type_3<Point3>(sub, "Point3");
    define_simple_type_2<Vector2>(sub, "Vector2");
    define_simple_type_3<Vector3>(sub, "Vector3");

    define_indices<V>(sub, "Vertex", "Vertices");
    define_indices<F>(sub, "Face", "Faces");
    define_indices<E>(sub, "Edge", "Edges");
    define_indices<H>(sub, "Halfedge", "Halfedges");

    py::class_<Mesh3>(sub, "Mesh3")
        .def(py::init<>())
        .def(py::init<const Mesh3&>())
        .def(py::self += py::self)

        .def_property_readonly("has_garbage", [](const Mesh3& mesh) {return mesh.has_garbage();})
        .def("collect_garbage", [](Mesh3& mesh) {mesh.collect_garbage();})

        .def_property_readonly_static("null_vertex",    [](py::object /* self */) { return Mesh3::null_vertex(); })
        .def_property_readonly_static("null_face",      [](py::object /* self */) { return Mesh3::null_face(); })
        .def_property_readonly_static("null_edge",      [](py::object /* self */) { return Mesh3::null_edge(); })
        .def_property_readonly_static("null_halfedge",  [](py::object /* self */) { return Mesh3::null_halfedge(); })

        .def_property_readonly("n_vertices", &Mesh3::number_of_vertices)
        .def_property_readonly("n_faces", &Mesh3::number_of_faces)
        .def_property_readonly("n_edges", &Mesh3::number_of_edges)
        .def_property_readonly("n_halfedges", &Mesh3::number_of_halfedges)

        .def_property_readonly("points", [](const Mesh3& mesh) { return mesh.points(); })

        .def_property_readonly("vertices", [](const Mesh3& mesh) {
            return Indices<V>::from_range<Mesh3::Vertex_range>(mesh.number_of_vertices(), mesh.vertices());
        })
        .def_property_readonly("faces", [](const Mesh3& mesh) {
            return Indices<F>::from_range<Mesh3::Face_range>(mesh.number_of_faces(), mesh.faces());
        })
        .def_property_readonly("edges", [](const Mesh3& mesh) {
            return Indices<E>::from_range<Mesh3::Edge_range>(mesh.number_of_edges(), mesh.edges());
        })
        .def_property_readonly("halfedges", [](const Mesh3& mesh) {
            return Indices<H>::from_range<Mesh3::Halfedge_range>(mesh.number_of_halfedges(), mesh.halfedges());
        })
        .def("is_valid", [](const Mesh3& mesh, bool verbose) { return mesh.is_valid(verbose); })
        .def("is_closed", [](const Mesh3& mesh) { return CGAL::is_closed(mesh); })
    ;

    sub
        .def("add_icosahedron", [](Mesh3& mesh, double x, double y, double z, double r){
            CGAL::make_icosahedron(mesh, Point3(x, y, z), r);
        })
        .def("add_tetrahedron", [](Mesh3& mesh, const py::array_t<double>& points){
            std::vector<Point3> pts = array_to_points_3(points);
            CGAL::make_tetrahedron(pts.at(0), pts.at(1), pts.at(2), pts.at(3), mesh);
        })
        .def("add_triangle", [](Mesh3& mesh, const py::array_t<double>& points){
            std::vector<Point3> pts = array_to_points_3(points);
            CGAL::make_triangle(pts.at(0), pts.at(1), pts.at(2), mesh);
        })
        .def("add_pyramid", [](
                Mesh3& mesh,
                Mesh3::size_type n_base_verts,
                const Point3 base_center,
                double height,
                double radius,
                bool closed
            ){
            CGAL::make_pyramid(n_base_verts, mesh, base_center, height, radius, closed);
        })
        .def("add_grid", [](
                Mesh3& mesh,
                V::size_type ni,
                V::size_type nj,
                std::function<Point3 (V::size_type, V::size_type)> calculator,
                bool triangulated
        ){
            CGAL::make_grid(ni, nj, mesh, calculator, triangulated);
        })

    ;
}
