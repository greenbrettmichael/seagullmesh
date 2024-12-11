#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/boost/graph/generators.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/Surface_mesh/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>

namespace PMP = CGAL::Polygon_mesh_processing;

template<typename T>
void define_simple_type_3(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<double, double, double>())
        .def(py::self == py::self)
        .def(py::self != py::self)
    ;
}

template<typename T>
void define_simple_type_2(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<double, double>())
        .def(py::self == py::self)
        .def(py::self != py::self)
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

    py::class_<Indices<T>>(m, idxs_name.c_str())
        .def(py::init< py::array_t<size_type> >() )
        .def(py::init< const std::vector<T>& >() )
        .def("from_indices", [](const std::vector<T>& idxs) { return Indices<T>(idxs); })
        .def_property_readonly("indices", [](const Indices<T>& idxs) { return idxs.get_indices(); })
    ;
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

        .def_static("null_vertex", &Mesh3::null_vertex)
        .def_static("null_face", &Mesh3::null_face)
        .def_static("null_edge", &Mesh3::null_edge)
        .def_static("null_halfedge", &Mesh3::null_halfedge)
        
        .def_property_readonly("is_valid", [](const Mesh3& mesh) { return mesh.is_valid(false); })
        .def_property_readonly("n_vertices", [](const Mesh3& mesh) { return mesh.number_of_vertices(); })
        .def_property_readonly("n_faces", [](const Mesh3& mesh) { return mesh.number_of_faces(); })
        .def_property_readonly("n_edges", [](const Mesh3& mesh) { return mesh.number_of_edges(); })
        .def_property_readonly("n_halfedges", [](const Mesh3& mesh) { return mesh.number_of_halfedges(); })
        .def_property_readonly("points", [](const Mesh3& mesh) { return mesh.points(); })

        .def_property_readonly("vertices", [](const Mesh3& mesh) {
            auto vs = mesh.vertices();
            const std::vector<V> idxs(vs.begin(), vs.end());
            return Indices<V>(idxs);
        })
        .def_property_readonly("faces", [](const Mesh3& mesh) {
            auto fs = mesh.faces();
            const std::vector<F> idxs(fs.begin(), fs.end());
            return Indices<F>(idxs);
        })
        .def_property_readonly("edges", [](const Mesh3& mesh) {
            auto es = mesh.edges();
            const std::vector<E> idxs(es.begin(), es.end());
            return Indices<E>(idxs);
        })
        .def_property_readonly("halfedges", [](const Mesh3& mesh) {
            auto hs = mesh.halfedges();
            const std::vector<H> idxs(hs.begin(), hs.end());
            return Indices<H>(idxs);
        })
        .def("icosahedron", [](Mesh3& mesh, double x, double y, double z, double r){
            CGAL::make_icosahedron(mesh, Point3(x, y, z), r);
        })
        .def("tetrahedron", [](Mesh3& mesh, const py::array_t<double>& points){
            std::vector<Point3> pts = array_to_points_3(points);
            CGAL::make_tetrahedron(pts.at(0), pts.at(1), pts.at(2), pts.at(3), mesh);
        })
        .def("triangle", [](Mesh3& mesh, const py::array_t<double>& points){
            std::vector<Point3> pts = array_to_points_3(points);
            CGAL::make_triangle(pts.at(0), pts.at(1), pts.at(2), mesh);
        })
        .def("pyramid", [](
                Mesh3& mesh,
                Mesh3::size_type n_base_verts,
                const Point3 base_center,
                double height,
                double radius,
                bool closed
            ){
            CGAL::make_pyramid(n_base_verts, mesh, base_center, height, radius, closed);
        })
    ;
}
