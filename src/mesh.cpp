#include "seagullmesh.hpp"
#include "util.hpp"
#include <CGAL/boost/graph/generators.h>

namespace PMP = CGAL::Polygon_mesh_processing;

template<typename T>
void define_simple_type_3(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<double, double, double>())
        .def("__eq__", [](const T& self, const T& other) {return self == other;})
    ;
}

template<typename T>
void define_simple_type_2(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<double, double>())
        .def("__eq__", [](const T& self, const T& other) {return self == other;})
    ;
}

template<typename Idx>
py::array_t<bool> broadcast_equals(const std::vector<Idx>& idxs, const Idx& idx, bool invert = false) {
    py::ssize_t n = idxs.size();
    py::array_t<bool> out({n});
    auto r = out.template mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) {
        r(i) = (idxs[i] == idx) ^ invert;
    }
    return out;
}

template<typename Idx>
py::array_t<bool> vector_equals(const std::vector<Idx>& self, const std::vector<Idx>& other, bool invert = false) {
    py::ssize_t n0 = self.size(), n1 = other.size();
    if (n0 != n1) { throw py::value_error("dimension mismatch"); }
    py::array_t<bool> out({n0});
    auto r = out.template mutable_unchecked<1>();
    for (int i = 0; i < n0; ++i) {
        r(i) = (self[i] == other[i]) ^ invert;
    }
    return out;
}

// Used for defining Vertex/Vertices, Face/Faces, etc.
template<typename Idx>
void define_indices(py::module &m, std::string idx_name, std::string idxs_name) {
    using size_type = typename Idx::size_type;
    using Idxs = typename std::vector<Idx>;

    // Vertex, Face, Edge, Halfedge
    py::class_<Idx>(m, idx_name.c_str())
        .def(py::init<size_type>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__eq__", [](const Idx idx, const Idxs& idxs) { return broadcast_equals(idxs, idx); })
        .def("__ne__", [](const Idx idx, const Idxs& idxs) { return broadcast_equals(idxs, idx, true); })
        .def("__hash__", [](const Idx& idx) {
            // Surface_mesh/include/CGAL/Surface_mesh/Surface_mesh.h Line 307 -- just returns the uint32
            return hash_value(idx);
        })
        .def("to_int", [](const Idx& idx) { return size_type(idx); })
    ;

    // Vertices, Faces, Edges, Halfedges
    py::class_<Idxs>(m, idxs_name.c_str())
        .def(py::init<const std::vector< Idx >& >())
        .def(py::init([](py::list list) {
            Idxs idxs;
            idxs.reserve(list.size());
            for (py::handle obj : list) {
                idxs.emplace_back(obj.cast<Idx>());
            }
            return idxs;
        }))
        .def("__len__", [](const Idxs& idxs) { return idxs.size(); })
        .def("__iter__", [](Idxs& idxs) {
                return py::make_iterator(idxs.begin(), idxs.end());
            }, py::keep_alive<0, 1>()  /* Keep vector alive while iterator is used */
        )
        // Normal indexing with an int
        .def("__getitem__", [](const Idxs& idxs, size_t i) {
            if (i >= idxs.size()) { throw py::index_error(); }
            return idxs[i];
        })
        // List slicing
        .def("__getitem__", [](const Idxs& idxs, py::slice slice) {
            py::ssize_t start, stop, step, slicelength;
            if (!slice.compute(idxs.size(), &start, &stop, &step, &slicelength)) {
                throw py::error_already_set();
            }
            Idxs out;
            out.reserve(slicelength);
            for (int i = 0; i < slicelength; ++i) {
                out.emplace_back(idxs[start]);
                start += step;
            }
            return out;
        })
        // Numpy-like indexing with ints
        .def("__getitem__", [](const Idxs& idxs, const py::array_t<size_type>& sub) {
            if (sub.ndim() != 1) { throw py::index_error("multi-dimensional indexing not supported"); }
            py::ssize_t n = sub.size();
            Idxs out;
            out.reserve(n);

            auto r = sub.template unchecked<1>();
            for (int i = 0; i < n; ++i) {
                out.emplace_back(idxs.at(r(i)));
            }
            return out;
        })
        // Numpy-like indexing with bools
        .def("__getitem__", [](const Idxs& idxs, const py::array_t<bool>& sub) {
            py::ssize_t n = sub.size();
            if (sub.ndim() != 1 || n != idxs.size()) {
                throw py::index_error("boolean index vector is the wrong size");
            }
            Idxs out;
            auto r = sub.template unchecked<1>();
            for (int i = 0; i < n; ++i) {
                if ( r(i) ) {
                    out.emplace_back(idxs[i]);
                }
            }
            return out;
        })
        // these_faces == those_faces -> vector<bool>
        .def("__eq__", [](const Idxs& self, const Idxs& other) { return vector_equals(self, other); })
        // these_faces == that_face -> vector<bool>
        .def("__eq__", [](const Idxs& self, const Idx other) { return broadcast_equals(self, other); })
        // these_faces != those_faces -> vector<bool>
        .def("__ne__", [](const Idxs& self, const Idxs& other) { return vector_equals(self, other, true); })
        // these_faces != that_face -> vector<bool>
        .def("__ne__", [](const Idxs& self, const Idx other) { return broadcast_equals(self, other, true); })

        // casting the descriptors to bare uint32s for debugging
        .def("to_ints", [](const Idxs& idxs) {
            const py::ssize_t n = idxs.size();
            py::template array_t<size_type> out({n});
            auto r = out.template mutable_unchecked<1>();
            for (int i = 0; i < n; ++i) {
                r(i) = size_type(idxs[i]);
            }
            return out;
        })
        .def("from_ints", [](const py::array_t<size_type>& idxs) {
            if (idxs.ndim() != 1) {
                throw py::index_error();
            }
            py::ssize_t n = idxs.size();
            Idxs out;
            out.reserve(n);
            auto r = idxs.template unchecked<1>();
            for (int i = 0; i < n; ++i) {
                out.emplace_back(V(r(i)));
            }
            return out;
        })
    ;
}

template<typename Idx, typename IdxRange>
std::vector<Idx> indices_from_range(typename Idx::size_type n, const IdxRange idx_range) {
    std::vector<Idx> idxs;
    idxs.reserve(n);
    for (Idx idx : idx_range) {
        idxs.emplace_back(idx);
    }
    return idxs;
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
    // define_indices<H>(sub, "Halfedge", "Halfedges");

    py::class_<Mesh3>(sub, "Mesh3")
        .def(py::init<>())
        .def(py::init<const Mesh3&>())
        .def(py::self += py::self)

        .def_property_readonly("has_garbage", [](const Mesh3& mesh) {return mesh.has_garbage();})
        .def("collect_garbage", [](Mesh3& mesh) {mesh.collect_garbage();})

        .def_property_readonly("null_vertex", [](const Mesh3& mesh) { return mesh.null_vertex(); })
        .def_property_readonly("null_face", [](const Mesh3& mesh) { return mesh.null_face(); })
        .def_property_readonly("null_edge", [](const Mesh3& mesh) { return mesh.null_edge(); })
        .def_property_readonly("null_halfedge", [](const Mesh3& mesh) { return mesh.null_halfedge(); })
        
        .def_property_readonly("is_valid", [](const Mesh3& mesh) { return mesh.is_valid(false); })
        .def_property_readonly("n_vertices", [](const Mesh3& mesh) { return mesh.number_of_vertices(); })
        .def_property_readonly("n_faces", [](const Mesh3& mesh) { return mesh.number_of_faces(); })
        .def_property_readonly("n_edges", [](const Mesh3& mesh) { return mesh.number_of_edges(); })
        .def_property_readonly("n_halfedges", [](const Mesh3& mesh) { return mesh.number_of_halfedges(); })
        .def_property_readonly("points", [](const Mesh3& mesh) { return mesh.points(); })

        .def_property_readonly("vertices", [](const Mesh3& mesh) {
            return indices_from_range<V, Mesh3::Vertex_range>(mesh.number_of_vertices(), mesh.vertices());
        })
        .def_property_readonly("faces", [](const Mesh3& mesh) {
            return indices_from_range<F, Mesh3::Face_range>(mesh.number_of_faces(), mesh.faces());
        })
        .def_property_readonly("edges", [](const Mesh3& mesh) {
            return indices_from_range<E, Mesh3::Edge_range>(mesh.number_of_edges(), mesh.edges());
        })
//        .def_property_readonly("halfedges", [](const Mesh3& mesh) {
//            return indices_from_range<H, Mesh3::Halfedge_range>(mesh.number_of_halfedges(), mesh.halfedges());
//        })
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
