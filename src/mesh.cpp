#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/Surface_mesh/IO/OFF.h>
#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/boost/graph/generators.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Bbox_3 BBox3;
typedef CGAL::Aff_transformation_3<Kernel> Transform3;

typedef CGAL::dynamic_vertex_property_t<size_t>                           VertexIndex;
typedef typename boost::property_map<Mesh3, VertexIndex>::const_type      VertexIndexMap;


Transform3 array_to_transform3(const py::array_t<double>& transform) {
    auto r = transform.unchecked<2>();
    if (r.shape(0) != r.shape(1)) {
        throw py::value_error("Must be a square matrix");
    }
    if (r.shape(0) == 3) {
        return Transform3(
            r(0, 0), r(0, 1), r(0, 2),
            r(1, 0), r(1, 1), r(1, 2),
            r(2, 0), r(2, 1), r(2, 2)
        );
    } else if (r.shape(0) == 4) {
        return Transform3(
            r(0, 0), r(0, 1), r(0, 2), r(0, 3),
            r(1, 0), r(1, 1), r(1, 2), r(1, 3),
            r(2, 0), r(2, 1), r(2, 2), r(2, 3),
                                       r(3, 3)
        );
    } else {
        throw py::value_error("Must be a 3x3 or 4x4 matrix");
    }
}

VertexIndexMap build_vertex_index_map(const Mesh3& mesh) {
    VertexIndexMap vim = get(VertexIndex(), mesh);
    size_t i = 0;
    for (const V v : mesh.vertices()) {
        put(vim, v, i++);
    }
    return vim;
}


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

// Used for defining Vertex/Vertices, Face/Faces, etc.
template<typename Idx>
void define_indices(py::module &m, std::string idx_name, std::string idxs_name) {
    using size_type = typename Idx::size_type;
    using Idxs = typename std::vector<Idx>;

    py::class_<Idx>(m, idx_name.c_str())
        .def(py::init<size_type>())
//        .def("__eq__", [](const Idx& self, const Idx& other) {return self == other;})
//        .def("__ne__", [](const Idx& self, const Idx& other) {return self == other;})
        .def("to_int", [](const Idx& idx) {return size_type(idx);})
    ;
    py::class_<Idxs>(m, idxs_name.c_str())
        // Numpy-like indexing with ints
        .def("__getitem__", [](const Idxs& idxs, const py::array_t<size_type>& sub) {
            if (sub.ndim() != 1) {
                throw py::index_error();
            }
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
                throw py::index_error();
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
        .def("__eq__", [](const Idxs& idxs, const Idxs& other) {
            if (idxs.size() != other.size()) {
                return false;
            }
            for (int i = 0; i < idxs.size(); ++i) {
                if ( idxs[i] != other[i] ) {
                    return false;
                }
            }
            return true;
        })
        .def("__eq__", [](const Idxs& idxs, const Idx& other) {
            py::ssize_t n = idxs.size();
            py::array_t<bool> out({n});
            auto r = out.template mutable_unchecked<1>();
            for (int i = 0; i < n; ++i) {
                r(i) = idxs[i] == other;
            }
            return out;
        })
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

    py::implicitly_convertible<py::list, Idxs>();
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
    define_indices<H>(sub, "Halfedge", "Halfedges");

    sub.def("polygon_soup_to_mesh3", [](
                py::array_t<double> &points,
                std::vector<std::vector<size_t>>& faces,
                const bool orient
        ) {
            Mesh3 mesh;
            std::vector<Point3> vertices = array_to_points_3(points);

            if (orient) {
                bool success = PMP::orient_polygon_soup(vertices, faces);
                if (!success) {
                    throw std::runtime_error("Polygon orientation failed");
                }
            }
            PMP::polygon_soup_to_polygon_mesh(vertices, faces, mesh);
            return mesh;
        })
        .def("load_mesh_from_file", [](const std::string filename) {
            Mesh3 mesh;
            if(!CGAL::IO::read_polygon_mesh(filename, mesh)) {
                throw std::runtime_error("Failed to load mesh");
            }
            return mesh;
        })
    ;

    py::class_<BBox3>(sub, "BoundingBox3", py::module_local())
        .def(py::init<double, double, double, double, double, double>())
        .def_property_readonly("x_min", &BBox3::xmin)
        .def_property_readonly("x_max", &BBox3::xmax)
        .def_property_readonly("y_min", &BBox3::ymin)
        .def_property_readonly("y_max", &BBox3::ymax)
        .def_property_readonly("x_min", &BBox3::zmin)
        .def_property_readonly("z_max", &BBox3::zmax)
        .def("diagonal", [](const BBox3& bbox) {
            return std::sqrt(
                CGAL::square(bbox.xmax() - bbox.xmin()) +
                CGAL::square(bbox.ymax() - bbox.ymin()) +
                CGAL::square(bbox.zmax() - bbox.zmin())
            );
        })
    ;

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
        .def_property_readonly("halfedges", [](const Mesh3& mesh) {
            return indices_from_range<H, Mesh3::Halfedge_range>(mesh.number_of_halfedges(), mesh.halfedges());
        })

        .def("bounding_box", [](const Mesh3& mesh) {
            return PMP::bbox(mesh);
        })
        .def("transform", [](Mesh3& mesh, const py::array_t<double>& transform) {
            Transform3 t = array_to_transform3(transform);
            auto points = mesh.points();
            for (V v : mesh.vertices() ) {
                points[v] = t(points[v]);
            }
        })
        .def("edge_soup", [](const Mesh3& mesh) {
            VertexIndexMap vim = build_vertex_index_map(mesh);
            const size_t ne = mesh.number_of_edges();
            py::array_t<size_t> verts({ne, size_t(2)});
            auto r = verts.mutable_unchecked<2>();
            size_t i = 0;
            for (E e : mesh.edges()) {
                for (size_t j = 0; j < 2; ++j) {
                    r(i, j) = get(vim, mesh.vertex(e, j));
                }
                ++i;
            }
            return verts;
        })
        .def("triangle_soup", [](const Mesh3& mesh) {
            VertexIndexMap vim = build_vertex_index_map(mesh);
            const size_t nf = mesh.number_of_faces();
            py::array_t<size_t> verts({nf, size_t(3)});
            auto r = verts.mutable_unchecked<2>();
            size_t i = 0;
            for (F f : mesh.faces()) {
                size_t j = 0;
                for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
                    r(i, j++) = get(vim, target(h, mesh));
                }
                ++i;
            }
            return verts;
        })
        .def("vertices_to_faces", [](const Mesh3& mesh, const std::vector<V>& verts) {
            std::set<F> faces;
            for (V v : verts) {
                // mesh.halfedge(v) returns an incoming halfedge of vertex v
                for (F f : faces_around_target(mesh.halfedge(v), mesh)) {
                    if (f != mesh.null_face()) {
                        faces.insert(f);
                    }
                }
            }
            return std::vector<F>(faces.begin(), faces.end());
        })
        .def("vertices_to_edges", [](const Mesh3& mesh, const std::vector<V>& verts) {
            std::set<E> edges;
            for (V v : verts) {
                for (H h : halfedges_around_source(v, mesh)) {
                    edges.insert(mesh.edge(h));
                }
            }
            return std::vector<E>(edges.begin(), edges.end());
        })
        .def("faces_to_edges", [](const Mesh3& mesh, const std::vector<F>& faces) {
            std::set<E> edges;
            for (F f : faces) {
                for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
                    edges.insert(mesh.edge(h));
                }
            }
            return std::vector<E>(edges.begin(), edges.end());
        })
        .def("vertex_degrees", [](const Mesh3& mesh, const std::vector<V>& verts) {
            size_t n = verts.size();
            py::array_t<Mesh3::size_type> degrees({py::ssize_t(n)});
            auto r = degrees.mutable_unchecked<1>();
            for (size_t i = 0; i < n; ++i) {
                r(i) = mesh.degree(verts[i]);
            }
            return degrees;
        })
        .def("to_polygon_soup", [](const Mesh3& mesh) {
            std::vector<Point3> verts;
            std::vector<std::vector<size_t>> faces;
            PMP::polygon_mesh_to_polygon_soup(mesh, verts, faces);
            auto points = points_to_array(verts);

            // Convert vector<vector<size_t>> to array
            const size_t nf = mesh.number_of_faces();
            py::array_t<size_t, py::array::c_style> faces_out({nf, size_t(3)});
            auto rf = faces_out.mutable_unchecked<2>();
            for (size_t i = 0; i < nf; i++) {
                for (size_t j = 0; j < 3; j++) {
                    rf(i, j) = faces[i][j];
                }
            }
            return std::make_tuple(points, faces_out);
        })
        .def("face_normals", [](const Mesh3& mesh, const std::vector<F>& faces) {
            return map_indices_to_vector<3, F, Vector3>(
                faces, [&mesh](F f) {return PMP::compute_face_normal(f, mesh);}
            );
        })
        .def("vertex_normals", [](const Mesh3& mesh, const std::vector<V>& verts) {
            return map_indices_to_vector<3, V, Vector3>(
                verts, [&mesh](V v) {return PMP::compute_vertex_normal(v, mesh);}
            );
        })
        .def("volume", [](const Mesh3& mesh) {return PMP::volume(mesh);})
        .def("area", [](const Mesh3& mesh) {return PMP::area(mesh);})
        .def("estimate_geodesic_distances", [](const Mesh3& mesh, Mesh3::Property_map<V, double>& distances, V source) {
            CGAL::Heat_method_3::estimate_geodesic_distances(mesh, distances, source);
        })
        .def("estimate_geodesic_distances", [](
                const Mesh3& mesh, Mesh3::Property_map<V, double>& distances, const std::vector<V>& sources) {
            CGAL::Heat_method_3::estimate_geodesic_distances(mesh, distances, sources);
        })

        .def("write_ply", [](Mesh3& mesh, std::string file) {
            std::ofstream out(file, std::ios::binary);
            CGAL::IO::set_binary_mode(out);
            bool success = CGAL::IO::write_PLY(out, mesh, "");
            if (!success) {
                throw std::runtime_error("writing failed");
            }
        })
        .def("write_off", [](Mesh3& mesh, std::string file) {
            std::ofstream out(file);
            bool success = CGAL::IO::write_OFF(out, mesh);
            if (!success) {
                throw std::runtime_error("writing failed");
            }
        })
        .def("face_areas", [](const Mesh3& mesh, const std::vector<F>& faces) {
            return map_indices_to_scalar<F, double>(
                faces, [&mesh](F f){ return PMP::face_area(f, mesh);}
            );
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
