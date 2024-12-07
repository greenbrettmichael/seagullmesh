#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/Surface_mesh/IO/OFF.h>

#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>

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

void init_mesh(py::module &m) {
    py::module sub = m.def_submodule("misc");

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

    sub
        .def("polygon_soup_to_mesh3", [](
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
    ;
}