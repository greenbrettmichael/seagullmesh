#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/Surface_mesh/IO/OFF.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, uint32_t>  VertexIndexMap;


template<typename Edges>
py::array_t<size_t> edge_soup(const Mesh3& mesh, const Edges& edges, size_t n_edges, const VertexIndexMap& vidx) {
    py::array_t<size_t> out({n_edges, size_t(2)});
    auto r = out.mutable_unchecked<2>();
    size_t i = 0;
    for (E e : edges) {
        for (size_t j = 0; j < 2; ++j) {
            r(i, j) = vidx[mesh.vertex(e, j)];
        }
        ++i;
    }
    return out;
}

template<typename Faces>
py::array_t<size_t> triangle_soup(const Mesh3& mesh, const Faces& faces, size_t n_faces, const VertexIndexMap& vidx) {
    py::array_t<size_t> out({n_faces, size_t(3)});
    auto r = out.mutable_unchecked<2>();
    size_t i = 0;
    for (F f : faces) {
        size_t j = 0;
        for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
            r(i, j++) = vidx[target(h, mesh)];
        }
        ++i;
    }
    return out;
}


void init_io(py::module &m) {
    m.def_submodule("io")
        .def("polygon_soup_to_mesh3", [](
                py::array_t<double> &points,
                std::vector<std::vector<size_t>>& faces,
                const bool orient,
                const bool validate = false
        ) {
            Mesh3 mesh;
            std::vector<Point3> vertices = array_to_points_3(points);
            if (orient) {
                bool success = PMP::orient_polygon_soup(vertices, faces);
                if (!success) {
                    throw std::runtime_error("Polygon orientation failed");
                }
            } else if (validate) {
                if (!PMP::is_polygon_soup_a_polygon_mesh(faces)) {
                    throw std::runtime_error("Polygon soup is not a polygon mesh");
                }
            }
            PMP::polygon_soup_to_polygon_mesh(vertices, faces, mesh);
            return mesh;
        })
        .def("mesh3_to_polygon_soup", [](const Mesh3& mesh) {
            std::vector<Point3> verts;
            std::vector<std::vector<size_t>> faces;
            PMP::polygon_mesh_to_polygon_soup(mesh, verts, faces);
            auto points = points_to_array(verts);
            return std::make_tuple(points, faces);
        })
        .def("load_mesh_from_file", [](const std::string filename) {
            Mesh3 mesh;
            if(!CGAL::IO::read_polygon_mesh(filename, mesh)) {
                throw std::runtime_error("Failed to load mesh");
            }
            return mesh;
        })
        .def("point_soup", [](const Mesh3& mesh) {
            size_t n = mesh.number_of_vertices();
            auto pts = mesh.points();
            py::array_t<double> out({py::ssize_t(n), py::ssize_t(3)});
            auto r = out.mutable_unchecked<2>();
            size_t i = 0;

            for (V v : mesh.vertices()) {
                Point3 pt = pts[v];
                for (size_t j = 0; j < 3; ++j) {
                    r(i, j) = pt[j];
                }
                i++;
            }
            return out;
        })
        .def("edge_soup", [](const Mesh3& mesh, const VertexIndexMap& vidx) {
            return edge_soup<Mesh3::Edge_range>(mesh, mesh.edges(), mesh.number_of_edges(), vidx);
        })
        .def("edge_soup", [](const Mesh3& mesh, const Indices<E>& edges, const VertexIndexMap& vidx) {
            return edge_soup<std::vector<E>>(mesh, edges.to_vector(), edges.size(), vidx);
        })
        .def("triangle_soup", [](const Mesh3& mesh, const VertexIndexMap& vidx) {
            return triangle_soup<Mesh3::Face_range>(mesh, mesh.faces(), mesh.number_of_faces(), vidx);
        })
        .def("triangle_soup", [](const Mesh3& mesh, const Indices<F>& faces, const VertexIndexMap& vidx) {
            return triangle_soup<std::vector<F>>(mesh, faces.to_vector(), faces.size(), vidx);
        })
        .def("from_triangle_soup", [](const py::array_t<double>& points, const py::array_t<size_t, py::array::c_style>& faces) {
            Mesh3 mesh;
            auto rv = points.unchecked<2>();
            std::vector<V> verts;
            size_t nv = rv.shape(0);
            verts.reserve(nv);
            for (size_t i = 0; i < nv; ++i) {
                verts.emplace_back(
                    mesh.add_vertex( Point3(rv(i, 0), rv(i, 1), rv(i, 2)) )
                );
            }

            auto rf = faces.unchecked<2>();
            size_t nf = rf.shape(0);
            for (size_t i = 0; i < nf; ++i) {
                mesh.add_face(verts[rf(i, 0)], verts[rf(i, 1)], verts[rf(i, 2)]);
            }

            return mesh;
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
    ;
}