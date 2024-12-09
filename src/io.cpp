#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/Surface_mesh/IO/OFF.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::dynamic_vertex_property_t<size_t>                           VertexIndex;
typedef typename boost::property_map<Mesh3, VertexIndex>::const_type      VertexIndexMap;


VertexIndexMap build_vertex_index_map(const Mesh3& mesh) {
    VertexIndexMap vim = get(VertexIndex(), mesh);
    size_t i = 0;
    for (const V v : mesh.vertices()) {
        put(vim, v, i++);
    }
    return vim;
}


void init_io(py::module &m) {
    m.def_submodule("io")
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
        .def("mesh3_to_polygon_soup", [](const Mesh3& mesh) {
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
        .def("load_mesh_from_file", [](const std::string filename) {
            Mesh3 mesh;
            if(!CGAL::IO::read_polygon_mesh(filename, mesh)) {
                throw std::runtime_error("Failed to load mesh");
            }
            return mesh;
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