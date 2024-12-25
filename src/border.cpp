#include "seagullmesh.hpp"
#include <boost/iterator/function_output_iterator.hpp>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/merge_border_vertices.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, bool>       VertBool;
typedef Mesh3::Property_map<E, bool>       EdgeBool;
typedef Mesh3::Property_map<F, bool>       FaceBool;


void init_border(py::module &m) {
    m.def_submodule("border")
        .def("extract_boundary_cycles", [](const Mesh3& mesh) {
            std::vector<H> out;
            PMP::extract_boundary_cycles(mesh, std::back_inserter(out));
            return Indices<H>(out);  // returns one halfedge per boundary cycle
        })
        .def("has_boundary", [](const Mesh3& mesh) {
            std::vector<H> out;
            PMP::extract_boundary_cycles(mesh, std::back_inserter(out));
            return out.size() > 0;
        })
        .def("trace_boundary_from_vertex", [](const Mesh3& mesh, V v) {
            std::vector<V> verts;
            for ( H h0 : halfedges_around_source(v, mesh) ) {
                if ( mesh.is_border(h0) ) {
                    for (H h1 : halfedges_around_face(h0, mesh)) {  // around the null face
                        verts.emplace_back(mesh.source(h1));
                    }
                    return Indices<V>(verts);
                }
            }
            throw py::value_error("Vertex is not on the boundary");
        })
        .def("label_face_patch_border_edges", [](const Mesh3& mesh, const Indices<F>& faces, EdgeBool& is_border) {
            auto output_iter = boost::make_function_output_iterator([&mesh, &is_border](H h) {
                is_border[mesh.edge(h)] = true;
            });
            PMP::border_halfedges(faces.to_vector(), mesh, output_iter);
        })
        .def("label_border_vertices", [](const Mesh3& mesh, VertBool& is_border) {
            auto output_iter = boost::make_function_output_iterator([&mesh, &is_border](H h) {
                is_border[mesh.target(h)] = true;
            });
            PMP::border_halfedges(mesh.faces(), mesh, output_iter);
        })
        .def("label_border_edges", [](const Mesh3& mesh, EdgeBool& is_border) {
            auto output_iter = boost::make_function_output_iterator([&mesh, &is_border](H h) {
                is_border[mesh.edge(h)] = true;
            });
            PMP::border_halfedges(mesh.faces(), mesh, output_iter);
        })
        .def("label_border_faces", [](const Mesh3& mesh, FaceBool& is_border) {
            auto output_iter = boost::make_function_output_iterator([&mesh, &is_border](H h) {
                is_border[mesh.face(h)] = true;
            });
        })
        .def("merge_duplicated_vertices_in_boundary_cycles", [](Mesh3& mesh) {
            PMP::merge_duplicated_vertices_in_boundary_cycles(mesh);
        })
        .def("stitch_borders", [](Mesh3& mesh) {
            PMP::stitch_borders(mesh);
        })
    ;
}