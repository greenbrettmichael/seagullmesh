#include "seagullmesh.hpp"

namespace PMP = CGAL::Polygon_mesh_processing;


void init_generators(py::module &m) {
    m.def_submodule("generators")
        .def("extract_boundary_cycles", [](const Mesh3& mesh) {
            std::vector<H> out;
            PMP::extract_boundary_cycles(mesh, std::back_inserter(out));
            return out;  // returns one halfedge per boundary cycle
        })
        .def("has_boundary", [](const Mesh3& mesh) {
            std::vector<H> out;
            PMP::extract_boundary_cycles(mesh, std::back_inserter(out));
            return out.size() > 0;
        })
        .def("face_patch_border", [](const Mesh3& mesh, const std::vector<F>& faces) {
            std::vector<H> out;
            PMP::border_halfedges(faces, mesh, std::back_inserter(out));
            return out;
        })
        .def("label_border_vertices", [](const Mesh3& mesh, VertBool& vert_is_border) {
            auto output_iter = boost::make_function_output_iterator(touch_border_vertices(mesh, vert_is_border));
            PMP::border_halfedges(faces(mesh), mesh, output_iter);
        })
        .def("label_border_edges", [](const Mesh3& mesh, EdgeBool& edge_is_border) {
            auto output_iter = boost::make_function_output_iterator(touch_border_edges(mesh, edge_is_border));
            PMP::border_halfedges(faces(mesh), mesh, output_iter);
        })
        .def("regularize_face_selection_borders", [](Mesh3& mesh, FaceBool& is_selected, double weight, bool prevent_unselection) {
            auto params = CGAL::parameters::prevent_unselection(prevent_unselection);
            CGAL::regularize_face_selection_borders(mesh, is_selected, weight, params);
        })
        .def("expand_face_selection_for_removal", [](Mesh3& mesh, std::vector<F>& faces, FaceBool& is_selected) {
            CGAL::expand_face_selection_for_removal(faces, mesh, is_selected);
        })
        .def("expand_vertex_selection", [](Mesh3& mesh, std::vector<V>& verts, unsigned int k, VertBool& is_selected) {
            std::vector<V> added;
            CGAL::expand_vertex_selection(verts, mesh, k, is_selected, std::back_inserter(added));
            return added;
        })
        .def("expand_face_selection", [](Mesh3& mesh, std::vector<F>& faces, unsigned int k, FaceBool& is_selected) {
            std::vector<F> added;
            CGAL::expand_face_selection(faces, mesh, k, is_selected, std::back_inserter(added));
            return added;
        })
    ;
}