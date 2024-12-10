#include "seagullmesh.hpp"
#include "util.hpp"
#include <boost/property_map/property_map.hpp>

#include <CGAL/boost/graph/selection.h>
#include <CGAL/boost/graph/Face_filtered_graph.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

typedef CGAL::Face_filtered_graph<Mesh3>        FilteredMesh;

typedef Mesh3::Property_map<V, bool>            VertBool;
typedef Mesh3::Property_map<F, bool>            FaceBool;
typedef Mesh3::Property_map<E, bool>            EdgeBool;

typedef F::size_type                            FaceIdx;
typedef Mesh3::Property_map<F, FaceIdx>         FacePatchMap;

namespace PMP = CGAL::Polygon_mesh_processing;

void init_connected(py::module &m) {
    m.def_submodule("connected")
        .def("vertices_to_faces", [](const Mesh3& mesh, const Indices<V>& verts) {
            std::set<F> faces;
            for (V v : verts.to_vector()) {
                // mesh.halfedge(v) returns an incoming halfedge of vertex v
                for (F f : faces_around_target(mesh.halfedge(v), mesh)) {
                    if (f != mesh.null_face()) {
                        faces.insert(f);
                    }
                }
            }
            return Indices<F>(faces);
        })
        .def("vertices_to_edges", [](const Mesh3& mesh, const Indices<V>& verts) {
            std::set<E> edges;
            for (V v : verts.to_vector()) {
                for (H h : halfedges_around_source(v, mesh)) {
                    edges.insert(mesh.edge(h));
                }
            }
            return Indices<E>(edges);
        })
        .def("faces_to_edges", [](const Mesh3& mesh, const Indices<F>& faces) {
            std::set<E> edges;
            for (F f : faces.to_vector()) {
                for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
                    edges.insert(mesh.edge(h));
                }
            }
            return Indices<E>(edges);
        })
        .def("faces_to_vertices", [](const Mesh3& mesh, const Indices<F>& faces) {
            std::set<V> verts;
            for (F f : faces.to_vector()) {
                for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
                    verts.insert(mesh.source(h));
                    // verts.insert(mesh.target(h));
                }
            }
            return Indices<V>(verts);
        })
        .def("vertex_degrees", [](const Mesh3& mesh, const Indices<V>& verts) {
            return verts.map_to_array_of_scalars<Mesh3::size_type>(
                [&mesh](V v) { return mesh.degree(v); });
        })
        .def("label_connected_components", [](const Mesh3& mesh, FacePatchMap& face_patch, EdgeBool& edge_is_constrained) {
            // Returns the number of connected components
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained);
            return PMP::connected_components(mesh, face_patch, params);
        })
        .def("label_selected_face_patches", [](Mesh3& mesh, const Indices<F>& faces, FacePatchMap& face_patch) {
            FilteredMesh filtered(mesh, faces.to_vector());

            std::map<F, F::size_type> filtered_face_patch;
            boost::associative_property_map< std::map<F, F::size_type> > filtered_face_patch_map(filtered_face_patch);
            auto n_components = PMP::connected_components(filtered, filtered_face_patch_map);

            for (auto const& [f, i] : filtered_face_patch) {
                face_patch[f] = i + 1;
            }
            return n_components;
        })
        .def("keep_connected_components", [](Mesh3& mesh, const std::vector<F::size_type>& components_to_keep, const FacePatchMap& components) {
            PMP::keep_connected_components(mesh, components_to_keep, components);
        })
        .def("connected_component", [](const Mesh3& mesh, F seed_face, EdgeBool& edge_is_constrained) {
            std::vector<F> out;
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained);
            PMP::connected_component(seed_face, mesh, std::back_inserter(out), params);
            return Indices<F>(out);
        })
        .def("remove_connected_faces", [](Mesh3& mesh, const Indices<F>& faces) {
            PMP::remove_connected_components(mesh, faces.to_vector());
        })
        .def("remove_connected_face_patches", [](Mesh3& mesh, const std::vector<FaceIdx>& components_to_remove, const FacePatchMap& components) {
            PMP::remove_connected_components(mesh, components_to_remove, components);
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
//      TODO: maybe I actually want make_hole?
//      #include <CGAL/boost/graph/Euler_operations.h>
//        .def("remove_faces", [](Mesh3& mesh, const Indices<F>& faces) {
//            faces.apply([&mesh](F f){ CGAL::Euler::remove_face(); });
//        })
    ;
}