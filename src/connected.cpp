#include "seagullmesh.hpp"
#include "util.hpp"
#include <boost/property_map/property_map.hpp>

#include <CGAL/boost/graph/selection.h>
#include <CGAL/boost/graph/Face_filtered_graph.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/boost/graph/copy_face_graph.h>

typedef CGAL::Face_filtered_graph<Mesh3>        FilteredMesh;

typedef Mesh3::Property_map<V, bool>            VertBool;
typedef Mesh3::Property_map<F, bool>            FaceBool;
typedef Mesh3::Property_map<E, bool>            EdgeBool;


typedef Mesh3::Property_map<V, V>            VertVertMap;
typedef Mesh3::Property_map<F, F>            FaceFaceMap;
typedef Mesh3::Property_map<E, E>            EdgeEdgeMap;
typedef Mesh3::Property_map<H, H>            HalfedgeHalfedgeMap;


namespace PMP = CGAL::Polygon_mesh_processing;


template<typename T>
void define_keep_or_remove_connected_components_for_property_map_type(py::module &m) {
    typedef std::vector<T> Vals;
    typedef Mesh3::Property_map<F, T> FaceMap;

    m.def("remove_connected_face_patches", [](Mesh3& mesh, const Vals& to_remove, const FaceMap& face_map) {
        PMP::remove_connected_components(mesh, to_remove, face_map);
    })
    .def("keep_connected_face_patches", [](Mesh3& mesh, const Vals& to_keep, const FaceMap& face_map) {
        PMP::keep_connected_components(mesh, to_keep, face_map);
    })
    ;
}

template<typename T>
void define_label_connected_components_for_property_map_type(py::module &m) {
    typedef Mesh3::Property_map<F, T> FaceMap;

    m.def("label_connected_components", [](const Mesh3& mesh, FaceMap& face_map, EdgeBool& edge_is_constrained) {
        // Returns the number of connected components
        auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained);
        return PMP::connected_components(mesh, face_map, params);
    })
    ;
}



void init_connected(py::module &m) {
    py::module sub = m.def_submodule("connected");
    sub
        .def("edge_halfedge"), [](const Mesh3& mesh, const Indices<E>& edges) {
            return edges.map_to_indices<H>(&mesh.halfedge);
        }
        .def("halfedge_edge", [](const Mesh3& mesh, const Indices<H>& halfedges) {
            return halfedges.map_to_indices<E>(&mesh.edge);
        })
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
        .def("faces_to_faces", [](const Mesh3& mesh, const Indices<F>& faces) {
            std::set<F> out;
            for (F f : faces.to_vector()) {
                for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
                    F f_adj = mesh.face(mesh.opposite(h));
                    if ( f_adj != mesh.null_face() ){
                        out.insert(f_adj);
                    }
                }
            }
            return Indices<F>(out);
        })
        .def("faces_to_vertices", [](const Mesh3& mesh, const Indices<F>& faces) {
            std::set<V> verts;
            for (F f : faces.to_vector()) {
                for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
                    verts.insert(mesh.source(h));
                }
            }
            return Indices<V>(verts);
        })
        .def("vertex_degrees", [](const Mesh3& mesh, const Indices<V>& verts) {
            return verts.map_to_array_of_scalars<Mesh3::size_type>(
                [&mesh](V v) { return mesh.degree(v); });
        })
        .def("face_degrees", [](const Mesh3& mesh, const Indices<F>& faces) {
            return faces.map_to_array_of_scalars<Mesh3::size_type>(
                [&mesh](F f) { return mesh.degree(f); });
        })
        .def("label_selected_face_patches", [](
                Mesh3& mesh,
                const Indices<F>& faces,
                Mesh3::Property_map<F, F::size_type>& face_map
            ) {
            FilteredMesh filtered(mesh, faces.to_vector());
            std::map<F, F::size_type> filtered_face_patch;
            boost::associative_property_map< std::map<F, F::size_type> > filtered_face_patch_map(filtered_face_patch);
            auto n_components = PMP::connected_components(filtered, filtered_face_patch_map);

            for (auto const& [f, i] : filtered_face_patch) {
                face_map[f] = i + 1;
            }
            return n_components;
        })
        .def("copy_faces", [](
                const Mesh3& src,
                Mesh3& dest,
                const Indices<F>& faces,
                VertVertMap& vvm,
                FaceFaceMap& ffm,
                HalfedgeHalfedgeMap hhm
            ) {
            FilteredMesh filtered(src, faces.to_vector());
            auto params = CGAL::parameters::vertex_to_vertex_map(vvm)
                .halfedge_to_halfedge_map(hhm)
                .face_to_face_map(ffm);

            CGAL::copy_face_graph(filtered, dest, params);
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
        .def("keep_connected_faces", [](Mesh3& mesh, const Indices<F>& faces) {
            PMP::keep_connected_components(mesh, faces.to_vector());
        })
        .def("regularize_face_selection_borders", [](Mesh3& mesh, FaceBool& is_selected, double weight, bool prevent_unselection) {
            // I think this hits the same problem with https://github.com/CGAL/cgal/issues/2788
            boost::unordered_map<F, bool> temp_map;
            for (F f : mesh.faces()) {
                temp_map[f] = is_selected[f];
            }
            boost::associative_property_map< boost::unordered_map<F, bool> > temp_prop_map(temp_map);

            auto params = CGAL::parameters::prevent_unselection(prevent_unselection);
            CGAL::regularize_face_selection_borders(mesh, temp_prop_map, weight, params);

            for (F f : mesh.faces()) {
                is_selected[f] = temp_map[f];
            }
        })
        .def("expand_face_selection_for_removal", [](Mesh3& mesh, Indices<F>& faces, FaceBool& is_selected) {
            CGAL::expand_face_selection_for_removal(faces.to_vector(), mesh, is_selected);
        })
        .def("expand_vertex_selection", [](Mesh3& mesh, Indices<V>& verts, unsigned int k, VertBool& is_selected) {
            std::vector<V> added;
            CGAL::expand_vertex_selection(verts.to_vector(), mesh, k, is_selected, std::back_inserter(added));
            return added;
        })
        .def("expand_face_selection", [](Mesh3& mesh, Indices<F>& faces, unsigned int k, FaceBool& is_selected) {
            std::vector<F> added;
            CGAL::expand_face_selection(faces.to_vector(), mesh, k, is_selected, std::back_inserter(added));
            return added;
        })
    ;

    define_keep_or_remove_connected_components_for_property_map_type<F::size_type>(sub);
    define_keep_or_remove_connected_components_for_property_map_type<int64_t>(sub);
    define_keep_or_remove_connected_components_for_property_map_type<bool>(sub);

    // Requires ++ so doesn't work with bools
    define_label_connected_components_for_property_map_type<F::size_type>(sub);
    define_label_connected_components_for_property_map_type<int64_t>(sub);

}