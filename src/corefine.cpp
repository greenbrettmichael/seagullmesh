#include "seagullmesh.hpp"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, V>           Vertndex;
typedef Mesh3::Property_map<V, bool>        VertBool;
typedef Mesh3::Property_map<V, uint32_t>    VertInt;
typedef Mesh3::Property_map<F, F>           FaceIndex;
typedef Mesh3::Property_map<F, uint32_t>    FaceInt;
typedef Mesh3::Property_map<E, bool>        EdgeConstrainedMap;


struct CorefineTracker : public PMP::Corefinement::Default_visitor<Mesh3> {
//https://github.com/CGAL/cgal/blob/master/Polygon_mesh_processing/examples/Polygon_mesh_processing/corefinement_mesh_union_with_attributes.cpp

    // Used for tracking for refinement indices
    // CGAL's corefine only uses a visitor for the first mesh, so we need the references to both
    // here to tell which is which
    Mesh3& mesh1;
    Mesh3& mesh2;
    VertexBool& vert_is_new;
    FaceInt face_src;
    F to_be_split;


    CorefineVertexFaceTracker(
        Mesh3& m1, Mesh3& m2, VertexIndex& v1, VertexIndex& v2, FaceIndex& f1, FaceIndex&f2
    ) : mesh1(m1), mesh2(m2), vert_ids1(v1), vert_ids2(v2), face_ids1(f1), face_ids2(f2), face_id(-1) {}

    void after_vertex_copy(V v_src, const Mesh3& m_src, V v_tgt, const Mesh3& m_tgt) {
        //
    }
    void new_vertex_added(size_t i_id, V v, const Mesh3& mesh) {
        vert_is_new[v] = true;
    }
    void before_subface_creations(F f_split, Mesh3& mesh) {
        to_be_split = f_split;
    }
    void after_subface_created(F f_new, Mesh3& mesh) {
        if (&mesh == &mesh1) {
            face_ids1[f_new] = face_id;
        } elseif (&mesh == &mesh2) {
            face_ids2[f_new] = face_id;
        }
    }
    void after_face_copy(F f_src, Mesh3& mesh_src, F f_tgt, Mesh3& mesh_tgt) {
        face_ids1[f_tgt] = face_ids2[f_src];
    }
};


void init_corefine(py::module &m) {
    py::module sub = m.def_submodule("corefine");

    py::class_<CorefinementVertexTracker>(sub, "CorefinementVertexTracker")
        .def(py::init<Mesh3&, Mesh3&, VertexIndex&, VertexIndex&>())
    ;

    py::class_<CorefinementVertexFaceTracker>(sub, "CorefinementVertexFaceTracker")
        .def(py::init<Mesh3&, Mesh3&, VertexIndex&, VertexIndex&, FaceIndex&, FaceIndex&>())
    ;

    sub.def("corefine", [](Mesh3& mesh1, Mesh3& mesh2){
        PMP::corefine(mesh1, mesh2);
    })
    .def("corefine", [](
            Mesh3& mesh1, Mesh3& mesh2, 
            EdgeConstrainedMap& ecm1, EdgeConstrainedMap& ecm2,
            CorefinementVertexTracker& tracker) {

        auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
        auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
        PMP::corefine(mesh1, mesh2, params1, params2);
    })
    .def("corefine", [](
            Mesh3& mesh1, Mesh3& mesh2,
            EdgeConstrainedMap& ecm1, EdgeConstrainedMap& ecm2,
            CorefinementVertexFaceTracker& tracker) {

        auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
        auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
        PMP::corefine(mesh1, mesh2, params1, params2);
    })
    .def("clip", [](Mesh3& mesh1, Mesh3& mesh2, CorefinementVertexTracker& tracker){
        auto params1 = PMP::parameters::visitor(tracker);
        PMP::clip(mesh1, mesh2, params1);
    })
    .def("clip", [](Mesh3& mesh1, Mesh3& mesh2, CorefinementVertexFaceTracker& tracker){
        auto params1 = PMP::parameters::visitor(tracker);
        PMP::clip(mesh1, mesh2, params1);
    })
    .def("split", [](Mesh3& mesh1, Mesh3& mesh2, CorefinementVertexTracker& tracker){
        auto params1 = PMP::parameters::visitor(tracker);
        PMP::split(mesh1, mesh2, params1);
    })
    .def("split", [](Mesh3& mesh1, Mesh3& mesh2, CorefinementVertexFaceTracker& tracker){
        auto params1 = PMP::parameters::visitor(tracker);
        PMP::split(mesh1, mesh2, params1);
    })
    .def("difference", [](Mesh3& mesh1, Mesh3& mesh2, Mesh3& out) {
        bool success = PMP::corefine_and_compute_difference(mesh1, mesh2, out);
        if (!success) {
            throw std::runtime_error("Boolean operation failed.");
        }
    })
    .def("union", [](Mesh3& mesh1, Mesh3& mesh2, Mesh3& out) {
        bool success = PMP::corefine_and_compute_union(mesh1, mesh2, out);
        if (!success) {
            throw std::runtime_error("Boolean operation failed.");
        }
    })
    .def("intersection", [](Mesh3& mesh1, Mesh3& mesh2, Mesh3& out) {
        bool success = CGAL::Polygon_mesh_processing::corefine_and_compute_intersection(mesh1, mesh2, out);
        if (!success) {
            throw std::runtime_error("Boolean operation failed.");
        }
    })
    .def("union", [](
            Mesh3& mesh1, Mesh3& mesh2,
            EdgeConstrainedMap& ecm1, EdgeConstrainedMap& ecm2,
            CorefinementVertexTracker& tracker) {

        auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
        auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
        bool success = PMP::corefine_and_compute_union(mesh1, mesh2, mesh1, params1, params2);
        if (!success) {
            throw std::runtime_error("Boolean operation failed.");
        }
    })
    ;
}