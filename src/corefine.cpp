#include "seagullmesh.hpp"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, V>           VertexMap;
typedef Mesh3::Property_map<V, uint32_t>    VertexInt;
typedef Mesh3::Property_map<F, F>           FaceMap;
typedef Mesh3::Property_map<F, uint32_t>    FaceInt;
typedef Mesh3::Property_map<E, bool>        EdgeBool;


struct CorefineTracker : public PMP::Corefinement::Default_visitor<Mesh3> {
//https://github.com/CGAL/cgal/blob/master/Polygon_mesh_processing/examples/Polygon_mesh_processing/corefinement_mesh_union_with_attributes.cpp

    // Used for tracking for refinement indices
    // CGAL's corefine only uses a visitor for the first mesh, so we need the references to both
    // here to tell which is which
    Mesh3& mesh1;
    Mesh3& mesh2;
    VertexInt& vert_mesh_idx;
    VertexMap& vert_map;
    FaceInt& face_mesh_idx;
    FaceMap& face_map;
    uint32_t pre_split_face_mesh_idx;
    F pre_split_face;

    CorefineTracker(
        Mesh3& m1, Mesh3& m2, VertexInt& vi, VertexMap& vm, FaceInt& fi, FaceMap& fm
    ) : mesh1(m1), mesh2(m2),
        vert_mesh_idx(vi), vert_map(vm),
        face_mesh_idx(fi), face_map(fm),
        pre_split_face_mesh_idx(2),
        pre_split_face(Mesh3::null_face())
        {}

    uint32_t mesh_idx(const Mesh3& mesh) {
        if (&mesh == &mesh1) {return 0;} else if (&mesh == &mesh2) {return 1;} else {return 2;};
    }
    void after_vertex_copy(V v_src, const Mesh3& m_src, V v_tgt, const Mesh3& m_tgt) {
        vert_mesh_idx[v_tgt] = mesh_idx(m_src);
        vert_map[v_tgt] = v_src;
    }
    void after_face_copy(F f_src, const Mesh3& m_src, F f_tgt, const Mesh3& m_tgt) {
        face_mesh_idx[f_tgt] = mesh_idx(m_src);
        face_map[f_tgt] = f_src;
    }
    void new_vertex_added(size_t i_id, V v, const Mesh3& mesh) {
        vert_mesh_idx[v] = 2;
    }
    void before_subface_creations(F f, Mesh3& mesh) {
        pre_split_face = f;
        pre_split_face_mesh_idx = mesh_idx(mesh);
    }
    void after_subface_created(F f_new, const Mesh3& mesh) {
        face_mesh_idx[f_new] = pre_split_mesh_idx;
        face_map[f_new] = pre_split_face;
    }
};


// TODO
// All methods accept the visitor, who cares about optimizing the simple case
// All methods return bool success, throw on the python side

void init_corefine(py::module &m) {
    py::module sub = m.def_submodule("corefine");

//    py::class_<CorefineTracker>(sub, "CorefineTracker")
//        // .def(py::init<Mesh3&, Mesh3&, VertexInt&, VertexMap&, FaceInt&>())
//        .def(py::init())
//    ;


//    sub
//        .def("corefine", [](
//                Mesh3& mesh1,
//                Mesh3& mesh2,
//                EdgeConstrainedMap& ecm1,
//                EdgeConstrainedMap& ecm2,
//                CorefineTracker& tracker
//        ) {
//
//            auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
//            auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
//            PMP::corefine(mesh1, mesh2, params1, params2);
//        })
//        .def("union", [](
//                Mesh3& mesh1,
//                Mesh3& mesh2,
//                Mesh3& ouput,
//                EdgeConstrainedMap& ecm1,
//                EdgeConstrainedMap& ecm2,
//                CorefineTracker& tracker) {
//
//            auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
//            auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
//            return PMP::corefine_and_compute_union(mesh1, mesh2, output, params1, params2);
//        })
//    ;
}