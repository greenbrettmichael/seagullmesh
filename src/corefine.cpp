#include "seagullmesh.hpp"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <boost/container/flat_map.hpp>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<E, bool> EdgeBool;
typedef Mesh3::Property_map<F, int64_t> FaceInt;


struct CorefineTracker : public PMP::Corefinement::Default_visitor<Mesh3> {
    struct Tracked {
        FaceInt face_idx;
    };

    boost::container::flat_map<const Mesh3*, Tracked> tracked;
    int64_t face_idx;

    CorefineTracker() {
        tracked.reserve(3);
        face_idx = -1;
    }
    void track_mesh(Mesh3& mesh, FaceInt& face_idx) {
        tracked[&mesh] = Tracked{face_idx};
    }

    void new_vertex_added (size_t i_id, V v, Mesh3& mesh) {}

    void before_subface_creations(F f_split, Mesh3& mesh) {
        face_idx = tracked[&mesh].face_idx[f_split];
    }
    void after_subface_created(F f_new, Mesh3& mesh) {
        tracked[&mesh].face_idx[f_new] = face_idx;
    }
    void after_face_copy(F f_src, Mesh3& m_src, F f_tgt, Mesh3& m_tgt) {
        tracked[&m_tgt].face_idx[f_tgt] = tracked[&m_src].face_idx[f_src];
    }

};
void init_corefine(py::module &m) {
    py::module sub = m.def_submodule("corefine");

    py::class_<CorefineTracker>(sub, "CorefineTracker")
        .def(py::init<>())
        .def("track_mesh", &CorefineTracker::track_mesh)
    ;

    sub
        .def("corefine", [](
                Mesh3& mesh1,
                Mesh3& mesh2,
                EdgeBool& ecm1,
                EdgeBool& ecm2,
                CorefineTracker &tracker
        ) {
            auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
            auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
            PMP::corefine(mesh1, mesh2, params1, params2);
        })
        .def("union", [](
                Mesh3& mesh1,
                Mesh3& mesh2,
                Mesh3& output,
                EdgeBool& ecm1,
                EdgeBool& ecm2,
                CorefineTracker &tracker
        ) {
            auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
            auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
            return PMP::corefine_and_compute_union(mesh1, mesh2, output, params1, params2);
        })
    ;
}