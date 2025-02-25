#include "seagullmesh.hpp"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <boost/container/flat_map.hpp>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<E, bool> EdgeBool;
typedef Mesh3::Property_map<F, int32_t> FaceMeshMap;
typedef Mesh3::Property_map<F, F> FaceFaceMap;
typedef Mesh3::Property_map<V, int32_t> VertMeshMap;


struct CorefineTracker : public PMP::Corefinement::Default_visitor<Mesh3> {
    struct Tracked {
        size_t mesh_idx; // 0, 1, or 2
        FaceMeshMap face_mesh_map;  // F -> 0, 1, 2
        FaceFaceMap face_face_map;  // F -> F_original
        VertMeshMap vert_mesh_map;  // V -> 0, 1, 2
        EdgeBool edge_is_constrained_map;
    };

    boost::container::flat_map<const Mesh3*, Tracked> tracked;
    F orig_face;
    size_t mesh_idx;

    CorefineTracker() {
        tracked.reserve(3);  //input mesh 0, input mesh1, maybe output mesh
        orig_face = Mesh3::null_face();
    }
    void track(
            Mesh3& mesh,
            size_t mesh_idx,
            FaceMeshMap& face_mesh_map,
            FaceFaceMap& face_face_map,
            VertMeshMap& vert_mesh_map,
            EdgeBool& edge_is_constrained_map
        ) {
        // Called from python to store references to the appropriate property maps
        tracked[&mesh] = Tracked{mesh_idx, face_mesh_map, face_face_map, vert_mesh_map, edge_is_constrained_map};
    }
    void new_vertex_added(size_t i_id, V v, const Mesh3& mesh) {
        Tracked& t = tracked[&mesh];
        t.vert_mesh_map[v] = t.mesh_idx;
    }
    void after_vertex_copy(V v_src, const Mesh3& m_src, V v_tgt, const Mesh3& m_tgt) {
        tracked[&m_tgt].vert_mesh_map[v_tgt] = tracked[&m_src].mesh_idx;
        // tracked[&m_tgt].vert_vert_map[v_tgt] = tracked[&m_src].vert_vert_map[v_src];
    }

    // todo p sure these are supposed to be const Mesh but it compiles anyway
    void before_subface_creations(F f_split, Mesh3& mesh) {
        orig_face = tracked[&mesh].face_face_map[f_split];
        mesh_idx = tracked[&mesh].mesh_idx;
    }
    void after_subface_created(F f_new, Mesh3& mesh) {
        tracked[&mesh].face_mesh_map[f_new] = mesh_idx;
        tracked[&mesh].face_face_map[f_new] = orig_face;
    }
    void after_face_copy(F f_src, Mesh3& m_src, F f_tgt, Mesh3& m_tgt) {
        tracked[&m_tgt].face_mesh_map[f_tgt] = tracked[&m_src].mesh_idx;
        tracked[&m_tgt].face_face_map[f_tgt] = tracked[&m_src].face_face_map[f_src];
    }

};
void init_corefine(py::module &m) {
    py::module sub = m.def_submodule("corefine");

    py::class_<CorefineTracker>(sub, "CorefineTracker")
        .def(py::init<>())
        .def("track", &CorefineTracker::track)
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
                EdgeBool& ecm1,
                EdgeBool& ecm2,
                CorefineTracker &tracker,
                Mesh3& output
        ) {
            auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
            auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
            return PMP::corefine_and_compute_union(mesh1, mesh2, output, params1, params2);
        })
        .def("intersection", [](
                Mesh3& mesh1,
                Mesh3& mesh2,
                EdgeBool& ecm1,
                EdgeBool& ecm2,
                CorefineTracker &tracker,
                Mesh3& output
        ) {
            auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
            auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
            return PMP::corefine_and_compute_intersection(mesh1, mesh2, output, params1, params2);
        })
        .def("difference", [](
                Mesh3& mesh1,
                Mesh3& mesh2,
                EdgeBool& ecm1,
                EdgeBool& ecm2,
                CorefineTracker &tracker,
                Mesh3& output
        ) {
            auto params1 = PMP::parameters::visitor(tracker).edge_is_constrained_map(ecm1);
            auto params2 = PMP::parameters::edge_is_constrained_map(ecm2);
            return PMP::corefine_and_compute_difference(mesh1, mesh2, output, params1, params2);
        })
        // TODO corefine_and_compute_boolean_operations
        // TODO just pass the tracker object
        .def("clip", [](
                Mesh3& mesh,
                Mesh3& clipper,
                CorefineTracker &tracker,
                bool clip_volume,
                bool use_compact_clipper
        ) {
            auto params = PMP::parameters::visitor(tracker)
                .clip_volume(clip_volume)
                .use_compact_clipper(use_compact_clipper)
            ;
            return PMP::clip(mesh, clipper, params);
        })
    ;
}