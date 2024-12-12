#include "seagullmesh.hpp"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <boost/container/flat_map.hpp>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<E, bool> EdgeBool;
typedef Mesh3::Property_map<F, int64_t> FaceInt;


struct CorefineTracker : public PMP::Corefinement::Default_visitor<Mesh3> {
    boost::container::flat_map<const Mesh3*, FaceInt> face_id_maps;
    int64_t face_id;

    CorefineTracker() {
        face_id_maps.reserve(3);
        face_id = -1;
    }
    void track_mesh(Mesh3& mesh, FaceInt& face_id_map) {
        face_id_maps[&mesh] = face_id_map;
    }

    void new_vertex_added (size_t i_id, V v, Mesh3& mesh) {}

    void before_subface_creations(F f_split, Mesh3& mesh) {
        face_id = face_id_maps[&mesh][f_split] ;
    }
    void after_subface_created(F f_new, Mesh3& mesh) {
        face_id_maps[&mesh][f_new] = face_id;
    }
    void after_face_copy(F f_src, Mesh3& m_src, F f_tgt, Mesh3& m_tgt) {
        face_id_maps[&m_tgt][f_tgt] = face_id_maps[&m_src][f_src];
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