#include "seagullmesh.hpp"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>

namespace PMP = CGAL::Polygon_mesh_processing;

struct CorefineTracker : public PMP::Corefinement::Default_visitor<Mesh3> {
    typedef std::map<F, F> FaceMap;
    struct FaceOrigin {
        mesh_idx: size_t
        face: F
    };
    Mesh3& mesh1;
    Mesh3& mesh2;

    std::array<FaceMap, 2> split_faces;     // subface -> original face for each source mesh
    std::array<FaceMap, 2> copied_faces;    // subface in output -> orig_face

    CorefineTracker(Mesh3& m1, Mesh3& m2) : mesh1(m1), mesh2(m2) {};

    uint32_t mesh_idx(const Mesh3& mesh) const {
        if (&mesh == &mesh1) {return 0;} else if (&mesh == &mesh2) {return 1;} else {return 2;};
    }

    F original_face(size_t mesh_idx, F face) const {
        auto it = split_faces[i].find(face);
        return (it != split_faces[i].end()) ? it->second : face;
    }

    void before_subface_creations(F f, const Mesh3& mesh) {
        // Going to split a face in mesh1 or mesh2 during the corefinement stage
        auto i = mesh_idx(mesh);
        face_origin = {i, original_face(i, f)};
    }
    void after_subface_created(F f_new, const Mesh3& mesh) {
        split_faces[face_origin.mesh_idx][f_new] = face_origin.face;
    }
    void after_face_copy(F f_src, const Mesh3& m_src, F f_tgt, const Mesh3& m_tgt) {
        auto i = mesh_idx(m_src);
        copied_faces[i][f_tgt] = original_face(i, f_src);
    }

//    void get_mapped_faces(size_t i) {
//        std::vector<F> faces_input;
//        std::vector<F> faces_output;
//        for (F f : output.faces()) {
//            const FaceOrigin& fo = get(face_origins, f);
//            if (fo.first == i) {
//                faces_input.push_back(fo.second);
//                faces_output.
//            }
//        }
//    }
};


// TODO
// All methods accept the visitor, who cares about optimizing the simple case
// All methods return bool success, throw on the python side

void init_corefine(py::module &m) {
    py::module sub = m.def_submodule("corefine");

    py::class_<CorefineTracker>(sub, "CorefineTracker")
        .def(py::init<Mesh3&, Mesh3&>())
    ;


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