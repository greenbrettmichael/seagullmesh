#include "seagullmesh.hpp"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>

namespace PMP = CGAL::Polygon_mesh_processing;

//typedef std::pair<uint32_t, V>                                      VertexOrigin;
//typedef CGAL::dynamic_vertex_property_t<VertexOrigin>               VertexOriginProperty;
//typedef boost::property_map<Mesh3, VertexOriginProperty>::type      VertexOriginPMap;

typedef std::pair<size_t, F>                                        FaceOrigin;
typedef CGAL::dynamic_face_property_t<FaceOrigin>                   FaceOriginProperty;
typedef boost::property_map<Mesh3, FaceOriginProperty>::type        FaceOriginPropertyMap;

typedef CGAL::dynamic_face_property_t<F>                            FaceId;
typedef boost::property_map<Mesh3, FaceId>::type                    FaceIdPropertyMap;


// todo: need pmap[F, F] on mesh1 and mesh2
struct CorefineTracker : public PMP::Corefinement::Default_visitor<Mesh3> {
    Mesh3& mesh1;
    Mesh3& mesh2;
    Mesh3& output;
    std::array<FaceIdPropertyMap, 2> original_faces;

    FaceOrigin face_origin;
    FaceOriginPropertyMap face_origins;

    CorefineTracker(Mesh3& m1, Mesh3& m2, Mesh3& out) : mesh1(m1), mesh2(m2), output(out) {
        original_faces = {
            get(FaceId(), mesh1, Mesh3::null_face()),
            get(FaceId(), mesh2, Mesh3::null_face())
        };
        for (F f : mesh1.faces()) { put(original_faces[0], f, f); }
        for (F f : mesh2.faces()) { put(original_faces[1], f, f); }

        face_origin = std::make_pair(2, Mesh3::null_face());
        face_origins = get(FaceOriginProperty(), output, face_origin);
    }
    uint32_t mesh_idx(const Mesh3& mesh) {
        if (&mesh == &mesh1) {return 0;} else if (&mesh == &mesh2) {return 1;} else {return 2;};
    }

    void before_subface_creations(F f_to_be_split, Mesh3& mesh) {
        // Going to split a face in mesh1 or mesh2 during the corefinement stage
        auto i = mesh_idx(mesh);
        F f_original = get(original_faces[i], f_to_be_split);
        face_origin = std::make_pair(i, f_original);
    }
    void after_subface_created(F f_new, const Mesh3& mesh) {
        auto i = face_origin.first;
        F f_original = face_origin.second;
        put(original_faces[i], f_new, f_original);
    }
    void after_face_copy(F f_src, const Mesh3& m_src, F f_tgt, const Mesh3& m_tgt) {
        put(face_origins, f_tgt, std::make_pair(mesh_idx(m_src), f_tgt));
    }
};


// TODO
// All methods accept the visitor, who cares about optimizing the simple case
// All methods return bool success, throw on the python side

void init_corefine(py::module &m) {
    py::module sub = m.def_submodule("corefine");

    py::class_<CorefineTracker>(sub, "CorefineTracker")
        .def(py::init<Mesh3&, Mesh3&, Mesh3&>())
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