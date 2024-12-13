#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, double>  VertDouble;
typedef Mesh3::Property_map<F, bool>    FaceBool;


class TubeMesher {
    private:

    Mesh3& mesh;
    VertDouble& t_map;
    VertDouble& theta_map;
    FaceBool& is_cap_map;

    // TODO hold both next and prev and std:swap?
    std::map<double, V> prev_xs;

    std::map<double, V> add_points_and_radial_edges(
            const double t,
            const py::array_t<double>& theta,
            const py::array_t<double>& pts
    ) {
        const size_t n = pts.shape(0);
        auto r_pts = pts.unchecked<2>();
        auto r_theta = theta.unchecked<1>();
        std::map<double, V> out;

        for (size_t i = 0; i < n; ++i) {
            Point3 pt = Point3(r_pts(i, 0), r_pts(i, 1), r_pts(i, 2));
            V v = mesh.add_vertex(pt);
            t_map[v] = t;
            theta_map[v] = r_theta(i);
            out[r_theta(i)] = v;
        }
        out[2 * CGAL_PI] = out[0];  // Close the loop
        return out;
    }

    public:

    TubeMesher(Mesh3& mesh, VertDouble& t_map, VertDouble& theta_map, FaceBool& is_cap_map)
        : mesh(mesh), t_map(t_map), theta_map(theta_map), is_cap_map(is_cap_map) {}

    void add_xs(double t, const py::array_t<double>& theta, const py::array_t<double>& pts) {
        std::map<double, V> next_xs = add_points_and_radial_edges(t, theta, pts);
        if (prev_xs.size() == 0) {  // Must be the first xs
            prev_xs = std::move(next_xs);
            return;
        }

        std::vector<V> face;
        double theta0 = 0, theta1 = 0;
        while (theta0 < 2 * CGAL_PI) {
            // Initialize face with axial edge from prev_xs to next_xs
            face.clear();
            face.push_back(prev_xs[theta0]);
            face.push_back(next_xs[theta0]);

            // Walk forward on next_xs until we find a theta1 that is also present in prev_xs
            for (auto it1 = next_xs.upper_bound(theta0); it1 != next_xs.end(); ++it1) {
                theta1 = it1->first;
                face.push_back(it1->second);

                if (auto it0 = prev_xs.find(theta1); it0 != prev_xs.end()) {
                    // Found matching point in prev_xs  -- Walk backwards on prev_xs until we return to theta0
                    while (it0->first > theta0) {
                        face.push_back(it0->second);
                        it0--;
                    }
                    break; // Finished this face
                }
            }
            mesh.add_face(face);
            theta0 = theta1;
        }
        prev_xs = std::move(next_xs);
    }
    void close_xs(bool flip) {
        std::vector<V> face;
        face.reserve(prev_xs.size());

        for (auto const& kv : prev_xs) {
            // NB the first vertex is stored twice, once at 0 and once at 2pi
            // don't add it twice
            if (kv.first != 2 * CGAL_PI) {
                face.push_back(kv.second);
            }
        }

        if (flip){
            std::reverse(face.begin(), face.end());
        }

        F f = mesh.add_face(face);
        TriangulateCapVisitor visitor(is_cap_map);
        PMP::triangulate_face(f, mesh, PMP::parameters::visitor(visitor));
    }

    private:
    struct TriangulateCapVisitor : public PMP::Triangulate_faces::Default_visitor<Mesh3> {
        FaceBool& is_cap_map;
        TriangulateCapVisitor(FaceBool& is_cap_map) : is_cap_map(is_cap_map) {}
        void after_subface_created(F f) {is_cap_map[f] = true;}
    };
};

void init_tube_mesher(py::module &m) {
    py::module sub = m.def_submodule("tube_mesher");

    py::class_<TubeMesher>(sub, "TubeMesher")
        .def(py::init<Mesh3&, VertDouble&, VertDouble&, FaceBool&>())
        .def("add_xs", &TubeMesher::add_xs)
        .def("close_xs", &TubeMesher::close_xs)
    ;

    sub
        .def("triangulate_faces", [](Mesh3& mesh, const std::vector<F>& faces) {
            PMP::triangulate_faces(faces, mesh);
        })
        .def("reverse_face_orientations", [](Mesh3& mesh, const std::vector<F>& faces) {
            PMP::reverse_face_orientations(faces, mesh);
        })
    ;
}
