#include "seagullmesh.hpp"
#include <CGAL/Min_sphere_of_points_d_traits_3.h>
#include <CGAL/Min_sphere_of_spheres_d.h>

typedef CGAL::Min_sphere_of_points_d_traits_3<Kernel, double>   Traits;
typedef CGAL::Min_sphere_of_spheres_d<Traits>                   MinSphere;

namespace PMP = CGAL::Polygon_mesh_processing;

void init_min_sphere(py::module &m) {
    m.def_submodule("min_sphere")
        .def("min_sphere", [](const Mesh3& mesh) {
            const auto vpm = mesh.points();
            std::vector<Point3> points;
            points.reserve(mesh.number_of_vertices());
            for (V v : mesh.vertices()) { points.emplace_back(vpm[v]); }
            MinSphere s(points.begin(), points.end());

            auto it = s.center_cartesian_begin();
            double x = *it++;
            double y = *it++;
            double z = *it;
            Point3 center(x, y, z);

            return std::make_tuple(center, s.radius());
        })
    ;
}