#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>

#include <CGAL/Surface_mesh_approximation/approximate_triangle_mesh.h>

namespace VSA = CGAL::Surface_mesh_approximation;
namespace PMP = CGAL::Polygon_mesh_processing;

void init_approximation(py::module &m) {
    m.def_submodule("approximation")
        .def("approximate_mesh", [](const Mesh3& mesh, const std::size_t n_max_proxies) {
            std::vector<Point3> anchors;
            std::vector<std::array<std::size_t, 3> > triangles;

            // todo: return points and faces or the mesh (raising exception if approx fails?)
            // todo: need options to specify target error instead of n_max proxies
            // todo face_proxy_map: a property map to output the proxy index of each face of the input polygon mesh
            bool is_manifold = VSA::approximate_triangle_mesh(
                mesh,
                CGAL::parameters::max_number_of_proxies(n_max_proxies)
                   .anchors(std::back_inserter(anchors))
                   .triangles(std::back_inserter(triangles))
            );

            if (is_manifold) {
                PMP::orient_polygon_soup(anchors, triangles);
                Mesh3 output;
                PMP::polygon_soup_to_polygon_mesh(anchors, triangles, output);
                if ( CGAL::is_closed(output) && (!PMP::is_outward_oriented(output)) ) {
                    PMP::reverse_face_orientations(output);
                }
                return output;
            } else {
                 throw std::runtime_error("Non-manifold approximation!");
            }
        })
    ;
}