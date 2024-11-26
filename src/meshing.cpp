#include "seagullmesh.hpp"

#include <cmath>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <pybind11/functional.h>

#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/tangential_relaxation.h>
#include <CGAL/Polygon_mesh_processing/remesh_planar_patches.h>
#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>
#include <CGAL/Polygon_mesh_processing/Adaptive_sizing_field.h>
#include <CGAL/Polygon_mesh_processing/Uniform_sizing_field.h>
#include <CGAL/Polygon_mesh_processing/refine_mesh_at_isolevel.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>

typedef std::vector<V>                                      Verts;
typedef std::vector<F>                                      Faces;
typedef std::vector<E>                                      Edges;

typedef Mesh3::Property_map<V, Point3>                      VertPoint;
typedef Mesh3::Property_map<V, double>                      VertDouble;
typedef Mesh3::Property_map<V, bool>                        VertBool;
typedef Mesh3::Property_map<E, bool>                        EdgeBool;
typedef Mesh3::Property_map<F, F>                           FaceMap;

namespace PMP = CGAL::Polygon_mesh_processing;

typedef PMP::Principal_curvatures_and_directions<Kernel>    PrincipalCurvDir;
typedef Mesh3::Property_map<V, PrincipalCurvDir>            VertPrincipalCurvDir;


struct TouchedVertPoint {
    // Used for tracking which verts get moved during remesh, etc
    using key_type = V;
    using value_type = Point3;
    using reference = Point3&;
    using category = boost::read_write_property_map_tag;

    VertPoint& points;
    VertBool& touched;

    TouchedVertPoint();  // TODO is this required?
    TouchedVertPoint(VertPoint& p, VertBool& t) : points(p), touched(t) {}

    friend Point3& get (const TouchedVertPoint& map, V v) { return map.points[v]; }
    friend void put (const TouchedVertPoint& map, V v, const Point3& point) {
        map.points[v] = point;
        map.touched[v] = true;
    }
};


struct VertParamInterpolator {
    // Used for tracking which verts get moved during remesh, etc
    using key_type = V;
    using value_type = Point3;
    using reference = Point3&;
    using category = boost::read_write_property_map_tag;
    typedef std::function< std::tuple<double, double, double>(double, double) > SurfFn;

    Mesh3& mesh;
    SurfFn& surf_fn;
    VertPoint& points;
    VertBool& touched;
    VertDouble& t_map;
    VertDouble& theta_map;

    VertParamInterpolator(
            Mesh3& m, SurfFn& fn, VertPoint& p, VertBool& touched, VertDouble& t, VertDouble& theta
        ) : mesh(m), surf_fn(fn), points(p), touched(touched), t_map(t), theta_map(theta) {}


    friend Point3& get (const VertParamInterpolator& map, V v) { return map.points[v]; }
    friend void put (const VertParamInterpolator& map, V v, const Point3& point) {
        if ( map.theta_map[v] != -1) { return; }
        map.touched[v] = true;
        std::set<double> nbr_t;
        double this_t = -1.0;

        // Find what t_section this belongs to
        for (H h : halfedges_around_source(v, map.mesh)) {
            V w = map.mesh.target(h);  // Neighbor vertex
            double t = map.t_map[w];
            if ( nbr_t.count(t) ) {  // If more than one neighbor has this t
                this_t = t;
                break;
            }
            nbr_t.insert(t);
        }
        if ( this_t == -1.0 ) {
            throw std::runtime_error("couldnt figure out t");
        }

        // Avg theta values of t-neighbors
        double cos_theta = 0, sin_theta = 0;
        for (H h : halfedges_around_source(v, map.mesh)) {
            V w = map.mesh.target(h);  // Neighbor vertex
            if ( this_t == map.t_map[w] ) {
                double theta = map.theta_map[w];
                cos_theta += std::cos(theta);
                sin_theta += std::sin(theta);
            }
        }
        double this_theta = std::atan2(sin_theta, cos_theta);

        map.t_map[v] = this_t;
        map.theta_map[v] = this_theta;
        std::tuple<double, double, double> xyz = map.surf_fn(this_t, this_theta);
        map.points[v] = Point3(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz));
    }
};


void init_meshing(py::module &m) {

    m.def_submodule("meshing")
        .def("upsample_tube", [](
            Mesh3& mesh,
            double thresh,
            std::function<std::tuple<double, double, double>(double, double)>& surf_fn,
            VertBool& touched,
            VertDouble& t,
            VertDouble& theta,
            const Edges& edges
        ) {
            VertParamInterpolator interpolator(mesh, surf_fn, mesh.points(), touched, t, theta);
            auto params = PMP::parameters::vertex_point_map(interpolator);
            PMP::split_long_edges(edges, thresh, mesh, params);
        })
        .def("uniform_isotropic_remeshing", [](
                Mesh3& mesh,
                const Faces& faces,
                const double target_edge_length,
                unsigned int n_iter,
                bool protect_constraints,
                VertBool& vertex_is_constrained_map,
                EdgeBool& edge_is_constrained_map,
                VertBool& touched
            ) {

            TouchedVertPoint vertex_point_map(mesh.points(), touched);
            PMP::Uniform_sizing_field<Mesh3, TouchedVertPoint> sizing_field(target_edge_length, vertex_point_map);

            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .vertex_point_map(vertex_point_map)
                .protect_constraints(protect_constraints)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
            ;
            PMP::isotropic_remeshing(faces, sizing_field, mesh, params);
        })
        .def("adaptive_isotropic_remeshing", [](
                Mesh3& mesh,
                const Faces& faces,
                const double tolerance,
                const double ball_radius,
                const std::pair<double, double>& edge_len_min_max,
                unsigned int n_iter,
                bool protect_constraints,
                VertBool& vertex_is_constrained_map,
                EdgeBool& edge_is_constrained_map,
                VertBool& touched
            ) {

            TouchedVertPoint vertex_point_map(mesh.points(), touched);
            PMP::Adaptive_sizing_field<Mesh3, TouchedVertPoint> sizing_field(
                tolerance, edge_len_min_max, faces, mesh, PMP::parameters::vertex_point_map(vertex_point_map));

            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .vertex_point_map(vertex_point_map)
                .protect_constraints(protect_constraints)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
            ;
            PMP::isotropic_remeshing(faces, sizing_field, mesh, params);
        })
        .def("fair", [](Mesh3& mesh, const Verts& verts, const unsigned int fairing_continuity) {
            // A value controling the tangential continuity of the output surface patch.
            // The possible values are 0, 1 and 2, refering to the C0, C1 and C2 continuity.
            auto params = PMP::parameters::fairing_continuity(fairing_continuity);
            bool success = PMP::fair(mesh, verts, params);
            if (!success) {
                throw std::runtime_error("Fairing failed");
            }
        })
        .def("refine", [](Mesh3& mesh, const Faces& faces, double density) {
            std::vector<V> new_verts;
            std::vector<F> new_faces;
            auto params = PMP::parameters::density_control_factor(density);
            PMP::refine(mesh, faces, std::back_inserter(new_faces), std::back_inserter(new_verts), params);
            return std::make_tuple(new_verts, new_faces);
        })
        .def("smooth_angle_and_area", [](
            Mesh3& mesh, 
            const std::vector<F>& faces, 
            unsigned int n_iter,
            bool use_area_smoothing,
            bool use_angle_smoothing,
            bool use_safety_constraints,
            bool do_project, 
            VertBool& vertex_is_constrained_map,
            EdgeBool& edge_is_constrained_map
        ) {
            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .use_area_smoothing(use_area_smoothing)
                .use_angle_smoothing(use_angle_smoothing)
                .use_safety_constraints(use_safety_constraints)
                .do_project(do_project)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
            ;
            PMP::angle_and_area_smoothing(faces, mesh, params);
        })
        .def("tangential_relaxation", [](
            Mesh3& mesh,
            const std::vector<V>& verts,
            unsigned int n_iter,
            bool relax_constraints,
            VertBool& vertex_is_constrained_map,
            EdgeBool& edge_is_constrained_map
        ) {
            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .relax_constraints(relax_constraints)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
            ;
            PMP::tangential_relaxation(verts, mesh, params);
        })
        .def("smooth_shape", [](
            Mesh3& mesh, 
            const std::vector<F>& faces, 
            const double time, 
            unsigned int n_iter,
            VertBool& vertex_is_constrained_map
        ) {
            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .vertex_is_constrained_map(vertex_is_constrained_map)
            ;
            PMP::smooth_shape(faces, mesh, time, params);
        })
        .def("does_self_intersect", [](const Mesh3& mesh) {
            return PMP::does_self_intersect(mesh);
        })
        .def("self_intersections", [](const Mesh3& mesh) {
            std::vector<std::pair<F, F>> pairs;
            PMP::self_intersections(mesh, std::back_inserter(pairs));

            std::vector<F> first, second;
            boost::copy(pairs | boost::adaptors::transformed([](const auto& pair) { return pair.first; }), std::back_inserter(first));
            boost::copy(pairs | boost::adaptors::transformed([](const auto& pair) { return pair.second; }), std::back_inserter(second));

            return std::make_tuple(first, second);
        })
        .def("remove_self_intersections", [](Mesh3& mesh) {
            // returns a bool, presumably success
            return PMP::experimental::remove_self_intersections(mesh);
        })
        .def("remesh_planar_patches", [](
                const Mesh3& mesh,
                EdgeBool& edge_is_constrained_map,
                FaceMap& face_patch_map,
                float cosine_of_maximum_angle
            ) {
            // TODO parameters.face_patch_map wants propertymap<F, std::size_t>
            // But we've only exposed pmap<F, int>
            // Will doing both signed and unsigned ints be too complicated?
            auto params = PMP::parameters::
                edge_is_constrained_map(edge_is_constrained_map)
                // .face_patch_map(face_patch_map)
                .cosine_of_maximum_angle(cosine_of_maximum_angle)
            ;

            Mesh3 out;
            PMP::remesh_planar_patches(mesh, out, params);
            return out;
        })
        .def("interpolated_corrected_curvatures", [](
            const Mesh3& mesh,
            VertDouble& mean_curv_map,
            VertDouble& gauss_curv_map,
            VertPrincipalCurvDir& princ_curv_dir_map,
            const double ball_radius
        ) {
            auto params = PMP::parameters::
                vertex_mean_curvature_map(mean_curv_map)
                .vertex_Gaussian_curvature_map(gauss_curv_map)
                .vertex_principal_curvatures_and_directions_map(princ_curv_dir_map)
                .ball_radius(ball_radius)
            ;
            PMP::interpolated_corrected_curvatures(mesh, params);
        })
        .def("refine_mesh_at_isolevel", [](
            Mesh3& mesh,
            VertDouble& value_map,
            double isovalue,
            EdgeBool& edge_is_constrained_map
        ) {
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained_map);
            PMP::refine_mesh_at_isolevel(mesh, value_map, isovalue, params);
        })
    ;
}
