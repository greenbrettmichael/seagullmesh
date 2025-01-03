#include "seagullmesh.hpp"
#include "util.hpp"
#include <iostream>
#include <cmath>
#include <pybind11/functional.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, double>  VertDouble;
typedef Mesh3::Property_map<F, bool>    FaceBool;


class TubeInterpolator {
    std::vector<double> ts;
    std::vector<double> thetas;
    const py::array_t<double>& surf;
    const py::detail::unchecked_reference<double, 3> rsurf;

    public:
    TubeInterpolator(
        std::vector<double> ts,
        std::vector<double> thetas,
        const py::array_t<double, py::array::c_style>& surf
    ) : ts(ts), thetas(thetas), surf(surf), rsurf(surf.unchecked<3>()) {
        CGAL_assertion(surf.shape(0) == ts.size());
        CGAL_assertion(surf.shape(1) == thetas.size());
        CGAL_assertion(surf.shape(2) == 3);
    }

    Point3 interpolate(double t, double theta) const {
        double t1, t2;
        size_t t1_idx, t2_idx;
        auto t_it = std::lower_bound(ts.begin(), ts.end(), t);
        t2 = *t_it;
        t2_idx = t_it - ts.begin();
        t1 = *(--t_it);
        t1_idx = t2_idx - 1;

        double theta1, theta2;
        size_t theta1_idx, theta2_idx;
        auto theta_it = std::lower_bound(thetas.begin(), thetas.end(), theta);
        if ( theta_it == thetas.end() ) {
            theta1 = *(--theta_it);
            theta1_idx = thetas.size() - 1;
            theta2 = 2 * CGAL_PI;
            theta2_idx = 0;
        } else {
            theta2 = *theta_it;
            theta2_idx = theta_it - thetas.begin();
            theta1 = *(--theta_it);
            theta1_idx = theta2_idx - 1;
        }

        double dt1 = t - t1;
        double dt2 = t2 - t;
        double dtheta1 = t - theta1;
        double dtheta2 = theta2 - theta;
        double d_t_theta = (t2 - t1) * (theta2 - theta1);

        double w11 = dt2 * dtheta2;
        double w12 = dt2 * dtheta1;
        double w21 = dt1 * dtheta2;
        double w22 = dt1 * dtheta1;

        double xyz[3] = {0.0, 0.0, 0.0};

        for (size_t i = 0; i < 3; ++i) {
            xyz[i] = (
                  w11 * rsurf(t1_idx, theta1_idx, i)
                + w12 * rsurf(t1_idx, theta2_idx, i)
                + w21 * rsurf(t2_idx, theta1_idx, i)
                + w22 * rsurf(t2_idx, theta2_idx, i)
            ) / d_t_theta;
        }

        return Point3(xyz[0], xyz[1], xyz[2]);
    }
};


class TubeMesher {
    public:
    size_t nxs;
    bool closed;
    bool triangulate;
    bool flip_normals;

    private:
    Mesh3& mesh;
    VertDouble& t_map;
    VertDouble& theta_map;
    FaceBool& is_cap_map;

    std::vector<V> verts;           // of next xs
    std::vector<H> radial_edges;    // of next xs

    H prev_radial_edge;             // on prev_xs
    H next_radial_edge;             // on next_xs

    public:
    TubeMesher(
        Mesh3& mesh, VertDouble& t_map, VertDouble& theta_map, FaceBool& is_cap_map,
        bool closed, bool triangulate, bool flip_normals
    )
    : nxs(0), closed(closed), triangulate(triangulate), flip_normals(flip_normals),
        mesh(mesh), t_map(t_map), theta_map(theta_map), is_cap_map(is_cap_map) {}

    private:
    void add_points_and_radial_edges(
            const double t,
            const py::array_t<double>& theta,
            const py::array_t<double>& pts
    ) {
        const size_t n = pts.shape(0);
        auto r_pts = pts.unchecked<2>();
        auto r_theta = theta.unchecked<1>();
        CGAL_assertion(r_theta(0) == 0);
        CGAL_assertion(pts.shape(0) == theta.shape(0));

        verts.resize(n + 1);
        for (size_t i = 0; i < n; ++i) {
            V v = mesh.add_vertex(Point3(r_pts(i, 0), r_pts(i, 1), r_pts(i, 2)));
            verts[i] = v;
            t_map[v] = t;
            theta_map[v] = r_theta(i);
        }
        verts[n] = verts[0];

        radial_edges.resize(n + 1);
        for (size_t i = 0; i < n; ++i) {
            H h = mesh.add_edge(verts[i], verts[i + 1]);
            radial_edges[i] = h;
            mesh.set_halfedge(verts[i + 1], h);
        }
        radial_edges[n] = radial_edges[0];

        for (size_t i = 0; i < n; ++i) {
            mesh.set_next(radial_edges[i], radial_edges[i + 1]);
            mesh.set_next(mesh.opposite(radial_edges[i + 1]), mesh.opposite(radial_edges[i]));
        }
    }
    H add_axial_face(H p, H q, H incoming) {
        /*
        The incoming edge connects two vertices on adjacent cross-sections
        with the same theta value. Incrementally walk in the positive theta
        direction until we find another pair of vertices with the same
        theta values and add an outgoing axial edge to close the loop around the
        newly created face.

        positive t
            ^                   q (halfedge on next_xs)
            |             +<----------- +
            |             |          theta_q
            |             |
            |    incoming |
            |             v          theta_p
            |             +-----------> +
            |                   p (halfedge on prev_xs)
            |
             -----------------------> positive theta
        */
        
        F f = mesh.add_face();
        mesh.set_face(q, f);
        mesh.set_face(incoming, f);
        mesh.set_face(p, f);
        
        mesh.set_next(q, incoming);
        mesh.set_next(incoming, p);
        
        double theta_q = theta_map[mesh.source(q)];
        double theta_p = theta_map[mesh.target(p)];

        while (theta_p != theta_q) {
            if ( theta_q != 0 && (theta_p > theta_q || theta_p == 0)) {  // Further ahead on prev xs, so advance on next xs
                q = mesh.prev(q);
                mesh.set_face(q, f);
                double theta_q1 = theta_map[mesh.source(q)];
                CGAL_assertion((theta_q1 > theta_q) || (theta_q1 == 0));
                theta_q = theta_q1;
            } else if (theta_p != 0 && (theta_q > theta_p || theta_q == 0)) {  // Further ahead on next xs, so advance on prev xs
                p = mesh.next(p);
                mesh.set_face(p, f);
                double theta_p1 = theta_map[mesh.target(p)];
                CGAL_assertion((theta_p1 > theta_p) || (theta_p1 == 0));
                theta_p = theta_p1;
            } else {
                throw std::runtime_error("theta values do not align");
            }
        }

        // Save state for next face
        prev_radial_edge = mesh.next(p);
        next_radial_edge = mesh.prev(q);

        // The axial edge from q back to p
        H outgoing;
        if (theta_p == 0) {  // Should already exist -- the first axial edge
            outgoing = mesh.halfedge(mesh.target(p), mesh.source(q));
        } else {
            outgoing = mesh.add_edge(mesh.target(p), mesh.source(q));
        }
        mesh.set_next(p, outgoing);
        mesh.set_next(outgoing, q);
        mesh.set_face(outgoing, f);
        mesh.set_halfedge(f, outgoing);
        return outgoing;
    }

    struct TriangulateTubeVisitor : public PMP::Triangulate_faces::Default_visitor<Mesh3> {
        FaceBool& is_cap_map;
        bool is_cap;
        TriangulateTubeVisitor(FaceBool& is_cap_map) : is_cap_map(is_cap_map) {}
        void before_subface_creations (F f) {is_cap = is_cap_map[f];}
        void after_subface_created(F f) {is_cap_map[f] = is_cap;}
    };

    void add_cap_face(H h) {
        F f = mesh.add_face();
        mesh.set_halfedge(f, h);
        is_cap_map[f] = true;
        H h0 = h;

        do {
            CGAL_assertion(mesh.face(h) == Mesh3::null_face());
            mesh.set_face(h, f);
            h = mesh.next(h);
        } while (h != h0);
    }

    public:
    void add_xs(double t, const py::array_t<double>& theta, const py::array_t<double>& pts) {
        add_points_and_radial_edges(t, theta, pts);
        
        if (nxs > 0) {
            // The first incoming edge between adjacent cross-sections at theta=0
            next_radial_edge = mesh.opposite(radial_edges[0]);
            H incoming = mesh.add_edge(verts[0], mesh.source(prev_radial_edge));
            H outgoing;

            do {
                outgoing = add_axial_face(prev_radial_edge, next_radial_edge, incoming);
                incoming = mesh.opposite(outgoing);
            } while (mesh.target(outgoing) != verts[0]);
        }

        nxs += 1;
        prev_radial_edge = radial_edges[0];
    }
    void finish() {
        if (flip_normals) { PMP::reverse_face_orientations(mesh); }
        if (closed) {
            // Extract halfedges first, then cap.
            // (Not sure if it's safe to add faces while iterating over boundaries)
            std::vector<H> boundary_cycles;
            PMP::extract_boundary_cycles(mesh, std::back_inserter(boundary_cycles));
            for (H h : boundary_cycles) {
                add_cap_face(h);
            }
        }
        if (triangulate) {
            TriangulateTubeVisitor visitor(is_cap_map);
            PMP::triangulate_faces(mesh, PMP::parameters::visitor(visitor));
        }
    }

    // (t, theta) -> Point
    using Calculator = std::function<Point3 (double, double)>;

    std::pair<double, double> parametric_midpoint(const V v0, const V v1) const {
        const double t0 = t_map[v0];
        const double t1 = t_map[v1];
        double t_mid, theta_mid;

        if (t0 == t1) { // a radial edge
            t_mid = t0;
            const double theta0 = theta_map[v0];
            const double theta1 = theta_map[v1];

            theta_mid = std::atan2(
                std::sin(theta0) + std::sin(theta1),
                std::cos(theta0) + std::cos(theta1)
            );
            if ( theta_mid < 0 ) { theta_mid += 2 * CGAL_PI; }  // remap [-pi, pi] to [0, 2pi]
        } else { // an axial edge
            theta_mid = theta_map[v0];
            t_mid = (t0 + t1) / 2.0;
        }

        return {t_mid, theta_mid};
    }

    public:
    void split_long_edges(double edge_length, Calculator calculator) {
        double sq_thresh = edge_length * edge_length;
        typedef std::pair<H, double> H_and_sql;
        auto vpm = mesh.points();

        // Collect long edges
        std::multiset< H_and_sql, std::function<bool(H_and_sql, H_and_sql)> >
            long_edges(
                [](const H_and_sql& p1, const H_and_sql& p2)
                { return p1.second > p2.second; }
            );

        for (E e : mesh.edges()) {
            H h = mesh.halfedge(e);
            double sqlen = PMP::squared_edge_length(h, mesh);
            if (sqlen > sq_thresh) {
                long_edges.emplace(h, sqlen);
            }
        }

        // Split edges
        while (!long_edges.empty()) {
            // The edge with longest length
            auto eit = long_edges.begin();
            H h = eit->first;
            long_edges.erase(eit);

            // Split edge
            const V v0 = mesh.source(h);
            const V v1 = mesh.target(h);
            auto [t_mid, theta_mid] = parametric_midpoint(v0, v1);
            H h_new = CGAL::Euler::split_edge(h, mesh);
            V v_mid = mesh.target(h_new);
            t_map[v_mid] = t_mid;
            theta_map[v_mid] = theta_mid;
            vpm[v_mid] = calculator(t_mid, theta_mid);

            // Check the subedges
            if (double sqlen = PMP::squared_edge_length(h_new, mesh); sqlen > sq_thresh) {
                long_edges.emplace(h_new, sqlen);
            }
            H h_next = mesh.next(h_new);
            if (double sqlen = PMP::squared_edge_length(h_next, mesh); sqlen > sq_thresh) {
                long_edges.emplace(h_next, sqlen);
            }
        }
    }
};

void init_tube_mesher(py::module &m) {
    py::module sub = m.def_submodule("tube_mesher");

    py::class_<TubeInterpolator>(sub, "TubeInterpolator")
        .def(py::init<
                std::vector<double>,
                std::vector<double>,
                const py::array_t<double, py::array::c_style>&
            >())
        .def("interpolate", &TubeInterpolator::interpolate)
    ;

    py::class_<TubeMesher>(sub, "TubeMesher")
        .def(py::init<Mesh3&, VertDouble&, VertDouble&, FaceBool&, bool, bool, bool>())
        .def("add_xs", &TubeMesher::add_xs)
        .def("finish", &TubeMesher::finish)
        .def_readonly("nxs", &TubeMesher::nxs)
        .def_readwrite("closed", &TubeMesher::closed)
        .def_readwrite("triangulate", &TubeMesher::triangulate)
        .def_readwrite("flip_normals", &TubeMesher::flip_normals)
//        .def("split_long_edges", &[](TubeMesher& tm, double edge_length, TubeMesher::Calculator calculator){
//            return tm.split_long_edges(edge_length, calculator);
//        })
//        .def("split_long_edges2", &[](TubeMesher& tm, double edge_length, const TubeInterpolator& interpolator) {
//            auto calculator = [&](double t, double theta) { return interpolator.interpolate(t, theta); };
//            return tm.split_long_edges(edge_length, calculator);
//        })
    ;
}
