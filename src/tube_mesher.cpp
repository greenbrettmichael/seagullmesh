#include "seagullmesh.hpp"
#include "util.hpp"
#include <iostream>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, double>  VertDouble;
typedef Mesh3::Property_map<F, bool>    FaceBool;


class TubeMesher {
    private:
    Mesh3& mesh;
    VertDouble& t_map;
    VertDouble& theta_map;
    FaceBool& is_cap_map;

    std::vector<V> verts;           // of next xs
    std::vector<H> radial_edges;    // of next xs

    H prev_xs;
    H next_xs;

    private:
    H add_edge(V u, V v) { return flip_normals ? mesh.add_edge(v, u) : mesh.add_edge(u, v); }
    H halfedge(V u, V v) { return flip_normals ? mesh.halfedge(v, u) : mesh.halfedge(u, v); }

    void add_points_and_radial_edges(
            const double t,
            const py::array_t<double>& theta,
            const py::array_t<double>& pts
    ) {
        const size_t n = pts.shape(0);
        auto r_pts = pts.unchecked<2>();
        auto r_theta = theta.unchecked<1>();

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
            radial_edges[i] = add_edge(verts[i], verts[i + 1]);
        }
        radial_edges[n] = radial_edges[0];

        for (size_t i = 0; i < n; ++i) {
            mesh.set_next(radial_edges[i], radial_edges[i + 1]);
            mesh.set_next(mesh.opposite(radial_edges[i + 1]), mesh.opposite(radial_edges[i]));
            mesh.set_halfedge(verts[i + 1], radial_edges[i]);
        }

        radial_edges.resize(n);  // More convenient for capping
    }
    H add_axial_face(H p, H q, H outgoing) {
        F f = mesh.add_face();
        mesh.set_face(p, f);
        mesh.set_face(outgoing, f);
        mesh.set_face(q, f);
        mesh.set_next(outgoing, q);
        mesh.set_next(p, outgoing);
        /*
        The outgoing edge connects two vertices on adjacent cross-sections
        with the same theta value. Incrementally walk in the positive theta
        direction until we find another pair of vertices with the same
        theta values and add an incoming edge to close the loop around the
        newly created face.

                    <== positive theta
                            q (halfedge on next_xs)
                    + <------------ +          positive t
                theta_q             ^              /\
                                    |              ||
                                    |  outgoing    ||
                theta_p             |              ||
                    +  -----------> +
                            p (halfedge on prev_xs)
        */
        double theta_p = theta_map[mesh.source(p)];
        double theta_q = theta_map[mesh.target(q)];
        while (theta_p != theta_q) {
            if ( (theta_p == 0) || (theta_q < theta_p ) ) {  // Further ahead on prev xs, so advance on next xs
                q = mesh.next(q);
                mesh.set_face(q, f);
                theta_q = theta_map[mesh.target(q)];
            } else {  // Further ahead on next xs, so advance on prev xs
                p = mesh.prev(p);
                mesh.set_face(p, f);
                theta_p = theta_map[mesh.source(p)];
            }
        }

        // Save state for next face
        prev_xs = mesh.prev(p);
        next_xs = mesh.next(q);

        // The axial edge from q back to p
        H incoming;
        if (theta_p == 0) {  // Should already exist -- the first axial edge
            incoming = halfedge(mesh.target(q), mesh.source(p));
        } else {
            incoming = add_edge(mesh.target(q), mesh.source(p));
        }
        mesh.set_face(incoming, f);
        mesh.set_next(q, incoming);
        mesh.set_next(incoming, p);
        mesh.set_halfedge(f, incoming);
        return incoming;
    }

    struct TriangulateCapVisitor : public PMP::Triangulate_faces::Default_visitor<Mesh3> {
        FaceBool& is_cap_map;
        bool is_cap;
        TriangulateCapVisitor(FaceBool& is_cap_map) : is_cap_map(is_cap_map) {}
        void before_subface_creations (F f) {is_cap = is_cap_map[f];}
        void after_subface_created(F f) {is_cap_map[f] = is_cap;}
    };

    void add_cap_face() {
        F f = mesh.add_face();
        if (nxs == 1) != (flip_normals) {  // The two caps have opposite orientation
            mesh.set_halfedge(f, radial_edges[0]);
            for (H h : radial_edges) { mesh.set_face(h, f); }
        } else {
            mesh.set_halfedge(f, mesh.opposite(radial_edges[0]));
            for (H h : radial_edges) { mesh.set_face(mesh.opposite(h), f); }
        }
        is_cap_map[f] = true;
    }

    public:
    void add_xs(double t, const py::array_t<double>& theta, const py::array_t<double>& pts) {
        nxs++;
        add_points_and_radial_edges(t, theta, pts);
        
        if (nxs == 1) {
            prev_xs = mesh.opposite(radial_edges[0]);
            if (closed) { add_cap_face(); }
            return;
        }

        // The first outgoing edge between adjacent cross-sections at theta=0
        next_xs = radial_edges[0];
        H outgoing = add_edge(mesh.target(prev_xs), mesh.source(next_xs));
        H incoming;

        do {
            incoming = add_axial_face(prev_xs, next_xs, outgoing);
            outgoing = mesh.opposite(incoming);
        } while (mesh.source(incoming) != verts[0]);

        prev_xs = mesh.opposite(radial_edges[0]);
    }
    void finish() {
        if (closed){ add_cap_face(); }
        if (triangulate) {
            TriangulateCapVisitor visitor(is_cap_map);
            PMP::triangulate_faces(mesh, PMP::parameters::visitor(visitor));
        }
    }

    TubeMesher(
        Mesh3& mesh, VertDouble& t_map, VertDouble& theta_map, FaceBool& is_cap_map,
        bool closed, bool triangulate, bool flip_normals
    )
    : mesh(mesh), t_map(t_map), theta_map(theta_map), is_cap_map(is_cap_map),
        closed(closed), triangulate(triangulate), flip_normals(flip_normals), nxs(0) {}

    size_t nxs;
    bool closed;
    bool triangulate;
    bool flip_normals;
};

void init_tube_mesher(py::module &m) {
    py::module sub = m.def_submodule("tube_mesher");

    py::class_<TubeMesher>(sub, "TubeMesher")
        .def(py::init<Mesh3&, VertDouble&, VertDouble&, FaceBool&, bool, bool, bool>())
        .def("add_xs", &TubeMesher::add_xs)
        .def("finish", &TubeMesher::finish)
        .def_readonly("nxs", &TubeMesher::nxs)
        .def_readwrite("closed", &TubeMesher::closed)
        .def_readwrite("triangulate", &TubeMesher::triangulate)
        .def_readwrite("flip_normals", &TubeMesher::flip_normals)
    ;
}
