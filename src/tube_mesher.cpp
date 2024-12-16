#include "seagullmesh.hpp"
#include "util.hpp"
#include <iostream>
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

    std::vector<V> verts;           // of next xs
    std::vector<H> radial_edges;    // of next xs

    H prev_radial_edge;             // on prev_xs
    H next_radial_edge;             // on next_xs

    void add_points_and_radial_edges(
            const double t,
            const py::array_t<double>& theta,
            const py::array_t<double>& pts
    ) {
        const size_t n = pts.shape(0);
        auto r_pts = pts.unchecked<2>();
        auto r_theta = theta.unchecked<1>();
        CGAL_assertion(r_theta(0) == 0);

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
            if ( (theta_q < theta_p) || (theta_p == 0)) {  // Further ahead on prev xs, so advance on next xs
                q = mesh.prev(q);
                mesh.set_face(q, f);
                theta_q = theta_map[mesh.source(q)];
            } else if ( (theta_p < theta_q) || (theta_q == 0)) {  // Further ahead on next xs, so advance on prev xs
                p = mesh.next(p);
                mesh.set_face(p, f);
                theta_p = theta_map[mesh.target(p)];
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

    void add_cap_face(H radial_edge) {
        F f = mesh.add_face();
        is_cap_map[f] = true;
        
        for (H h : CGAL::halfedges_around_face(radial_edge, mesh)) { // border loop around the null face
            mesh.set_face(h, f);
        }
    }

    public:
    void add_xs(double t, const py::array_t<double>& theta, const py::array_t<double>& pts) {
        nxs++;
        add_points_and_radial_edges(t, theta, pts);
        
        if (nxs == 1) {
            prev_radial_edge = radial_edges[0];
            if (closed) { add_cap_face(mesh.opposite(prev_radial_edge)); }
            return;
        }

        // The first incoming edge between adjacent cross-sections at theta=0
        next_radial_edge = mesh.opposite(radial_edges[0]);
        H incoming = mesh.add_edge(verts[0], mesh.source(prev_radial_edge));
        H outgoing;

        do {
            outgoing = add_axial_face(prev_radial_edge, next_radial_edge, incoming);
            incoming = mesh.opposite(outgoing);
        } while (mesh.target(outgoing) != verts[0]);

        prev_radial_edge = radial_edges[0];
    }
    void finish() {
        if (closed){ add_cap_face(radial_edges[0]); }
        if (flip_normals) { PMP::reverse_face_orientations(mesh); }
        if (triangulate) {
            TriangulateTubeVisitor visitor(is_cap_map);
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
