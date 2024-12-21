#include "seagullmesh.hpp"
#include "util.hpp"
#include <iostream>
#include <cmath>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef Mesh3::Property_map<V, double>  VertDouble;
typedef Mesh3::Property_map<F, bool>    FaceBool;


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
        // std::cout << "init face: theta_p " << theta_p << " theta_q " << theta_q << "\n";

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
            // std::cout << "theta_p " << theta_p << " theta_q " << theta_q << "\n";
        }
        // std::cout << "\n";

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

    private:
    /*
        todo: split_long_edges sets the coordinate immediately after splitting the edge
        and then calls sizing.register_split_vertex(vnew, mesh_)
        and then inserts new edges to maintain triangular mesh
        the new edges never get added to the queue
        so could use a cheap property map to only track the original radial/axial edges
    */
    struct ParametrizedVertexPointMap {
        // using Calculator = std::function<(double, double) std::tuple<double, double, double>>;
        using key_type = V;
        using value_type = Point3;
        using reference = Point3&;
        using category = boost::read_write_property_map_tag;

        Mesh3& mesh;
        // Calculator calculator;
        VertDouble& t_map;
        VertDouble& theta_map;

        ParametrizedVertexPointMap(Mesh3& m, /* Calculator c, */ VertDouble& t, VertDouble& theta)
            : mesh(m), /* calculator(c), */  t_map(t), theta_map(theta) {}

        friend Point3& get (const ParametrizedVertexPointMap& self, V v) { return self.mesh.points()[v]; }

        friend void put (const ParametrizedVertexPointMap& self, V v, const Point3& point) {
            double t_v, cos_theta_v, sin_theta_v;
            std::set<double> t_vals, theta_vals;

            for (V u : vertices_around_target(self.mesh.halfedge(v), self.mesh) ) {
                double t_u = self.t_map[u];
                if ( t_vals.insert(t_u).second ) { t_v += t_u; }
                double theta_u = self.theta_map[u];
                if ( theta_vals.insert(theta_u).second ) {
                    cos_theta_v += std::cos(theta_u);
                    sin_theta_v += std::sin(theta_u);
                }
            }

            self.t_map[v] = t_v / t_vals.size();
            self.theta_map[v] = CGAL_PI + std::atan2(sin_theta_v, cos_theta_v);
            self.mesh.points()[v] = point;
        }
    };

    public:
    void split_long_edges(double edge_length) {
        ParametrizedVertexPointMap vpm(mesh, t_map, theta_map);
        auto params = PMP::parameters::vertex_point_map(vpm);
        PMP::split_long_edges(mesh.edges(), edge_length, mesh, params);
    }
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
        .def("split_long_edges", &TubeMesher::split_long_edges)
    ;
}
