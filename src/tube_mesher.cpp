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
    bool closed;
    bool triangulate;
    size_t nxs;

    H prev_xs;
    H next_xs;
    std::vector<V> verts;
    std::vector<H> edges;

    public:
    TubeMesher(
            Mesh3& mesh, VertDouble& t_map, VertDouble& theta_map, FaceBool& is_cap_map,
            bool closed, bool triangulate
        )
        : mesh(mesh), t_map(t_map), theta_map(theta_map), is_cap_map(is_cap_map),
            closed(closed), triangulate(triangulate), nxs(0) {}

    size_t get_nxs() { return nxs; }


    private:
    H add_points_and_radial_edges(
            const double t,
            const py::array_t<double>& theta,
            const py::array_t<double>& pts
    ) {
        nxs++;
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

        for (V v : verts) {
            std::cout << v << " ";
        }
        std::cout << "\n";

        edges.resize(n + 1);
        for (size_t i = 0; i < n; ++i) {
            // Sets the targets of the halfedge to the given vertices,
            // but does not modify the halfedge associated to the vertices.
            std::cout << " edge " << i << " " << verts[i] << " " << verts[i + 1] << "\n";
            edges[i] = mesh.add_edge(verts[i], verts[i + 1]);
        }
        edges[n] = edges[0];

        for (H h : edges) {
            std::cout << h << " ";
        }
        std::cout << "\n";

        for (size_t i = 0; i < n; ++i) {
            mesh.set_next(edges[i], edges[i + 1]);
            mesh.set_next(mesh.opposite(edges[i + 1]), mesh.opposite(edges[i]));
            mesh.set_halfedge(verts[i + 1], edges[i]);  // sets the incoming halfedge of vertex v to h.
            //mesh.set_halfedge(verts[i], mesh.opposite(edges[i]));
            CGAL_assertion(mesh.is_valid(verts[i], true));
            CGAL_assertion(mesh.is_valid(verts[i + 1], true));
        }

        for (H h : mesh.halfedges() ) {
            std::cout << h << " next: " << mesh.next(h) << "\n";
        }

        for (H h : mesh.halfedges() ) {
            std::cout << h << " " << mesh.source(h) << "->" << mesh.target(h) << "\n";
        }
        for (V v : mesh.vertices() ) {
            std::cout << v << ": " << mesh.halfedge(v) << "\n";
        }
        for (H h : mesh.halfedges() ) {
            CGAL_assertion(mesh.is_valid(h, true));
        }
        for (E e : mesh.edges() ) {
            CGAL_assertion(mesh.is_valid(e, true));
        }
        for (V v : mesh.vertices() ) {
            CGAL_assertion(mesh.is_valid(v, true));
        }

        V v0 = verts[0];
        H h = mesh.halfedge(v0);
        V v1 = mesh.target(h);
        std::cout << "v0 " << v0 << " its h: " << h << " that h's target: " << v1 << "\n";
        CGAL_assertion( v0 == v1 );
        CGAL_assertion( mesh.is_valid(true) );
        return edges[0];
    }
    H add_axial_face(H p, H q) {
        std::cout << "nxs " << nxs << " " << mesh.number_of_vertices();

        F f = mesh.add_face();
        H outgoing = mesh.add_edge(mesh.source(prev_xs), mesh.source(next_xs));
        CGAL_assertion( outgoing != Mesh3::null_halfedge() );
        mesh.set_face(mesh.opposite(p), f);
        mesh.set_face(outgoing, f);
        mesh.set_face(q, f);
        mesh.set_next(mesh.opposite(p), outgoing);
        mesh.set_next(outgoing, q);

        /*
                    <== positive theta
                            q
                    + <------------ +
                theta_q             ^
                                    |
                                    |  outgoing
                theta_p             |
                    + <------------ +
                            p
        */
        double theta_p = theta_map[mesh.target(p)];
        double theta_q = theta_map[mesh.target(q)];

        while (theta_p != theta_q) {
            if ( (theta_p == 0) || (theta_q < theta_p ) ) {  // Further ahead on prev xs, so advance on next xs
                q = mesh.next(q);
                CGAL_assertion( q != Mesh3::null_halfedge() );
                CGAL_assertion( mesh.target(q) != Mesh3::null_vertex() );
                mesh.set_face(q, f);
                theta_q = theta_map[mesh.target(q)];
            } else {  // Further ahead on next xs, so advance on prev xs
                p = mesh.next(p);
                CGAL_assertion( p != Mesh3::null_halfedge() );
                CGAL_assertion( mesh.target(p) != Mesh3::null_vertex() );
                mesh.set_face(mesh.opposite(p), f);
                theta_p = theta_map[mesh.target(p)];
            }
        }

        // Save state for next iter
        prev_xs = mesh.next(p);
        next_xs = mesh.next(q);
        CGAL_assertion( next_xs != Mesh3::null_halfedge() );
        CGAL_assertion( prev_xs != Mesh3::null_halfedge() );

        // The axial edge from q back to p
        H incoming;
        if (theta_p == 0) {  // Should already exist -- the first axial edge
            incoming = mesh.halfedge(mesh.target(q), mesh.target(p));
        } else {
            incoming = mesh.add_edge(mesh.target(q), mesh.target(p));
        }
        CGAL_assertion( incoming != Mesh3::null_halfedge() );
        mesh.set_face(incoming, f);
        mesh.set_next(q, incoming);
        mesh.set_next(incoming, mesh.opposite(p));
        mesh.set_halfedge(f, outgoing);

        if (closed) { PMP::triangulate_face(f, mesh); }
        return incoming;
    }

    struct TriangulateCapVisitor : public PMP::Triangulate_faces::Default_visitor<Mesh3> {
        FaceBool& is_cap_map;
        TriangulateCapVisitor(FaceBool& is_cap_map) : is_cap_map(is_cap_map) {}
        void after_subface_created(F f) {is_cap_map[f] = true;}
    };

    void add_cap_face(bool is_t_0) {
        F f = mesh.add_face();

        // todo note this includes an extra edge because of the circular loop
        if (is_t_0) {
            for (H h : edges) {
                mesh.set_face(mesh.opposite(h), f);
            }
            mesh.set_halfedge(f, mesh.opposite(edges[0]));
        } else {
            for (H h : edges) {
                mesh.set_face(h, f);
            }
            mesh.set_halfedge(f, edges[0]);
        }

        if (triangulate) {
            PMP::triangulate_face(f, mesh, PMP::parameters::visitor(TriangulateCapVisitor(is_cap_map)));
        } else {
            is_cap_map[f] = true;
        }
    }

    public:
    void add_xs(double t, const py::array_t<double>& theta, const py::array_t<double>& pts) {
        if (nxs == 0) {
            prev_xs = add_points_and_radial_edges(t, theta, pts);
            if (closed) { add_cap_face(true);}
            return;
        }

        // Writes to the verts and edges vectors
        next_xs = add_points_and_radial_edges(t, theta, pts);
        H incoming;
        do {
            incoming = add_axial_face(prev_xs, next_xs);
        } while (mesh.target(incoming) != verts[0]);

        prev_xs = edges[0];  // reset prev_xs to the first radial edge on next_xs
    }
    void finish() {
        if (closed) { add_cap_face(false);}
    }
};

void init_tube_mesher(py::module &m) {
    py::module sub = m.def_submodule("tube_mesher");

    py::class_<TubeMesher>(sub, "TubeMesher")
        .def(py::init<Mesh3&, VertDouble&, VertDouble&, FaceBool&, bool, bool>())
        .def("add_xs", &TubeMesher::add_xs)
        .def("finish", &TubeMesher::finish)
        .def_property_readonly("nxs", &TubeMesher::get_nxs)
    ;
}
