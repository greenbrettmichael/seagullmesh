#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>

typedef CGAL::Bbox_3 BBox3;
typedef CGAL::Aff_transformation_3<Kernel> Transform3;


Transform3 array_to_transform3(const py::array_t<double>& transform) {
    auto r = transform.unchecked<2>();
    if (r.shape(0) != r.shape(1)) {
        throw py::value_error("Must be a square matrix");
    }
    if (r.shape(0) == 3) {
        return Transform3(
            r(0, 0), r(0, 1), r(0, 2),
            r(1, 0), r(1, 1), r(1, 2),
            r(2, 0), r(2, 1), r(2, 2)
        );
    } else if (r.shape(0) == 4) {
        return Transform3(
            r(0, 0), r(0, 1), r(0, 2), r(0, 3),
            r(1, 0), r(1, 1), r(1, 2), r(1, 3),
            r(2, 0), r(2, 1), r(2, 2), r(2, 3),
                                       r(3, 3)
        );
    } else {
        throw py::value_error("Must be a 3x3 or 4x4 matrix");
    }
}

void init_mesh(py::module &m) {
    py::module sub = m.def_submodule("misc");

    py::class_<BBox3>(sub, "BoundingBox3", py::module_local())
        .def(py::init<double, double, double, double, double, double>())
        .def_property_readonly("x_min", &BBox3::xmin)
        .def_property_readonly("x_max", &BBox3::xmax)
        .def_property_readonly("y_min", &BBox3::ymin)
        .def_property_readonly("y_max", &BBox3::ymax)
        .def_property_readonly("x_min", &BBox3::zmin)
        .def_property_readonly("z_max", &BBox3::zmax)
        .def("diagonal", [](const BBox3& bbox) {
            return std::sqrt(
                CGAL::square(bbox.xmax() - bbox.xmin()) +
                CGAL::square(bbox.ymax() - bbox.ymin()) +
                CGAL::square(bbox.zmax() - bbox.zmin())
            );
        })
    ;

    sub
        .def("bounding_box", [](const Mesh3& mesh) {
            return PMP::bbox(mesh);
        })
        .def("transform", [](Mesh3& mesh, const py::array_t<double>& transform) {
            Transform3 t = array_to_transform3(transform);
            auto points = mesh.points();
            for (V v : mesh.vertices() ) {
                points[v] = t(points[v]);
            }
        })
        .def("vertices_to_faces", [](const Mesh3& mesh, const Indices<V>& verts) {
            std::set<F> faces;
            for (V v : verts.to_vector()) {
                // mesh.halfedge(v) returns an incoming halfedge of vertex v
                for (F f : faces_around_target(mesh.halfedge(v), mesh)) {
                    if (f != mesh.null_face()) {
                        faces.insert(f);
                    }
                }
            }
            return Indices<F>(std::vector<F>(faces.begin(), faces.end()));
        })
        .def("vertices_to_edges", [](const Mesh3& mesh, const Indices<V>& verts) {
            std::set<E> edges;
            for (V v : verts.to_vector()) {
                for (H h : halfedges_around_source(v, mesh)) {
                    edges.insert(mesh.edge(h));
                }
            }
            return Indices<E>(std::vector<E>(edges.begin(), edges.end()));
        })
        .def("faces_to_edges", [](const Mesh3& mesh, const Indices<F>& faces) {
            std::set<E> edges;
            for (F f : faces.to_vector()) {
                for (H h : halfedges_around_face(mesh.halfedge(f), mesh)) {
                    edges.insert(mesh.edge(h));
                }
            }
            return Indices<E>(std::vector<E>(edges.begin(), edges.end()));
        })
        .def("vertex_degrees", [](const Mesh3& mesh, const Indices<V>& verts) {
            return verts.map_to_array_of_scalars<Mesh3::size_type>(
                [&mesh](V v) { return mesh.degree(v); });
        })
        .def("face_normals", [](const Mesh3& mesh, const Indices<F>& faces) {
            return faces.map_to_array_of_vectors<3, F, Vector3>(
                [&mesh](F f) {return PMP::compute_face_normal(f, mesh);}
            );
        })
        .def("vertex_normals", [](const Mesh3& mesh, const Indices<V>& verts) {
            return verts.map_to_array_of_vectors<3, V, Vector3>(
                [&mesh](V v) {return PMP::compute_vertex_normal(v, mesh);}
            );
        })
        .def("face_areas", [](const Mesh3& mesh, const Indices<F>& faces) {
            return faces.map_to_array_of_scalars<F, double>(
                [&mesh](F f){ return PMP::face_area(f, mesh);}
            );
        })
        .def("volume", [](const Mesh3& mesh) {return PMP::volume(mesh);})
        .def("area", [](const Mesh3& mesh) {return PMP::area(mesh);})
    ;
}