#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>

typedef CGAL::Bbox_3 BBox3;
typedef CGAL::Aff_transformation_3<Kernel> Transform3;


namespace PMP = CGAL::Polygon_mesh_processing;


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

void init_geometry(py::module &m) {
    py::module sub = m.def_submodule("geometry");

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
        .def("face_normals", [](const Mesh3& mesh, const Indices<F>& faces) {
            return faces.map_to_array_of_vectors<Vector3, 3, double>(
                [&mesh](F f) {return PMP::compute_face_normal(f, mesh);}
            );
        })
        .def("vertex_normals", [](const Mesh3& mesh, const Indices<V>& verts) {
            return verts.map_to_array_of_vectors<Vector3, 3, double>(
                [&mesh](V v) {return PMP::compute_vertex_normal(v, mesh);}
            );
        })
        .def("face_areas", [](const Mesh3& mesh, const Indices<F>& faces) {
            return faces.map_to_array_of_scalars<double>(
                [&mesh](F f){ return PMP::face_area(f, mesh);}
            );
        })
        .def("edge_lengths", [](const Mesh3& mesh, const Indices<E>& edges) {
            return edges.map_to_array_of_scalars<double>(
                [&mesh](E e){ return PMP::edge_length(e, mesh); }
            );
        })
        .def("volume", [](const Mesh3& mesh) {return PMP::volume(mesh);})
        .def("area", [](const Mesh3& mesh) {return PMP::area(mesh);})
        .def("bounding_box", [](const Mesh3& mesh) { return PMP::bbox(mesh); })
        .def("transform", [](Mesh3& mesh, const py::array_t<double>& transform) {
            Transform3 t = array_to_transform3(transform);
            auto points = mesh.points();
            for (V v : mesh.vertices() ) {
                points[v] = t(points[v]);
            }
        })
        .def("does_bound_a_volume", [](const Mesh3& mesh) {
            return PMP::does_bound_a_volume(mesh);
        })
        .def("is_outward_oriented", [](const Mesh3& mesh) {
            return PMP::is_outward_oriented(mesh);
        })
    ;
}