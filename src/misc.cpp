#include "seagullmesh.hpp"
#include "util.hpp"

#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>

typedef CGAL::Bbox_3 BBox3;
typedef CGAL::Aff_transformation_3<Kernel> Transform3;


void init_mesh(py::module &m) {
    py::module sub = m.def_submodule(