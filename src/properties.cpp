#include "seagullmesh.hpp"

#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>
typedef CGAL::Polygon_mesh_processing::Principal_curvatures_and_directions<Kernel>    PrincipalCurvDir;

template <typename Key, typename Val>
auto add_property_map(Mesh3& mesh, std::string name, const Val default_val) {
    typename Mesh3::Property_map<Key, Val> pmap;
    bool created;
    std::tie(pmap, created) = mesh.add_property_map<Key, Val>(name, default_val);
    if (!created) {
        throw std::runtime_error("Property map already exists");
    }
    return pmap;
}


template <typename Key, typename Val>
auto define_property_map(py::module &m, std::string name, bool is_scalar = true) {
    using PMap = typename Mesh3::Property_map<Key, Val>;
    m.def("remove_property_map", [](Mesh3& mesh, PMap& pmap) {
        mesh.remove_property_map(pmap);
    });

    return py::class_<PMap>(m, name.c_str(), py::buffer_protocol())
        .def(py::init([](Mesh3& mesh, std::string name, const Val default_val) {
                return add_property_map<Key, Val>(mesh, name, default_val);
            })
        )
        .def("get_property_map", [](const Mesh3& mesh, const std::string& name) {
            return mesh.property_map<Key, Val>(name);  // returns std::optional<pmap>
        })
        .def_property_readonly_static("is_sgm_property_map", [](py::object /* self */) { return true; })
        .def_property_readonly_static("is_scalar", [is_scalar](py::object /* self */) { return is_scalar; })

        // single key, single value
        .def("__getitem__", [](const PMap& pmap, const Key key) {
            Val val = pmap[key];
            return val;
        })
        .def("__setitem__", [](PMap& pmap, const Key key, const Val val) { pmap[key] = val; })

        // array of keys, single value
        .def("__setitem__", [](PMap& pmap, const Indices<Key>& indices, const Val val) {
            indices.apply([&pmap, val] (Key k) { pmap[k] = val; });
        })
        // array of keys, vector of values
        .def("__setitem__", [](PMap& pmap, const Indices<Key>& indices, const std::vector<Val>& vals) {
            indices.apply([&pmap, &vals](size_t i, Key k) { pmap[k] = vals[i]; });
        })
        // array of keys, return vector<Val>
        .def("get_vector", [](const PMap& pmap, const Indices<Key>& indices) {
            return indices.map_to_vector<Val>([&pmap](Key k) { return pmap[k]; });
        })
    ;
}

template <typename Key, typename Val>
void define_scalar_property_map(py::module &m, std::string name) {
    using PMap = typename Mesh3::Property_map<Key, Val>;
    define_property_map<Key, Val>(m, name, true)
        // get array of keys, return array of values
        .def("__getitem__", [](const PMap& pmap, const Indices<Key>& indices) {
            return indices.map_to_array_of_scalars<Val>([&pmap](Key k) { return pmap[k]; });
        })
        // set array of keys from array of values
        .def("__setitem__", [](PMap& pmap, const Indices<Key>& indices, const py::array_t<Val>& vals) {
            auto r = vals.unchecked<1>();
            indices.apply([&pmap, &r](size_t i, Key k) { pmap[k] = r(i); });
        })
    ;
}

// For Point2/3 and Vector2/3
// template variable U here could be hardcoded to double but left generic for future int-valued vectors
template <typename Key, typename Val, size_t Dim, typename U>
void define_array_property_map(py::module &m, std::string name) {
    using PMap = typename Mesh3::Property_map<Key, Val>;

    define_property_map<Key, Val>(m, name, false)
        // Get array of indices, return 2d array of values
        .def("__getitem__", [](const PMap& pmap, const Indices<Key>& indices) {
            return indices.map_to_array_of_vectors<Val, Dim, U>( [&pmap](Key k) { return pmap[k]; });
        })
        // Set array of indices from array of values
        .def("__setitem__", [](PMap& pmap, const Indices<Key>& indices, const py::array_t<U>& vals) {
            auto r = vals.unchecked<2>();
            indices.apply([&pmap, &r](size_t i, Key k) {
                if constexpr ( Dim == 3 ) {
                    pmap[k] = Val(r(i, 0), r(i, 1), r(i, 2));
                } else {
                    pmap[k] = Val(r(i, 0), r(i, 1));
                }
            });
        })
    ;
}


struct ExpandedPrincipalCurvaturesAndDirections {
    py::array_t<double> min_curvature;
    py::array_t<double> max_curvature;
    py::array_t<double> min_direction;
    py::array_t<double> max_direction;
};


template <typename Key>
void define_princ_curv_dir_property_map(py::module &m, std::string name) {
    using PMap = typename Mesh3::Property_map<Key, PrincipalCurvDir>;

    define_property_map<Key, PrincipalCurvDir>(m, name, false)
        .def("__getitem__", [](const PMap& pmap, const Indices<Key>& indices) {
            return indices.map_to_vector<PrincipalCurvDir>([&pmap] (Key k) { return pmap[k]; });
        })
        .def("get_expanded", [](const PMap& pmap, const Indices<Key>& indices) {
            const size_t n = indices.size();
            py::array_t<double> min_curvature({int(n)});
            py::array_t<double> max_curvature({int(n)});
            py::array_t<double> min_direction({n, size_t(3)});
            py::array_t<double> max_direction({n, size_t(3)});

            auto r_min_cur = min_curvature.mutable_unchecked<1>();
            auto r_max_cur = max_curvature.mutable_unchecked<1>();
            auto r_min_dir = min_direction.mutable_unchecked<2>();
            auto r_max_dir = max_direction.mutable_unchecked<2>();

            indices.apply([&pmap, &r_min_cur, &r_max_cur, &r_min_dir, &r_max_dir] (size_t i, Key k){
                PrincipalCurvDir x = pmap[k];
                r_min_cur(i) = x.min_curvature;
                r_max_cur(i) = x.max_curvature;
                for (size_t j = 0; j < 3; ++j) {
                    r_min_dir(i, j) = x.min_direction[j];
                    r_max_dir(i, j) = x.max_direction[j];
                }
            });

            return ExpandedPrincipalCurvaturesAndDirections{
                min_curvature, max_curvature, min_direction, max_direction};
        })
    ;
}


void init_properties(py::module &m) {
    py::module sub = m.def_submodule("properties");
    define_scalar_property_map<V, bool     >(sub, "V_bool_PropertyMap");
    define_scalar_property_map<F, bool     >(sub, "F_bool_PropertyMap");
    define_scalar_property_map<E, bool     >(sub, "E_bool_PropertyMap");
    define_scalar_property_map<H, bool     >(sub, "H_bool_PropertyMap");

    define_scalar_property_map<V, int32_t  >(sub, "V_int32_PropertyMap");
    define_scalar_property_map<F, int32_t  >(sub, "F_int32_PropertyMap");
    define_scalar_property_map<E, int32_t  >(sub, "E_int32_PropertyMap");
    define_scalar_property_map<H, int32_t  >(sub, "H_int32_PropertyMap");

    define_scalar_property_map<V, int64_t  >(sub, "V_int64_PropertyMap");
    define_scalar_property_map<F, int64_t  >(sub, "F_int64_PropertyMap");
    define_scalar_property_map<E, int64_t  >(sub, "E_int64_PropertyMap");
    define_scalar_property_map<H, int64_t  >(sub, "H_int64_PropertyMap");

    define_scalar_property_map<V, uint32_t >(sub, "V_uint32_PropertyMap");
    define_scalar_property_map<F, uint32_t >(sub, "F_uint32_PropertyMap");
    define_scalar_property_map<E, uint32_t >(sub, "E_uint32_PropertyMap");
    define_scalar_property_map<H, uint32_t >(sub, "H_uint32_PropertyMap");

    define_scalar_property_map<V, double   >(sub, "V_double_PropertyMap");
    define_scalar_property_map<F, double   >(sub, "F_double_PropertyMap");
    define_scalar_property_map<E, double   >(sub, "E_double_PropertyMap");
    define_scalar_property_map<H, double   >(sub, "H_double_PropertyMap");

    define_array_property_map<V, Point2, 2, double >(sub, "V_Point2_PropertyMap");
    define_array_property_map<F, Point2, 2, double >(sub, "F_Point2_PropertyMap");
    define_array_property_map<E, Point2, 2, double >(sub, "E_Point2_PropertyMap");
    define_array_property_map<H, Point2, 2, double >(sub, "H_Point2_PropertyMap");

    define_array_property_map<V, Point3, 3, double >(sub, "V_Point3_PropertyMap");
    define_array_property_map<F, Point3, 3, double >(sub, "F_Point3_PropertyMap");
    define_array_property_map<E, Point3, 3, double >(sub, "E_Point3_PropertyMap");
    define_array_property_map<H, Point3, 3, double >(sub, "H_Point3_PropertyMap");

    define_array_property_map<V, Vector2, 2, double >(sub, "V_Vector2_PropertyMap");
    define_array_property_map<F, Vector2, 2, double >(sub, "F_Vector2_PropertyMap");
    define_array_property_map<E, Vector2, 2, double >(sub, "E_Vector2_PropertyMap");
    define_array_property_map<H, Vector2, 2, double >(sub, "H_Vector2_PropertyMap");

    define_array_property_map<V, Vector3, 3, double >(sub, "V_Vector3_PropertyMap");
    define_array_property_map<F, Vector3, 3, double >(sub, "F_Vector3_PropertyMap");
    define_array_property_map<E, Vector3, 3, double >(sub, "E_Vector3_PropertyMap");
    define_array_property_map<H, Vector3, 3, double >(sub, "H_Vector3_PropertyMap");


    py::class_<PrincipalCurvDir>(sub, "PrincipalCurvaturesAndDirections")
        .def(py::init<double, double, Vector3, Vector3>())
        .def(py::init<>())  // default (0, 0, Vec3(0, 0, 0), Vec3(0, 0, 0))
        .def_property_readonly("min_curvature", [](const PrincipalCurvDir& p) {return p.min_curvature;})
        .def_property_readonly("max_curvature", [](const PrincipalCurvDir& p) {return p.max_curvature;})
        .def_property_readonly("min_direction", [](const PrincipalCurvDir& p) {return p.min_direction;})
        .def_property_readonly("max_direction", [](const PrincipalCurvDir& p) {return p.max_direction;})
    ;
    py::class_<ExpandedPrincipalCurvaturesAndDirections>(sub, "ExpandedPrincipalCurvaturesAndDirections")
        .def_readonly("min_curvature", &ExpandedPrincipalCurvaturesAndDirections::min_curvature)
        .def_readonly("max_curvature", &ExpandedPrincipalCurvaturesAndDirections::max_curvature)
        .def_readonly("min_direction", &ExpandedPrincipalCurvaturesAndDirections::min_direction)
        .def_readonly("max_direction", &ExpandedPrincipalCurvaturesAndDirections::max_direction)
    ;

    define_princ_curv_dir_property_map<V>(sub, "V_PrincipalCurvaturesAndDirections_PropertyMap");
}