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

        .def("__getitem__", [](const PMap& pmap, const Key& key) { return pmap[key]; })
        .def("__getitem__", [](const PMap& pmap, const Indices<Key>& indices) {
            return indices.map_to_array_of_scalars<Val>([&pmap](Key k) { return pmap[k]; });
        })
        .def("__setitem__", [](PMap& pmap, const Key& key, const Val val) {
            pmap[key] = val;
        })
        .def("__setitem__", [](PMap& pmap, const Indices<Key>& indices, const Val val) {
            // TODO need a custom iterator
            for (Key key : indices.to_vector()) {
                pmap[key] = val;
            }
        })
//        .def("__setitem__", [](PMap& pmap, const std::vector<Key>& keys, const std::vector<Val>& vals) {
//            size_t nk = keys.size();
//            size_t nv = vals.size();
//            if (nk != nv) {
//                throw std::runtime_error("Key and value array sizes do not match");
//            }
//            for (size_t i = 0; i < nk; i++) {
//                pmap[keys[i]] = vals[i];
//            }
//        })
    ;
}

//
//// For Point2/3 and Vector2/3
//template <unsigned int N, typename Key, typename Val>
//void define_array_property_map(py::module &m, std::string name) {
//    using PMap = typename Mesh3::Property_map<Key, Val>;
//
//    define_property_map<Key, Val>(m, name, false)
//        .def("get_array", [](const PMap& pmap, const std::vector<Key>& keys) {
//            const size_t n_keys = keys.size();
//            py::array_t<double, py::array::c_style> vals({n_keys, size_t(N)});
//            auto r = vals.mutable_unchecked<2>();
//
//            for (auto i = 0; i < n_keys; i++) {
//                auto val = pmap[keys[i]];
//                for (auto j = 0; j < N; j++) {
//                    r(i, j) = val[j];
//                }
//            }
//            return vals;
//        })
//        .def("get_objects", [](const PMap& pmap, const Key& key) {
//            return pmap[key];
//        })
//        .def("get_objects", [](const PMap& pmap, const std::vector<Key>& keys) {
//            std::vector<Val> vals;
//            vals.reserve(keys.size());
//            for (Key k : keys) {
//                vals.emplace_back(pmap[k]);
//            }
//            return vals;
//        })
//        .def("set_objects", [](PMap& pmap, const Key& key, const Val val) {
//            pmap[key] = val;
//        })
//        .def("set_objects", [](PMap& pmap, const std::vector<Key>& keys, const std::vector<Val>& vals) {
//            for (size_t i = 0; i < keys.size(); ++i) {
//                pmap[keys[i]] = vals[i];
//            }
//        })
//        .def("set_array", [](PMap& pmap, const std::vector<Key>& keys, const py::array_t<double>& vals) {
//            const size_t n_keys = keys.size();
//            auto r = vals.unchecked<2>();
//            if (n_keys != r.shape(0)) {
//                throw std::runtime_error("Key and value array sizes do not match");
//            }
//            if (N != r.shape(1)) {
//                throw std::runtime_error("Array has wrong number of columns");
//            }
//            for (auto i = 0; i < n_keys; i++) {
//                if constexpr ( N == 3 ) {
//                    pmap[keys[i]] = Val(r(i, 0), r(i, 1), r(i, 2));
//                } else {
//                    pmap[keys[i]] = Val(r(i, 0), r(i, 1));
//                }
//            }
//        })
//    ;
//}
//
//
//struct ExpandedPrincipalCurvaturesAndDirections {
//    py::array_t<double> min_curvature;
//    py::array_t<double> max_curvature;
//    py::array_t<double> min_direction;
//    py::array_t<double> max_direction;
//};
//
//
//template <typename Key>
//void define_princ_curv_dir_property_map(py::module &m, std::string name) {
//    using PMap = typename Mesh3::Property_map<Key, PrincipalCurvDir>;
//
//    define_property_map<Key, PrincipalCurvDir>(m, name, false)
//        .def("get_array", [](const PMap& pmap, const std::vector<Key>& keys) {
//            const size_t nk = keys.size();
//            std::vector<PrincipalCurvDir> vals;
//            vals.reserve(nk);
//            for (auto i = 0; i < nk; i++) {
//                vals.emplace_back(pmap[keys[i]]);
//            }
//            return vals;
//        })
//        .def("get_expanded", [](const PMap& pmap, const std::vector<Key>& keys) {
//            const size_t nk = keys.size();
//            py::array_t<double, py::array::c_style> min_curvature({int(nk)});
//            py::array_t<double, py::array::c_style> max_curvature({int(nk)});
//            py::array_t<double, py::array::c_style> min_direction({nk, size_t(3)});
//            py::array_t<double, py::array::c_style> max_direction({nk, size_t(3)});
//
//            auto r_min_curvature = min_curvature.mutable_unchecked<1>();
//            auto r_max_curvature = max_curvature.mutable_unchecked<1>();
//            auto r_min_direction = min_direction.mutable_unchecked<2>();
//            auto r_max_direction = max_direction.mutable_unchecked<2>();
//
//            for (auto i = 0; i < nk; i++) {
//                PrincipalCurvDir x = pmap[keys[i]];
//                r_min_curvature(i) = x.min_curvature;
//                r_max_curvature(i) = x.max_curvature;
//                for (auto j = 0; j < 3; j++) {
//                    r_min_direction(i, j) = x.min_direction[j];
//                    r_max_direction(i, j) = x.max_direction[j];
//                }
//            }
//            return ExpandedPrincipalCurvaturesAndDirections{min_curvature, max_curvature, min_direction, max_direction};
//        })
//    ;
//}
//

void init_properties(py::module &m) {
    py::module sub = m.def_submodule("properties");
    define_property_map<V, bool     >(sub, "V_bool_PropertyMap");
//    define_property_map<F, bool     >(sub, "F_bool_PropertyMap");
//    define_property_map<E, bool     >(sub, "E_bool_PropertyMap");
//    define_property_map<H, bool     >(sub, "H_bool_PropertyMap");
//
//    define_property_map<V, int32_t  >(sub, "V_int32_PropertyMap");
//    define_property_map<F, int32_t  >(sub, "F_int32_PropertyMap");
//    define_property_map<E, int32_t  >(sub, "E_int32_PropertyMap");
//    define_property_map<H, int32_t  >(sub, "H_int32_PropertyMap");
//
//    define_property_map<V, int64_t  >(sub, "V_int64_PropertyMap");
//    define_property_map<F, int64_t  >(sub, "F_int64_PropertyMap");
//    define_property_map<E, int64_t  >(sub, "E_int64_PropertyMap");
//    define_property_map<H, int64_t  >(sub, "H_int64_PropertyMap");
//
//    define_property_map<V, uint32_t >(sub, "V_uint32_PropertyMap");
//    define_property_map<F, uint32_t >(sub, "F_uint32_PropertyMap");
//    define_property_map<E, uint32_t >(sub, "E_uint32_PropertyMap");
//    define_property_map<H, uint32_t >(sub, "H_uint32_PropertyMap");
//
//    define_property_map<V, double   >(sub, "V_double_PropertyMap");
//    define_property_map<F, double   >(sub, "F_double_PropertyMap");
//    define_property_map<E, double   >(sub, "E_double_PropertyMap");
//    define_property_map<H, double   >(sub, "H_double_PropertyMap");
//
//    define_array_property_map<2, V, Point2 >(sub, "V_Point2_PropertyMap");
//    define_array_property_map<2, F, Point2 >(sub, "F_Point2_PropertyMap");
//    define_array_property_map<2, E, Point2 >(sub, "E_Point2_PropertyMap");
//    define_array_property_map<2, H, Point2 >(sub, "H_Point2_PropertyMap");
//
//    define_array_property_map<3, V, Point3 >(sub, "V_Point3_PropertyMap");
//    define_array_property_map<3, F, Point3 >(sub, "F_Point3_PropertyMap");
//    define_array_property_map<3, E, Point3 >(sub, "E_Point3_PropertyMap");
//    define_array_property_map<3, H, Point3 >(sub, "H_Point3_PropertyMap");
//
//    define_array_property_map<2, V, Vector2>(sub, "V_Vector2_PropertyMap");
//    define_array_property_map<2, F, Vector2>(sub, "F_Vector2_PropertyMap");
//    define_array_property_map<2, E, Vector2>(sub, "E_Vector2_PropertyMap");
//    define_array_property_map<2, H, Vector2>(sub, "H_Vector2_PropertyMap");
//
//    define_array_property_map<3, V, Vector3>(sub, "V_Vector3_PropertyMap");
//    define_array_property_map<3, F, Vector3>(sub, "F_Vector3_PropertyMap");
//    define_array_property_map<3, E, Vector3>(sub, "E_Vector3_PropertyMap");
//    define_array_property_map<3, H, Vector3>(sub, "H_Vector3_PropertyMap");
//
//
//    py::class_<PrincipalCurvDir>(sub, "PrincipalCurvaturesAndDirections")
//        .def(py::init<double, double, Vector3, Vector3>())
//        .def(py::init<>())  // default (0, 0, Vec3(0, 0, 0), Vec3(0, 0, 0))
//        .def_property_readonly("min_curvature", [](const PrincipalCurvDir& p) {return p.min_curvature;})
//        .def_property_readonly("max_curvature", [](const PrincipalCurvDir& p) {return p.max_curvature;})
//        .def_property_readonly("min_direction", [](const PrincipalCurvDir& p) {return p.min_direction;})
//        .def_property_readonly("max_direction", [](const PrincipalCurvDir& p) {return p.max_direction;})
//    ;
//    py::class_<ExpandedPrincipalCurvaturesAndDirections>(sub, "ExpandedPrincipalCurvaturesAndDirections")
//        .def_readonly("min_curvature", &ExpandedPrincipalCurvaturesAndDirections::min_curvature)
//        .def_readonly("max_curvature", &ExpandedPrincipalCurvaturesAndDirections::max_curvature)
//        .def_readonly("min_direction", &ExpandedPrincipalCurvaturesAndDirections::min_direction)
//        .def_readonly("max_direction", &ExpandedPrincipalCurvaturesAndDirections::max_direction)
//    ;
//
//    define_princ_curv_dir_property_map<V>(sub, "V_PrincipalCurvaturesAndDirections_PropertyMap");

}