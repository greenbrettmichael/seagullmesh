#pragma once
#include "seagullmesh.hpp"

std::vector<Point3> array_to_points_3(const py::array_t<double> &verts);
std::vector<Point2> array_to_points_2(const py::array_t<double> &verts);

py::array_t<double, py::array::c_style> points_to_array(const std::vector<Point3>& points);
py::array_t<double, py::array::c_style> points_to_array(const std::vector<Point2>& points);

// e.g. given vector<V> and function V -> Point3 return a py::array of shape (n_verts, 3)
template<size_t N, typename Idx, typename Val>
py::array_t<double> map_indices_to_vector(const std::vector<Idx>& idxs, const std::function<Val (Idx)> fn) {
    const size_t n_idxs = idxs.size();
    py::array_t<double> vals({n_idxs, N});
    auto r = vals.mutable_unchecked<2>();
    for (auto i = 0; i < n_idxs; ++i) {
        Val val = fn(idxs[i]);
        for (auto j = 0; j < N; ++j) {
            r(i, j) = CGAL::to_double(val[j]);
        }
    }
    return vals;
}

// e.g. given vector<V> and function V -> float return a (n_verts) array
template<typename Idx, typename Val>
py::array_t<Val> map_indices_to_scalar(const std::vector<Idx>& idxs, const std::function<Val (Idx)> fn) {
    const py::ssize_t n_idxs = idxs.size();
    py::array_t<Val> vals({n_idxs});
    auto r = vals.mutable_unchecked<1>();
    for (auto i = 0; i < n_idxs; ++i) {
        r(i) = fn(idxs[i]);
    }
    return vals;
}
