#pragma once
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

// This is necessary to build on appveyor for some reason
#define CGAL_EIGEN3_ENABLED 1

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>


namespace py = pybind11;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

typedef Kernel::Point_2                 Point2;
typedef Kernel::Point_3                 Point3;
typedef Kernel::Vector_2                Vector2;
typedef Kernel::Vector_3                Vector3;

typedef CGAL::Surface_mesh<Point3>      Mesh3;
typedef Mesh3::Vertex_index             V;
typedef Mesh3::Face_index               F;
typedef Mesh3::Halfedge_index           H;
typedef Mesh3::Edge_index               E;


template<typename T>
struct Indices {
    using size_type = uint32_t;

    py::array_t<size_type> indices;

    Indices() {}
    Indices(py::array_t<size_type> indices) : indices(indices) {}

    // Convert a vector of e.g. vertex_descriptors into uint32s
    static Indices from_vector(std::vector<T> idxs) {
        size_t n = idxs.size();
        py::array_t<size_type> indices({py::ssize_t(n)});
        auto r = indices.mutable_unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            r(i) = size_type(idxs[i]);
        }
        return Indices(indices);
    }

    // Convert ints back into descriptors
    // TODO this can just self.map_to_vector
//    std::vector<T> to_vector() {
//        std::vector<T> out;
//        size_t n = indices.size();
//        auto r = indices.unchecked<1>();
//        for (size_t i = 0; i < n; ++i) {
//            out.emplace_back(T(r(i)));
//        }
//        return out;
//    }

    template<typename U>
    std::vector<U> map_to_vector(const std::function<U (T)> fn) const {
        size_t n = indices.size();
        std::vector<U> out;
        out.reserve(n);
        auto ridxs = indices.unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            T idx = T(ridxs(i));
            out.emplace_back(fn(idx));
        }
        return out;
    }

    template<typename U>
    py::array_t<U> map_to_array_of_scalars(const std::function<U (T)> fn) const {
        size_t n = indices.size();
        py::array_t<U> out({py::ssize_t(n)});
        //py::array_t<U> out(indices.shape());   // How to do this?
        auto rout = out.mutable_unchecked<1>();
        auto ridxs = indices.unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            T idx = T(ridxs(i));
            rout(i) = fn(idx);
        }

    }

//    template<size_t N, typename U>
//    py::array_t<double> map_to_array(const std::function<U (T)> fn) {
//        const size_t n = indices.size();
//        py::array_t<double> vals({n_idxs, N});
//        auto r = vals.template mutable_unchecked<2>();
//        for (auto i = 0; i < n_idxs; ++i) {
//            U val = fn(T());
//            for (auto j = 0; j < N; ++j) {
//                r(i, j) = CGAL::to_double(val[j]);
//            }
//        }
//        return vals;
//    }
};