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

    std::vector<T> to_vector() const {
        return map_to_vector<T>([](T idx) { return idx; } );
    }

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
        py::array_t<U> out({ py::ssize_t(n) });
        auto rout = out.mutable_unchecked<1>();
        auto ridxs = indices.unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            T idx = T( ridxs(i) );
            rout(i) = fn(idx);
        }
    }

    template <typename Vec, size_t Dim, typename U>
    py::array_t<U> map_to_array_of_vectors(const std::function<Vec (T)> fn) const {
        size_t n = indices.size();
        py::array_t<U> out( {n, Dim} );
        auto rout = out.mutable_unchecked<2>();
        auto ridxs = indices.unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            T idx = T( ridxs(i) );
            Vec vec = fn(idx);
            for (size_t j = 0; j < Dim; ++j) {
                rout(i, j) = vec[j];
            }
        }
        return out;
    }

    void map(const std::function<void (size_t, T)> fn) const {
        size_t n = indices.size();
        auto ridxs = indices.unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            T idx = T( ridxs(i) );
            fn(i, idx);
        }
    }
};