#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
// #include "rdp.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


PYBIND11_MODULE(_core, m)
{
    m.doc() = R"pbdoc(
        pocketpy with numpy
        -------------------

        credits:
            -   https://github.com/pocketpy/pocketpy
            -   https://github.com/pocketpy/xtensor-numpy
    )pbdoc";

    // TODO, bind pocketpy repl, debugging, profiling, etc.
    /*
    namespace py = pybind11;
    using namespace pybind11::literals;

    using namespace rdp;
    py::class_<LineSegment>(m, "LineSegment") //
        .def(py::init<const Eigen::Vector3d, const Eigen::Vector3d>(), "A"_a,
             "B"_a)
        .def("distance", &LineSegment::distance, "P"_a)
        .def("distance2", &LineSegment::distance2, "P"_a)
        //
        ;

    auto rdp_doc = R"pbdoc(
        Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.

        Example:
        >>> from pybind11_rdp import rdp
        >>> rdp([[1, 1], [2, 2], [3, 3], [4, 4]])
        [[1, 1], [4, 4]]
    )pbdoc";
    m.def(
        "rdp",
        [](const Eigen::Ref<const RowVectors> &coords, double epsilon,
           bool recursive) -> RowVectors {
            return douglas_simplify(coords, epsilon, recursive);
        },
        rdp_doc, "coords"_a, //
        py::kw_only(), "epsilon"_a = 0.0, "recursive"_a = true);
    m.def(
        "rdp",
        [](const Eigen::Ref<const RowVectorsNx2> &coords, double epsilon,
           bool recursive) -> RowVectorsNx2 {
            RowVectors xyzs(coords.rows(), 3);
            xyzs.setZero();
            xyzs.leftCols(2) = coords;
            return douglas_simplify(xyzs, epsilon, recursive).leftCols(2);
        },
        rdp_doc, "coords"_a, //
        py::kw_only(), "epsilon"_a = 0.0, "recursive"_a = true);

    auto rdp_mask_doc = R"pbdoc(
        Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.
        return a mask.
    )pbdoc";
    m.def(
        "rdp_mask",
        [](const Eigen::Ref<const RowVectors> &coords, double epsilon,
           bool recursive) -> Eigen::VectorXi {
            return douglas_simplify_mask(coords, epsilon, recursive);
        },
        rdp_mask_doc, "coords"_a, //
        py::kw_only(), "epsilon"_a = 0.0, "recursive"_a = true);
    m.def(
        "rdp_mask",
        [](const Eigen::Ref<const RowVectorsNx2> &coords, double epsilon,
           bool recursive) -> Eigen::VectorXi {
            RowVectors xyzs(coords.rows(), 3);
            xyzs.setZero();
            xyzs.leftCols(2) = coords;
            return douglas_simplify_mask(xyzs, epsilon, recursive);
        },
        rdp_mask_doc, "coords"_a, //
        py::kw_only(), "epsilon"_a = 0.0, "recursive"_a = true);

    */

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}