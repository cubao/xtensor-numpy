#include <pybind11/pybind11.h>

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

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}