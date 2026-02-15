#include <ndarray_binding.hpp>
#include <pkbind/eigen.h>
#include "rdp.hpp"


class Random {
public:
    static py::object rand() { return py::float_(pkpy::numpy::random::rand<float64>()); }

    static ndarray_base* rand_shape(py::args args) {
        std::vector<int> shape;
        for(auto item: args)
            shape.push_back(py::cast<int>(item));
        return new ndarray<float64>(pkpy::numpy::random::rand<float64>(shape));
    }

    static py::object randn() { return py::float_(pkpy::numpy::random::randn<float64>()); }

    static ndarray_base* randn_shape(py::args args) {
        std::vector<int> shape;
        for(auto item: args)
            shape.push_back(py::cast<int>(item));
        return new ndarray<float64>(pkpy::numpy::random::randn<float64>(shape));
    }

    static py::object randint(int low, int high) { return py::int_(pkpy::numpy::random::randint<int>(low, high)); }

    static ndarray_base* randint_shape(int_ low, int_ high, const std::vector<int>& shape) {
        return new ndarray<int_>(pkpy::numpy::random::randint<int_>(low, high, shape));
    }

    static ndarray_base* uniform(float64 low, float64 high, const std::vector<int>& shape) {
        return new ndarray<float64>(pkpy::numpy::random::uniform<float64>(low, high, shape));
    }
};

// Declare ndarray types
using ndarray_bool = ndarray<bool_>;
using ndarray_int8 = ndarray<int8>;
using ndarray_int16 = ndarray<int16>;
using ndarray_int32 = ndarray<int32>;
using ndarray_int = ndarray<int_>;
using ndarray_int = ndarray<int64>;
using ndarray_float32 = ndarray<float32>;
using ndarray_float = ndarray<float64>;
using ndarray_float = ndarray<float_>;

// Define template for creating n-dimensional vectors
template <typename T, std::size_t N>
struct nvector_impl {
    using type = std::vector<typename nvector_impl<T, N - 1>::type>;
};
template <typename T>
struct nvector_impl<T, 0> {
    using type = T;
};
template <typename T, std::size_t N>
using nvector = typename nvector_impl<T, N>::type;

// Transform nvector<U, N> to nvector<T, N>
template <typename U, typename T, std::size_t N>
nvector<T, N> transform(const nvector<U, N>& values) {
    nvector<T, N> result;
    if constexpr(N != 0) {
        for (const auto& value : values) {
            result.push_back(transform<U, T, N - 1>(value));
        }
    } else {
        result = static_cast<T>(values);
    }
    return result;
}

void register_array_int(py::module_& m) {
    m.def("array", [](int_ value, const std::string& dtype) {
        if (dtype == "bool") {
            return std::unique_ptr<ndarray_base>(new ndarray_bool(value));
        } else if (dtype == "int8") {
            return std::unique_ptr<ndarray_base>(new ndarray_int8(value));
        } else if (dtype == "int16") {
            return std::unique_ptr<ndarray_base>(new ndarray_int16(value));
        } else if (dtype == "int32") {
            return std::unique_ptr<ndarray_base>(new ndarray_int32(value));
        } else if (dtype == "float32") {
            return std::unique_ptr<ndarray_base>(new ndarray_float32(value));
        } else if (dtype == "float64") {
            return std::unique_ptr<ndarray_base>(new ndarray_float(value));
        }
        return std::unique_ptr<ndarray_base>(new ndarray_int(value));
    }, py::arg("value"), py::arg("dtype") = "int64");
}

template<std::size_t N>
void register_array_int(py::module_& m) {
    m.def("array", [](const nvector<int_, N>& values, const std::string& dtype) {
        if (dtype == "bool") {
            return std::unique_ptr<ndarray_base>(new ndarray<bool_>(transform<int_, bool_, N>(values)));
        } else if (dtype == "int8") {
            return std::unique_ptr<ndarray_base>(new ndarray<int8>(transform<int_, int8, N>(values)));
        } else if (dtype == "int16") {
            return std::unique_ptr<ndarray_base>(new ndarray<int16>(transform<int_, int16, N>(values)));
        } else if (dtype == "int32") {
            return std::unique_ptr<ndarray_base>(new ndarray<int32>(transform<int_, int32, N>(values)));
        } else if (dtype == "float32") {
            return std::unique_ptr<ndarray_base>(new ndarray<float32>(transform<int_, float32, N>(values)));
        } else if (dtype == "float64") {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(transform<int_, float64, N>(values)));
        }
        return std::unique_ptr<ndarray_base>(new ndarray<int_>(values));
    }, py::arg("values"), py::arg("dtype") = "int64");
}

void register_array_float(py::module_& m) {
    m.def("array", [](float64 value, const std::string& dtype) {
        if (dtype == "bool") {
            return std::unique_ptr<ndarray_base>(new ndarray_bool(value));
        } else if (dtype == "int8") {
            return std::unique_ptr<ndarray_base>(new ndarray_int8(value));
        } else if (dtype == "int16") {
            return std::unique_ptr<ndarray_base>(new ndarray_int16(value));
        } else if (dtype == "int32") {
            return std::unique_ptr<ndarray_base>(new ndarray_int32(value));
        } else if (dtype == "int64") {
            return std::unique_ptr<ndarray_base>(new ndarray_int(value));
        } else if (dtype == "float32") {
            return std::unique_ptr<ndarray_base>(new ndarray_float32(value));
        }
        return std::unique_ptr<ndarray_base>(new ndarray_float(value));
    }, py::arg("value"), py::arg("dtype") = "float64");
}

template<std::size_t N>
void register_array_float(py::module_& m) {
    m.def("array", [](const nvector<float64, N>& values, const std::string& dtype) {
        if (dtype == "bool") {
            return std::unique_ptr<ndarray_base>(new ndarray<bool_>(transform<float64, bool_, N>(values)));
        } else if (dtype == "int8") {
            return std::unique_ptr<ndarray_base>(new ndarray<int8>(transform<float64, int8, N>(values)));
        } else if (dtype == "int16") {
            return std::unique_ptr<ndarray_base>(new ndarray<int16>(transform<float64, int16, N>(values)));
        } else if (dtype == "int32") {
            return std::unique_ptr<ndarray_base>(new ndarray<int32>(transform<float64, int32, N>(values)));
        } else if (dtype == "int64") {
            return std::unique_ptr<ndarray_base>(new ndarray<int_>(transform<float64, int_, N>(values)));
        } else if (dtype == "float32") {
            return std::unique_ptr<ndarray_base>(new ndarray<float32>(transform<float64, float32, N>(values)));
        }
        return std::unique_ptr<ndarray_base>(new ndarray<float64>(values));
    }, py::arg("values"), py::arg("dtype") = "float64");
}

// Register array creation functions.
void array_creation_registry(py::module_& m) {
    register_array_int(m);
    register_array_int<1>(m);
    register_array_int<2>(m);
    register_array_int<3>(m);
    register_array_int<4>(m);
    register_array_int<5>(m);

    register_array_float(m);
    register_array_float<1>(m);
    register_array_float<2>(m);
    register_array_float<3>(m);
    register_array_float<4>(m);
    register_array_float<5>(m);
}

PYBIND11_MODULE(numpy, m) {
    m.doc() = "Python bindings for pkpy::numpy::ndarray using pybind11";

    m.attr("bool_") = "bool";
    m.attr("int8") = "int8";
    m.attr("int16") = "int16";
    m.attr("int32") = "int32";
    m.attr("int64") = "int64";
    m.attr("int_") = "int64";
    m.attr("float32") = "float32";
    m.attr("float64") = "float64";
    m.attr("float_") = "float64";

    py::class_<ndarray_base>(m, "ndarray")
        .def_property_readonly("ndim", &ndarray_base::ndim)
        .def_property_readonly("size", &ndarray_base::size)
        .def_property_readonly("dtype", &ndarray_base::dtype)
        .def_property_readonly("shape", &ndarray_base::shape)
        .def("all", &ndarray_base::all)
        .def("any", &ndarray_base::any)
        .def("sum", &ndarray_base::sum)
        .def("sum", &ndarray_base::sum_axis)
        .def("sum", &ndarray_base::sum_axes)
        .def("prod", &ndarray_base::prod)
        .def("prod", &ndarray_base::prod_axis)
        .def("prod", &ndarray_base::prod_axes)
        .def("min", &ndarray_base::min)
        .def("min", &ndarray_base::min_axis)
        .def("min", &ndarray_base::min_axes)
        .def("max", &ndarray_base::max)
        .def("max", &ndarray_base::max_axis)
        .def("max", &ndarray_base::max_axes)
        .def("mean", &ndarray_base::mean)
        .def("mean", &ndarray_base::mean_axis)
        .def("mean", &ndarray_base::mean_axes)
        .def("std", &ndarray_base::std)
        .def("std", &ndarray_base::std_axis)
        .def("std", &ndarray_base::std_axes)
        .def("var", &ndarray_base::var)
        .def("var", &ndarray_base::var_axis)
        .def("var", &ndarray_base::var_axes)
        .def("argmin", &ndarray_base::argmin)
        .def("argmin", &ndarray_base::argmin_axis)
        .def("argmax", &ndarray_base::argmax)
        .def("argmax", &ndarray_base::argmax_axis)
        .def("argsort", &ndarray_base::argsort)
        .def("argsort", &ndarray_base::argsort_axis)
        .def("sort", &ndarray_base::sort)
        .def("sort", &ndarray_base::sort_axis)
        .def("reshape", &ndarray_base::reshape)
        .def("resize", &ndarray_base::resize)
        .def("squeeze", &ndarray_base::squeeze)
        .def("squeeze", &ndarray_base::squeeze_axis)
        .def("transpose", &ndarray_base::transpose)
        .def("transpose", &ndarray_base::transpose_tuple)
        .def("transpose", &ndarray_base::transpose_args)
        .def("repeat", &ndarray_base::repeat, py::arg("repeats"), py::arg("axis") = INT_MAX)
        .def("repeat", &ndarray_base::repeat_axis)
        .def("round", &ndarray_base::round)
        .def("flatten", &ndarray_base::flatten)
        .def("copy", &ndarray_base::copy)
        .def("astype", &ndarray_base::astype)
        .def("tolist", &ndarray_base::tolist)
        .def("__eq__", &ndarray_base::eq)
        .def("__ne__", &ndarray_base::ne)
        .def("__add__", &ndarray_base::add)
        .def("__add__", &ndarray_base::add_bool)
        .def("__add__", &ndarray_base::add_int)
        .def("__add__", &ndarray_base::add_float)
        .def("__radd__", &ndarray_base::add_bool)
        .def("__radd__", &ndarray_base::add_int)
        .def("__radd__", &ndarray_base::add_float)
        .def("__sub__", &ndarray_base::sub)
        .def("__sub__", &ndarray_base::sub_int)
        .def("__sub__", &ndarray_base::sub_float)
        .def("__rsub__", &ndarray_base::rsub_int)
        .def("__rsub__", &ndarray_base::rsub_float)
        .def("__mul__", &ndarray_base::mul)
        .def("__mul__", &ndarray_base::mul_bool)
        .def("__mul__", &ndarray_base::mul_int)
        .def("__mul__", &ndarray_base::mul_float)
        .def("__rmul__", &ndarray_base::mul_bool)
        .def("__rmul__", &ndarray_base::mul_int)
        .def("__rmul__", &ndarray_base::mul_float)
        .def("__truediv__", &ndarray_base::div)
        .def("__truediv__", &ndarray_base::div_bool)
        .def("__truediv__", &ndarray_base::div_int)
        .def("__truediv__", &ndarray_base::div_float)
        .def("__rtruediv__", &ndarray_base::rdiv_bool)
        .def("__rtruediv__", &ndarray_base::rdiv_int)
        .def("__rtruediv__", &ndarray_base::rdiv_float)
        .def("__matmul__", &ndarray_base::matmul)
        .def("__pow__", &ndarray_base::pow)
        .def("__pow__", &ndarray_base::pow_int)
        .def("__pow__", &ndarray_base::pow_float)
        .def("__rpow__", &ndarray_base::rpow_int)
        .def("__rpow__", &ndarray_base::rpow_float)
        .def("__len__", &ndarray_base::len)
        .def("__and__", &ndarray_base::and_array)
        .def("__and__", &ndarray_base::and_bool)
        .def("__and__", &ndarray_base::and_int)
        .def("__rand__", &ndarray_base::and_bool)
        .def("__rand__", &ndarray_base::and_int)
        .def("__or__", &ndarray_base::or_array)
        .def("__or__", &ndarray_base::or_bool)
        .def("__or__", &ndarray_base::or_int)
        .def("__ror__", &ndarray_base::or_bool)
        .def("__ror__", &ndarray_base::or_int)
        .def("__xor__", &ndarray_base::xor_array)
        .def("__xor__", &ndarray_base::xor_bool)
        .def("__xor__", &ndarray_base::xor_int)
        .def("__rxor__", &ndarray_base::xor_bool)
        .def("__rxor__", &ndarray_base::xor_int)
        .def("__invert__", &ndarray_base::invert)
        .def("__getitem__", &ndarray_base::get_item_int)
        .def("__getitem__", &ndarray_base::get_item_tuple)
        .def("__getitem__", &ndarray_base::get_item_vector)
        .def("__getitem__", &ndarray_base::get_item_slice)
        .def("__setitem__", &ndarray_base::set_item_int)
        .def("__setitem__", &ndarray_base::set_item_index_int)
        .def("__setitem__", &ndarray_base::set_item_index_int_2d)
        .def("__setitem__", &ndarray_base::set_item_index_int_3d)
        .def("__setitem__", &ndarray_base::set_item_index_int_4d)
        .def("__setitem__", &ndarray_base::set_item_float)
        .def("__setitem__", &ndarray_base::set_item_index_float)
        .def("__setitem__", &ndarray_base::set_item_index_float_2d)
        .def("__setitem__", &ndarray_base::set_item_index_float_3d)
        .def("__setitem__", &ndarray_base::set_item_index_float_4d)
        .def("__setitem__", &ndarray_base::set_item_tuple_int1)
        .def("__setitem__", &ndarray_base::set_item_tuple_int2)
        .def("__setitem__", &ndarray_base::set_item_tuple_int3)
        .def("__setitem__", &ndarray_base::set_item_tuple_int4)
        .def("__setitem__", &ndarray_base::set_item_tuple_int5)
        .def("__setitem__", &ndarray_base::set_item_tuple_float1)
        .def("__setitem__", &ndarray_base::set_item_tuple_float2)
        .def("__setitem__", &ndarray_base::set_item_tuple_float3)
        .def("__setitem__", &ndarray_base::set_item_tuple_float4)
        .def("__setitem__", &ndarray_base::set_item_tuple_float5)
        .def("__setitem__", &ndarray_base::set_item_vector_int1)
        .def("__setitem__", &ndarray_base::set_item_vector_int2)
        .def("__setitem__", &ndarray_base::set_item_vector_int3)
        .def("__setitem__", &ndarray_base::set_item_vector_int4)
        .def("__setitem__", &ndarray_base::set_item_vector_int5)
        .def("__setitem__", &ndarray_base::set_item_vector_float1)
        .def("__setitem__", &ndarray_base::set_item_vector_float2)
        .def("__setitem__", &ndarray_base::set_item_vector_float3)
        .def("__setitem__", &ndarray_base::set_item_vector_float4)
        .def("__setitem__", &ndarray_base::set_item_vector_float5)
        .def("__setitem__", &ndarray_base::set_item_slice_int1)
        .def("__setitem__", &ndarray_base::set_item_slice_int2)
        .def("__setitem__", &ndarray_base::set_item_slice_int3)
        .def("__setitem__", &ndarray_base::set_item_slice_int4)
        .def("__setitem__", &ndarray_base::set_item_slice_int5)
        .def("__setitem__", &ndarray_base::set_item_slice_float1)
        .def("__setitem__", &ndarray_base::set_item_slice_float2)
        .def("__setitem__", &ndarray_base::set_item_slice_float3)
        .def("__setitem__", &ndarray_base::set_item_slice_float4)
        .def("__setitem__", &ndarray_base::set_item_slice_float5)

        .def("__str__",
             [](const ndarray_base& self) {
                 std::ostringstream os;
                 os << self.to_string();
                 return os.str();
             })
        .def("__repr__", [](const ndarray_base& self) {
            std::ostringstream os;
            os << "array(" << self.to_string() << ")";
            return os.str();
        });

    py::class_<ndarray<int8>, ndarray_base>(m, "ndarray_int8")
        .def(py::init<>())
        .def(py::init<int8>())
        .def(py::init<const std::vector<int8>&>())
        .def(py::init<const std::vector<std::vector<int8>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<int8>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<int8>>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<std::vector<int8>>>>>&>());

    py::class_<ndarray<int16>, ndarray_base>(m, "ndarray_int16")
        .def(py::init<>())
        .def(py::init<int16>())
        .def(py::init<const std::vector<int16>&>())
        .def(py::init<const std::vector<std::vector<int16>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<int16>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<int16>>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<std::vector<int16>>>>>&>());

    py::class_<ndarray<int32>, ndarray_base>(m, "ndarray_int32")
        .def(py::init<>())
        .def(py::init<int32>())
        .def(py::init<const std::vector<int32>&>())
        .def(py::init<const std::vector<std::vector<int32>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<int32>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<int32>>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<std::vector<int32>>>>>&>());

    py::class_<ndarray<bool_>, ndarray_base>(m, "ndarray_bool")
        .def(py::init<>())
        .def(py::init<bool_>())
        .def(py::init<const std::vector<bool_>&>())
        .def(py::init<const std::vector<std::vector<bool_>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<bool_>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<bool_>>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<std::vector<bool_>>>>>&>());

    py::class_<ndarray<int_>, ndarray_base>(m, "ndarray_int")
        .def(py::init<>())
        .def(py::init<int_>())
        .def(py::init<const std::vector<int_>&>())
        .def(py::init<const std::vector<std::vector<int_>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<int_>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<int_>>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<std::vector<int_>>>>>&>());

    py::class_<ndarray<float32>, ndarray_base>(m, "ndarray_float32")
        .def(py::init<>())
        .def(py::init<float32>())
        .def(py::init<const std::vector<float32>&>())
        .def(py::init<const std::vector<std::vector<float32>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<float32>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<float32>>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<std::vector<float32>>>>>&>());

    py::class_<ndarray<float64>, ndarray_base>(m, "ndarray_float")
        .def(py::init<>())
        .def(py::init<float64>())
        .def(py::init<const std::vector<float64>&>())
        .def(py::init<const std::vector<std::vector<float64>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<float64>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<float64>>>>&>())
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<std::vector<float64>>>>>&>());

    py::class_<Random>(m, "random")
        .def_static("rand", &Random::rand)
        .def_static("rand_shape", &Random::rand_shape)
        .def_static("randn", &Random::randn)
        .def_static("randn_shape", &Random::randn_shape)
        .def_static("randint", &Random::randint)
        .def_static("randint_shape", &Random::randint_shape)
        .def_static("uniform", &Random::uniform);

    array_creation_registry(m);

    m.def("array", [](bool_ value) {
    return std::unique_ptr<ndarray_base>(new ndarray_bool(value));
    });
    m.def("array", [](const std::vector<bool_>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_bool(values));
    });
    m.def("array", [](const std::vector<std::vector<bool_>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_bool(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<bool_>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_bool(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<bool_>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_bool(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<std::vector<bool_>>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_bool(values));
    });

    m.def("array", [](int8 value) {
    return std::unique_ptr<ndarray_base>(new ndarray_int8(value));
    });
    m.def("array", [](const std::vector<int8>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int8(values));
    });
    m.def("array", [](const std::vector<std::vector<int8>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int8(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<int8>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int8(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<int8>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int8(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<std::vector<int8>>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int8(values));
    });

    m.def("array", [](int16 value) {
    return std::unique_ptr<ndarray_base>(new ndarray_int16(value));
    });
    m.def("array", [](const std::vector<int16>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int16(values));
    });
    m.def("array", [](const std::vector<std::vector<int16>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int16(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<int16>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int16(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<int16>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int16(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<std::vector<int16>>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int16(values));
    });

    m.def("array", [](int32 value) {
    return std::unique_ptr<ndarray_base>(new ndarray_int32(value));
    });
    m.def("array", [](const std::vector<int32>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int32(values));
    });
    m.def("array", [](const std::vector<std::vector<int32>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int32(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<int32>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int32(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<int32>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int32(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<std::vector<int32>>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_int32(values));
    });

    m.def("array", [](float32 value) {
    return std::unique_ptr<ndarray_base>(new ndarray_float32(value));
    });
    m.def("array", [](const std::vector<float32>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_float32(values));
    });
    m.def("array", [](const std::vector<std::vector<float32>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_float32(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<float32>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_float32(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<float32>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_float32(values));
    });
    m.def("array", [](const std::vector<std::vector<std::vector<std::vector<std::vector<float32>>>>>& values) {
    return std::unique_ptr<ndarray_base>(new ndarray_float32(values));
    });

    // Array Creation Functions
    m.def("ones", [](const std::vector<int>& shape) {
        return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::ones<float64>(shape)));
    });
    m.def("zeros", [](const std::vector<int>& shape) {
        return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::zeros<float64>(shape)));
    });
    m.def("full", [](const std::vector<int>& shape, int_ value) {
        return std::unique_ptr<ndarray_base>(new ndarray_int(pkpy::numpy::full<int_>(shape, value)));
    });
    m.def("full", [](const std::vector<int>& shape, float64 value) {
        return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::full<float64>(shape, value)));
    });
    m.def("identity", [](int n) {
        return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::identity<float64>(n)));
    });
    m.def("arange", [](int_ stop) {
        return std::unique_ptr<ndarray_base>(new ndarray_int(pkpy::numpy::arange<int_>(0, stop)));
    });
    m.def("arange", [](int_ start, int_ stop) {
        return std::unique_ptr<ndarray_base>(new ndarray_int(pkpy::numpy::arange<int_>(start, stop)));
    });
    m.def("arange", [](int_ start, int_ stop, int_ step) {
        return std::unique_ptr<ndarray_base>(new ndarray_int(pkpy::numpy::arange<int_>(start, stop, step)));
    });
    m.def("arange", [](float64 stop) {
        return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::arange<float64>(0, stop)));
    });
    m.def("arange", [](float64 start, float64 stop) {
        return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::arange<float64>(start, stop)));
    });
    m.def("arange", [](float64 start, float64 stop, float64 step) {
        return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::arange<float64>(start, stop, step)));
    });
    m.def(
        "linspace",
        [](float64 start, float64 stop, int num, bool endpoint) {
            return std::unique_ptr<ndarray_base>(new ndarray_float(pkpy::numpy::linspace(start, stop, num, endpoint)));
        },
        py::arg("start"),
        py::arg("stop"),
        py::arg("num") = 50,
        py::arg("endpoint") = true);

    // Trigonometric Functions
    m.def("sin", [](const ndarray_base& arr) {
        if (auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::sin(p->data)));
        } else if (auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::sin(p->data)));
        } else if (auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::sin(p->data)));
        } else if (auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::sin(p->data)));
        } else if (auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::sin(p->data)));
        } else if (auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::sin(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("cos", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::cos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::cos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::cos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::cos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::cos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::cos(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("tan", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::tan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::tan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::tan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::tan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::tan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::tan(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("arcsin", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arcsin(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arcsin(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arcsin(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arcsin(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arcsin(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arcsin(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("arccos", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arccos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arccos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arccos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arccos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arccos(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arccos(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("arctan", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arctan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arctan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arctan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arctan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arctan(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::arctan(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });

    // Exponential and Logarithmic Functions
    m.def("exp", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::exp(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::exp(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::exp(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::exp(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::exp(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::exp(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("log", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("log2", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log2(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log2(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log2(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log2(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log2(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log2(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("log10", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log10(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log10(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log10(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log10(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log10(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::log10(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });

    // Miscellaneous Functions
    m.def("round", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int8>(pkpy::numpy::round(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int16>(pkpy::numpy::round(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int32>(pkpy::numpy::round(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int_>(pkpy::numpy::round(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float32>(pkpy::numpy::round(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::round(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("floor", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int8>(pkpy::numpy::floor(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int16>(pkpy::numpy::floor(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int32>(pkpy::numpy::floor(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int_>(pkpy::numpy::floor(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float32>(pkpy::numpy::floor(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::floor(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("ceil", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int8>(pkpy::numpy::ceil(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int16>(pkpy::numpy::ceil(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int32>(pkpy::numpy::ceil(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int_>(pkpy::numpy::ceil(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float32>(pkpy::numpy::ceil(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::ceil(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def("abs", [](const ndarray_base& arr) {
        if(auto p = dynamic_cast<const ndarray<int8>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int8>(pkpy::numpy::abs(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int16>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int16>(pkpy::numpy::abs(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int32>(pkpy::numpy::abs(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<int_>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<int_>(pkpy::numpy::abs(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float32>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float32>(pkpy::numpy::abs(p->data)));
        } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr)) {
            return std::unique_ptr<ndarray_base>(new ndarray<float64>(pkpy::numpy::abs(p->data)));
        }
        throw std::invalid_argument("Invalid dtype");
    });
    m.def(
        "concatenate",
        [](const ndarray_base& arr1, const ndarray_base& arr2, int axis) {
            if(auto p = dynamic_cast<const ndarray<int_>*>(&arr1)) {
                if(auto q = dynamic_cast<const ndarray<int_>*>(&arr2)) {
                    return std::unique_ptr<ndarray_base>(
                        new ndarray<int_>(pkpy::numpy::concatenate(p->data, q->data, axis)));
                } else if(auto q = dynamic_cast<const ndarray<float64>*>(&arr2)) {
                    return std::unique_ptr<ndarray_base>(
                        new ndarray<float64>(pkpy::numpy::concatenate(p->data, q->data, axis)));
                }
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr1)) {
                if(auto q = dynamic_cast<const ndarray<int_>*>(&arr2)) {
                    return std::unique_ptr<ndarray_base>(
                        new ndarray<float64>(pkpy::numpy::concatenate(p->data, q->data, axis)));
                } else if(auto q = dynamic_cast<const ndarray<float64>*>(&arr2)) {
                    return std::unique_ptr<ndarray_base>(
                        new ndarray<float64>(pkpy::numpy::concatenate(p->data, q->data, axis)));
                }
            }
            throw std::invalid_argument("Invalid dtype");
        },
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("axis") = 0);

    // Constants
    m.attr("pi") = pkpy::numpy::pi;
    m.attr("inf") = pkpy::numpy::inf;

    // Testing Functions
    m.def(
        "allclose",
        [](const ndarray_base& arr1, const ndarray_base& arr2, float64 rtol, float64 atol) {
            if(auto p = dynamic_cast<const ndarray<int_>*>(&arr1)) {
                if(auto q = dynamic_cast<const ndarray<int_>*>(&arr2)) {
                    return pkpy::numpy::allclose(p->data, q->data, rtol, atol);
                } else if(auto q = dynamic_cast<const ndarray<float64>*>(&arr2)) {
                    return pkpy::numpy::allclose(p->data, q->data, rtol, atol);
                }
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&arr1)) {
                if(auto q = dynamic_cast<const ndarray<int_>*>(&arr2)) {
                    return pkpy::numpy::allclose(p->data, q->data, rtol, atol);
                } else if(auto q = dynamic_cast<const ndarray<float64>*>(&arr2)) {
                    return pkpy::numpy::allclose(p->data, q->data, rtol, atol);
                }
            }
            throw std::invalid_argument("Invalid dtype");
        },
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("rtol") = 1e-5,
        py::arg("atol") = 1e-8);

    // --- RDP bindings (Eigen types auto-converted via pybind11/eigen.h type_caster) ---

    auto rdp_doc = R"pbdoc(
        Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.

        Example:
        >>> from pocket_numpy import rdp
        >>> rdp([[1, 1], [2, 2], [3, 3], [4, 4]])
        [[1, 1], [4, 4]]
    )pbdoc";

    // rdp: Nx3 (auto-converted from ndarray via type_caster)
    m.def("rdp", [](const Eigen::Ref<const rdp::RowVectors>& coords, double epsilon, bool recursive) -> rdp::RowVectors {
        return rdp::douglas_simplify(coords, epsilon, recursive);
    }, rdp_doc, py::arg("coords"), py::arg("epsilon") = 0.0, py::arg("recursive") = true);

    // rdp: Nx2 (type_caster rejects Nx3, picks this overload)
    m.def("rdp", [](const Eigen::Ref<const rdp::RowVectorsNx2>& coords, double epsilon, bool recursive) -> rdp::RowVectorsNx2 {
        rdp::RowVectors xyzs(coords.rows(), 3);
        xyzs.setZero();
        xyzs.leftCols(2) = coords;
        return rdp::douglas_simplify(xyzs, epsilon, recursive).leftCols(2);
    }, rdp_doc, py::arg("coords"), py::arg("epsilon") = 0.0, py::arg("recursive") = true);

    // rdp: from list<list<double>> (stl.h handles the conversion)
    m.def("rdp", [](std::vector<std::vector<double>> coords, double epsilon, bool recursive) -> std::unique_ptr<ndarray_base> {
        int rows = (int)coords.size();
        if (rows == 0) throw std::invalid_argument("rdp: empty coords");
        int cols = (int)coords[0].size();
        if (cols != 2 && cols != 3) throw std::invalid_argument("rdp: expected Nx2 or Nx3");
        rdp::RowVectors eigen_coords(rows, 3);
        eigen_coords.setZero();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                eigen_coords(i, j) = coords[i][j];
        auto result = rdp::douglas_simplify(eigen_coords, epsilon, recursive);
        int N = (int)result.rows();
        std::vector<std::vector<double>> data(N, std::vector<double>(cols));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = result(i, j);
        return std::unique_ptr<ndarray_base>(new ::ndarray<float64>(data));
    }, py::arg("coords"), py::arg("epsilon") = 0.0, py::arg("recursive") = true);

    auto rdp_mask_doc = R"pbdoc(
        Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.
        Returns a mask.
    )pbdoc";

    // rdp_mask: Nx3
    m.def("rdp_mask", [](const Eigen::Ref<const rdp::RowVectors>& coords, double epsilon, bool recursive) -> Eigen::VectorXi {
        return rdp::douglas_simplify_mask(coords, epsilon, recursive);
    }, rdp_mask_doc, py::arg("coords"), py::arg("epsilon") = 0.0, py::arg("recursive") = true);

    // rdp_mask: Nx2
    m.def("rdp_mask", [](const Eigen::Ref<const rdp::RowVectorsNx2>& coords, double epsilon, bool recursive) -> Eigen::VectorXi {
        rdp::RowVectors xyzs(coords.rows(), 3);
        xyzs.setZero();
        xyzs.leftCols(2) = coords;
        return rdp::douglas_simplify_mask(xyzs, epsilon, recursive);
    }, rdp_mask_doc, py::arg("coords"), py::arg("epsilon") = 0.0, py::arg("recursive") = true);

    // rdp_mask: from list<list<double>>
    m.def("rdp_mask", [](std::vector<std::vector<double>> coords, double epsilon, bool recursive) -> std::unique_ptr<ndarray_base> {
        int rows = (int)coords.size();
        if (rows == 0) throw std::invalid_argument("rdp: empty coords");
        int cols = (int)coords[0].size();
        if (cols != 2 && cols != 3) throw std::invalid_argument("rdp: expected Nx2 or Nx3");
        rdp::RowVectors eigen_coords(rows, 3);
        eigen_coords.setZero();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                eigen_coords(i, j) = coords[i][j];
        auto mask = rdp::douglas_simplify_mask(eigen_coords, epsilon, recursive);
        std::vector<int_> mask_data(mask.size());
        for (int i = 0; i < mask.size(); ++i) mask_data[i] = mask[i];
        return std::unique_ptr<ndarray_base>(new ::ndarray<int_>(mask_data));
    }, py::arg("coords"), py::arg("epsilon") = 0.0, py::arg("recursive") = true);

    // Create pocket_numpy module that re-exports rdp/rdp_mask from numpy
    {
        py_GlobalRef pocket_numpy_mod = py_newmodule("pocket_numpy");
        if (py_getattr(m.ptr(), py_name("rdp"))) {
            py_setattr(pocket_numpy_mod, py_name("rdp"), py_retval());
        }
        if (py_getattr(m.ptr(), py_name("rdp_mask"))) {
            py_setattr(pocket_numpy_mod, py_name("rdp_mask"), py_retval());
        }
    }
}
