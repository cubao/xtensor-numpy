#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numpy.hpp>
#include <typeinfo>

namespace py = pybind11;

using bool_ = pkpy::bool_;
using int8 = pkpy::int8;
using int16 = pkpy::int16;
using int32 = pkpy::int32;
using int64 = pkpy::int64;
using int_ = pkpy::int_;
using float32 = pkpy::float32;
using float64 = pkpy::float64;
using float_ = pkpy::float_;

// Function to parse attributes
inline int parseAttr(const py::object& obj) {
    if(py::isinstance<py::none>(obj)) {
        return INT_MAX;
    } else if(py::isinstance<py::int_>(obj)) {
        return obj.cast<int>();
    } else {
        throw std::runtime_error("Unsupported type");
    }
};

class ndarray_base {
public:
    virtual ~ndarray_base() = default;

    virtual int ndim() const = 0;

    virtual int size() const = 0;

    virtual std::string dtype() const = 0;

    virtual py::tuple shape() const = 0;

    virtual bool all() const = 0;

    virtual bool any() const = 0;

    virtual py::object sum() const = 0;

    virtual py::object sum_axis(int axis) const = 0;

    virtual py::object sum_axes(py::tuple axes) const = 0;

    virtual py::object prod() const = 0;

    virtual py::object prod_axis(int axis) const = 0;

    virtual py::object prod_axes(py::tuple axes) const = 0;

    virtual py::object min() const = 0;

    virtual py::object min_axis(int axis) const = 0;

    virtual py::object min_axes(py::tuple axes) const = 0;

    virtual py::object max() const = 0;

    virtual py::object max_axis(int axis) const = 0;

    virtual py::object max_axes(py::tuple axes) const = 0;

    virtual py::object mean() const = 0;

    virtual py::object mean_axis(int axis) const = 0;

    virtual py::object mean_axes(py::tuple axes) const = 0;

    virtual py::object std() const = 0;

    virtual py::object std_axis(int axis) const = 0;

    virtual py::object std_axes(py::tuple axes) const = 0;

    virtual py::object var() const = 0;

    virtual py::object var_axis(int axis) const = 0;

    virtual py::object var_axes(py::tuple axes) const = 0;

    virtual py::object argmin() const = 0;

    virtual ndarray_base* argmin_axis(int axis) const = 0;

    virtual py::object argmax() const = 0;

    virtual ndarray_base* argmax_axis(int axis) const = 0;

    virtual ndarray_base* argsort() const = 0;

    virtual ndarray_base* argsort_axis(int axis) const = 0;

    virtual void sort() = 0;

    virtual void sort_axis(int axis) = 0;

    virtual ndarray_base* reshape(const std::vector<int>& shape) const = 0;

    virtual void resize(const std::vector<int>& shape) = 0;

    virtual ndarray_base* squeeze() const = 0;

    virtual ndarray_base* squeeze_axis(int axis) const = 0;

    virtual ndarray_base* transpose() const = 0;

    virtual ndarray_base* transpose_tuple(py::tuple permutations) const = 0;

    virtual ndarray_base* transpose_args(py::args args) const = 0;

    virtual ndarray_base* repeat(int repeats, int axis) const = 0;

    virtual ndarray_base* repeat_axis(const std::vector<size_t>& repeats, int axis) const = 0;

    virtual ndarray_base* round() const = 0;

    virtual ndarray_base* flatten() const = 0;

    virtual ndarray_base* copy() const = 0;

    virtual ndarray_base* astype(const std::string& dtype) const = 0;

    virtual py::list tolist() const = 0;

    virtual ndarray_base* eq(const ndarray_base& other) const = 0;

    virtual ndarray_base* ne(const ndarray_base& other) const = 0;

    virtual ndarray_base* add(const ndarray_base& other) const = 0;

    virtual ndarray_base* add_bool(bool_ other) const = 0;

    virtual ndarray_base* add_int(int_ other) const = 0;

    virtual ndarray_base* add_float(float64 other) const = 0;

    virtual ndarray_base* sub(const ndarray_base& other) const = 0;

    virtual ndarray_base* sub_int(int_ other) const = 0;

    virtual ndarray_base* sub_float(float64 other) const = 0;

    virtual ndarray_base* rsub_int(int_ other) const = 0;

    virtual ndarray_base* rsub_float(float64 other) const = 0;

    virtual ndarray_base* mul(const ndarray_base& other) const = 0;

    virtual ndarray_base* mul_bool(bool_ other) const = 0;

    virtual ndarray_base* mul_int(int_ other) const = 0;

    virtual ndarray_base* mul_float(float64 other) const = 0;

    virtual ndarray_base* div(const ndarray_base& other) const = 0;

    virtual ndarray_base* div_bool(bool_ other) const = 0;

    virtual ndarray_base* div_int(int_ other) const = 0;

    virtual ndarray_base* div_float(float64 other) const = 0;

    virtual ndarray_base* rdiv_bool(bool_ other) const = 0;

    virtual ndarray_base* rdiv_int(int_ other) const = 0;

    virtual ndarray_base* rdiv_float(float64 other) const = 0;

    virtual ndarray_base* matmul(const ndarray_base& other) const = 0;

    virtual ndarray_base* pow(const ndarray_base& other) const = 0;

    virtual ndarray_base* pow_int(int_ other) const = 0;

    virtual ndarray_base* pow_float(float64 other) const = 0;

    virtual ndarray_base* rpow_int(int_ other) const = 0;

    virtual ndarray_base* rpow_float(float64 other) const = 0;

    virtual ndarray_base* and_array(const ndarray_base& other) const = 0;

    virtual ndarray_base* and_bool(bool_ other) const = 0;

    virtual ndarray_base* and_int(int_ other) const = 0;

    virtual ndarray_base* or_array(const ndarray_base& other) const = 0;

    virtual ndarray_base* or_bool(bool_ other) const = 0;

    virtual ndarray_base* or_int(int_ other) const = 0;

    virtual ndarray_base* xor_array(const ndarray_base& other) const = 0;

    virtual ndarray_base* xor_bool(bool_ other) const = 0;

    virtual ndarray_base* xor_int(int_ other) const = 0;

    virtual ndarray_base* invert() const = 0;

    virtual py::object get_item_int(int index) const = 0;

    virtual py::object get_item_tuple(py::tuple indices) const = 0;

    virtual ndarray_base* get_item_vector(const std::vector<int>& indices) const = 0;

    virtual ndarray_base* get_item_slice(py::slice slice) const = 0;

    virtual void set_item_int(int index, int_ value) = 0;

    virtual void set_item_index_int(int index, const std::vector<int_>& value) = 0;

    virtual void set_item_index_int_2d(int index, const std::vector<std::vector<int_>>& value) = 0;

    virtual void set_item_index_int_3d(int index, const std::vector<std::vector<std::vector<int_>>>& value) = 0;

    virtual void set_item_index_int_4d(int index, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) = 0;

    virtual void set_item_float(int index, float64 value) = 0;

    virtual void set_item_index_float(int index, const std::vector<float64>& value) = 0;

    virtual void set_item_index_float_2d(int index, const std::vector<std::vector<float64>>& value) = 0;

    virtual void set_item_index_float_3d(int index, const std::vector<std::vector<std::vector<float64>>>& value) = 0;

    virtual void set_item_index_float_4d(int index, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) = 0;

    virtual void set_item_tuple_int1(py::tuple args, int_ value) = 0;

    virtual void set_item_tuple_int2(py::tuple args, const std::vector<int_>& value) = 0;

    virtual void set_item_tuple_int3(py::tuple args, const std::vector<std::vector<int_>>& value) = 0;

    virtual void set_item_tuple_int4(py::tuple args, const std::vector<std::vector<std::vector<int_>>>& value) = 0;

    virtual void set_item_tuple_int5(py::tuple args, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) = 0;

    virtual void set_item_tuple_float1(py::tuple args, float64 value) = 0;

    virtual void set_item_tuple_float2(py::tuple args, const std::vector<float64>& value) = 0;

    virtual void set_item_tuple_float3(py::tuple args, const std::vector<std::vector<float64>>& value) = 0;

    virtual void set_item_tuple_float4(py::tuple args, const std::vector<std::vector<std::vector<float64>>>& value) = 0;

    virtual void set_item_tuple_float5(py::tuple args, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) = 0;

    virtual void set_item_vector_int1(const std::vector<int>& indices, int_ value) = 0;

    virtual void set_item_vector_int2(const std::vector<int>& indices, const std::vector<int_>& value) = 0;

    virtual void set_item_vector_int3(const std::vector<int>& indices, const std::vector<std::vector<int_>>& value) = 0;

    virtual void set_item_vector_int4(const std::vector<int>& indices, const std::vector<std::vector<std::vector<int_>>>& value) = 0;

    virtual void set_item_vector_int5(const std::vector<int>& indices, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) = 0;

    virtual void set_item_vector_float1(const std::vector<int>& indices, float64 value) = 0;

    virtual void set_item_vector_float2(const std::vector<int>& indices, const std::vector<float64>& value) = 0;

    virtual void set_item_vector_float3(const std::vector<int>& indices, const std::vector<std::vector<float64>>& value) = 0;

    virtual void set_item_vector_float4(const std::vector<int>& indices, const std::vector<std::vector<std::vector<float64>>>& value) = 0;

    virtual void set_item_vector_float5(const std::vector<int>& indices, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) = 0;

    virtual void set_item_slice_int1(py::slice slice, int_ value) = 0;

    virtual void set_item_slice_int2(py::slice slice, const std::vector<int_>& value) = 0;

    virtual void set_item_slice_int3(py::slice slice, const std::vector<std::vector<int_>>& value) = 0;

    virtual void set_item_slice_int4(py::slice slice, const std::vector<std::vector<std::vector<int_>>>& value) = 0;

    virtual void set_item_slice_int5(py::slice slice, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) = 0;

    virtual void set_item_slice_float1(py::slice slice, float64 value) = 0;

    virtual void set_item_slice_float2(py::slice slice, const std::vector<float64>& value) = 0;

    virtual void set_item_slice_float3(py::slice slice, const std::vector<std::vector<float64>>& value) = 0;

    virtual void set_item_slice_float4(py::slice slice, const std::vector<std::vector<std::vector<float64>>>& value) = 0;

    virtual void set_item_slice_float5(py::slice slice, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) = 0;

    virtual int len() const = 0;

    virtual std::string to_string() const = 0;
};

template <typename T>
class ndarray : public ndarray_base {
public:
    pkpy::numpy::ndarray<T> data;
    // Constructors
    ndarray() = default;

    ndarray(const bool_ value) : data(value) {}

    ndarray(const int8 value) : data(value) {}

    ndarray(const int16 value) : data(value) {}

    ndarray(const int32 value) : data(value) {}

    ndarray(const int_ value) : data(static_cast<T>(value)) {}

    ndarray(const float32 value) : data(value) {}

    ndarray(const float64 value) : data(static_cast<T>(value)) {}

    ndarray(const pkpy::numpy::ndarray<T>& _arr) : data(_arr) {}

    ndarray(const std::vector<T>& init_list) : data(pkpy::numpy::adapt<T>(init_list)) {}

    ndarray(const std::vector<std::vector<T>>& init_list) : data(pkpy::numpy::adapt<T>(init_list)) {}

    ndarray(const std::vector<std::vector<std::vector<T>>>& init_list) : data(pkpy::numpy::adapt<T>(init_list)) {}

    ndarray(const std::vector<std::vector<std::vector<std::vector<T>>>>& init_list) :
        data(pkpy::numpy::adapt<T>(init_list)) {}

    ndarray(const std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>& init_list) :
        data(pkpy::numpy::adapt<T>(init_list)) {}

    // Properties
    int ndim() const override { return data.ndim(); }

    int size() const override { return data.size(); }

    std::string dtype() const override { return data.dtype(); }

    py::tuple shape() const override { return py::cast(data.shape()); }

    // Boolean Functions
    bool all() const override { return data.all(); }

    bool any() const override { return data.any(); }

    // Aggregation Functions
    py::object sum() const override {
        if constexpr (std::is_same_v<T, bool_> || std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64>) {
            return py::int_(data.sum());
        } else if constexpr(std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
            return py::float_(data.sum());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    py::object sum_axis(int axis) const override {
        if ((data.sum(axis)).ndim() == 0) {
            return py::cast((data.sum(axis))());
        } else {
            return py::cast(ndarray<T>(data.sum(axis)));
        }
    }

    py::object sum_axes(py::tuple axes) const override {
        std::vector<int> axes_;
        for(auto item: axes) {
            axes_.push_back(py::cast<int>(item));
        }
        if ((data.sum(axes_)).ndim() == 0) {
            return py::cast((data.sum(axes_))());
        } else {
            return py::cast(ndarray<T>(data.sum(axes_)));
        }
    }

    py::object prod() const override {
        if constexpr (std::is_same_v<T, bool_> || std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64>) {
            return py::int_(data.prod());
        } else if constexpr(std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
            return py::float_(data.prod());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    py::object prod_axis(int axis) const override {
        if ((data.prod(axis)).ndim() == 0) {
            return py::cast((data.prod(axis))());
        } else {
            return py::cast(ndarray<T>(data.prod(axis)));
        }
    }

    py::object prod_axes(py::tuple axes) const override {
        std::vector<int> axes_;
        for(auto item: axes) {
            axes_.push_back(py::cast<int>(item));
        }
        if ((data.prod(axes_)).ndim() == 0) {
            return py::cast((data.prod(axes_))());
        } else {
            return py::cast(ndarray<T>(data.prod(axes_)));
        }
    }

    py::object min() const override {
        if constexpr (std::is_same_v<T, bool_>) {
            return py::bool_(data.min());
        } else if constexpr (std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64>) {
            return py::int_(data.min());
        } else if constexpr(std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
            return py::float_(data.min());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    py::object min_axis(int axis) const override {
        if ((data.min(axis)).ndim() == 0) {
            return py::cast((data.min(axis))());
        } else {
            return py::cast(ndarray<T>(data.min(axis)));
        }

    }

    py::object min_axes(py::tuple axes) const override {
        std::vector<int> axes_;
        for(auto item: axes) {
            axes_.push_back(py::cast<int>(item));
        }
        if ((data.min(axes_)).ndim() == 0) {
            return py::cast((data.min(axes_))());
        } else {
            return py::cast(ndarray<T>(data.min(axes_)));
        }
    }

    py::object max() const override {
        if constexpr (std::is_same_v<T, bool_>) {
            return py::bool_(data.max());
        } else if constexpr (std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                             std::is_same_v<T, int32> || std::is_same_v<T, int64>) {
            return py::int_(data.max());
        } else if constexpr(std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
            return py::float_(data.max());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    py::object max_axis(int axis) const override {
        if ((data.max(axis)).ndim() == 0) {
            return py::cast((data.max(axis))());
        } else {
            return py::cast(ndarray<T>(data.max(axis)));
        }
    }

    py::object max_axes(py::tuple axes) const override {
        std::vector<int> axes_;
        for(auto item: axes) {
            axes_.push_back(py::cast<int>(item));
        }
        if ((data.max(axes_)).ndim() == 0) {
            return py::cast((data.max(axes_))());
        } else {
            return py::cast(ndarray<T>(data.max(axes_)));
        }
    }

    py::object mean() const override {
        if constexpr (std::is_same_v<T, bool_> || std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64> || std::is_same_v<T, float32> ||
                        std::is_same_v<T, float64>) {
            return py::float_(data.mean());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    py::object mean_axis(int axis) const override {
        if ((data.mean(axis)).ndim() == 0) {
            return py::cast((data.mean(axis))());
        } else {
            return py::cast(ndarray<float64>(data.mean(axis)));
        }
    }

    py::object mean_axes(py::tuple axes) const override {
        std::vector<int> axes_;
        for(auto item: axes)
            axes_.push_back(py::cast<int>(item));
        if ((data.mean(axes_)).ndim() == 0) {
            return py::cast((data.mean(axes_))());
        } else {
            return py::cast(ndarray<float64>(data.mean(axes_)));
        }
    }

    py::object std() const override {
        if constexpr (std::is_same_v<T, bool_> || std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64> || std::is_same_v<T, float32> ||
                        std::is_same_v<T, float64>) {
            return py::float_(data.std());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    py::object std_axis(int axis) const override {
        if ((data.std(axis)).ndim() == 0) {
            return py::cast((data.std(axis))());
        } else {
            return py::cast(ndarray<float64>(data.std(axis)));
        }
    }

    py::object std_axes(py::tuple axes) const override {
        std::vector<int> axes_;
        for(auto item: axes)
            axes_.push_back(py::cast<int>(item));
        if ((data.std(axes_)).ndim() == 0) {
            return py::cast((data.std(axes_))());
        } else {
            return py::cast(ndarray<float64>(data.std(axes_)));
        }
    }

    py::object var() const override {
        if constexpr (std::is_same_v<T, bool_> || std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64> || std::is_same_v<T, float32> ||
                        std::is_same_v<T, float64>) {
            return py::float_(data.var());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    py::object var_axis(int axis) const override {
        if ((data.var(axis)).ndim() == 0) {
            return py::cast((data.var(axis))());
        } else {
            return py::cast(ndarray<float64>(data.var(axis)));
        }
    }

    py::object var_axes(py::tuple axes) const override {
        std::vector<int> axes_;
        for(auto item: axes)
            axes_.push_back(py::cast<int>(item));
        if ((data.var(axes_)).ndim() == 0) {
            return py::cast((data.var(axes_))());
        } else {
            return py::cast(ndarray<float64>(data.var(axes_)));
        }
    }

    py::object argmin() const override {
        if constexpr (std::is_same_v<T, bool_> || std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64>) {
            return py::int_(data.argmin());
        } else if constexpr(std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
            return py::int_(data.argmin());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    ndarray_base* argmin_axis(int axis) const override { return new ndarray<T>(data.argmin(axis)); }

    py::object argmax() const override {
        if constexpr (std::is_same_v<T, bool_> || std::is_same_v<T, int8> || std::is_same_v<T, int16> ||
                      std::is_same_v<T, int32> || std::is_same_v<T, int64>) {
            return py::int_(data.argmax());
        } else if constexpr(std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
            return py::int_(data.argmax());
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }

    ndarray_base* argmax_axis(int axis) const override { return new ndarray<T>(data.argmax(axis)); }

    ndarray_base* argsort() const override { return new ndarray<T>(data.argsort()); }

    ndarray_base* argsort_axis(int axis) const override { return new ndarray<T>(data.argsort(axis)); }

    void sort() override { data = data.sort(); }

    void sort_axis(int axis) override { data = data.sort(axis); }

    ndarray_base* reshape(const std::vector<int>& shape) const override { return new ndarray<T>(data.reshape(shape)); }

    void resize(const std::vector<int>& shape) override { data = data.resize(shape); }

    ndarray_base* squeeze() const override { return new ndarray<T>(data.squeeze()); }

    ndarray_base* squeeze_axis(int axis) const override { return new ndarray<T>(data.squeeze(axis)); }

    ndarray_base* transpose() const override { return new ndarray<T>(data.transpose()); }

    ndarray_base* transpose_tuple(py::tuple permutations) const override {
        std::vector<int> perm;
        for(auto item: permutations)
            perm.push_back(py::cast<int>(item));
        return new ndarray<T>(data.transpose(perm));
    }

    ndarray_base* transpose_args(py::args args) const override {
        std::vector<int> perm;
        for(auto item: args)
            perm.push_back(py::cast<int>(item));
        return new ndarray<T>(data.transpose(perm));
    }

    ndarray_base* repeat(int repeats, int axis) const override {
        if (axis == INT_MAX) {
            return new ndarray<T>(data.repeat(repeats, data.ndim() - 1));
        }
        return new ndarray<T>(data.repeat(repeats, axis));
    }

    ndarray_base* repeat_axis(const std::vector<size_t>& repeats, int axis) const override {
        return new ndarray<T>(data.repeat(repeats, axis));
    }

    ndarray_base* round() const override { return new ndarray<T>(data.round()); }

    ndarray_base* flatten() const override { return new ndarray<T>(data.flatten()); }

    ndarray_base* copy() const override { return new ndarray<T>(data.copy()); }

    ndarray_base* astype(const std::string& dtype) const override {
        if(dtype == "bool_") {
            return new ndarray<bool_>(data.template astype<bool_>());
        } else if(dtype == "int8") {
            return new ndarray<int8>(data.template astype<int8>());
        } else if(dtype == "int16") {
            return new ndarray<int16>(data.template astype<int16>());
        } else if(dtype == "int32") {
            return new ndarray<int32>(data.template astype<int32>());
        } else if(dtype == "int_") {
            return new ndarray<int_>(data.template astype<int_>());
        } else if(dtype == "float32") {
            return new ndarray<float32>(data.template astype<float32>());
        } else if(dtype == "float64") {
            return new ndarray<float64>(data.template astype<float64>());
        } else {
            throw std::invalid_argument("Invalid dtype");
        }
    }

    py::list tolist() const override {
        py::list list;
        if(data.ndim() == 1) {
            return py::cast(data.to_list());
        } else {
            for(int i = 0; i < data.shape()[0]; i++) {
                list.append(ndarray<T>(data[i]).tolist());
            }
        }
        return list;
    }

    // Dunder Methods
    ndarray_base* eq(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 == int8 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 == int16 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 == int32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 == int64 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 == float32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 == float64 */
                return new ndarray<bool_>(data == p->data);
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 == int8 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 == int16 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 == int32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 == int64 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 == float32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 == float64 */
                return new ndarray<bool_>(data == p->data);
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 == int8 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 == int16 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 == int32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 == int64 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 == float32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 == float64 */
                return new ndarray<bool_>(data == p->data);
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 == int8 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 == int16 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 == int32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 == int64 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 == float32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 == float64 */
                return new ndarray<bool_>(data == p->data);
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 == int8 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 == int16 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 == int32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 == int64 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 == float32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 == float64 */
                return new ndarray<bool_>(data == p->data);
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 == int8 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 == int16 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 == int32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 == int64 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 == float32 */
                return new ndarray<bool_>(data == p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 == float64 */
                return new ndarray<bool_>(data == p->data);
            }
        }

        const ndarray<T>& other_ = dynamic_cast<const ndarray<T>&>(other);
        return new ndarray<bool_>(data == other_.data);
    }

    ndarray_base* ne(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 != int8 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 != int16 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 != int32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 != int64 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 != float32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 != float64 */
                return new ndarray<bool_>(data != p->data);
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 != int8 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 != int16 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 != int32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 != int64 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 != float32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 != float64 */
                return new ndarray<bool_>(data != p->data);
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 != int8 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 != int16 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 != int32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 != int64 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 != float32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 != float64 */
                return new ndarray<bool_>(data != p->data);
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 != int8 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 != int16 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 != int32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 != int64 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 != float32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 != float64 */
                return new ndarray<bool_>(data != p->data);
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 != int8 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 != int16 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 != int32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 != int64 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 != float32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 != float64 */
                return new ndarray<bool_>(data != p->data);
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 != int8 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 != int16 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 != int32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 != int64 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 != float32 */
                return new ndarray<bool_>(data != p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 != float64 */
                return new ndarray<bool_>(data != p->data);
            }
        }

        const ndarray<T>& other_ = dynamic_cast<const ndarray<T>&>(other);
        return new ndarray<bool_>(data != other_.data);
    }

    ndarray_base* add(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 + int8 */
                return new ndarray<int8>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 + int16 */
                return new ndarray<int16>((data + p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 + int32 */
                return new ndarray<int32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 + int64 */
                return new ndarray<int_>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 + float32 */
                return new ndarray<float32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 + float64 */
                return new ndarray<float64>(data + p->data);
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 + int8 */
                return new ndarray<int16>((data + p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 + int16 */
                return new ndarray<int16>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 + int32 */
                return new ndarray<int32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 + int64 */
                return new ndarray<int_>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 + float32 */
                return new ndarray<float32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 + float64 */
                return new ndarray<float64>(data + p->data);
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 + int8 */
                return new ndarray<int32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 + int16 */
                return new ndarray<int32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 + int32 */
                return new ndarray<int32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 + int64 */
                return new ndarray<int_>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 + float32 */
                return new ndarray<float32>(data + p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 + float64 */
                return new ndarray<float64>(data + p->data);
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 + int8 */
                return new ndarray<int_>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 + int16 */
                return new ndarray<int_>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 + int32 */
                return new ndarray<int_>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 + int64 */
                return new ndarray<int_>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 + float32 */
                return new ndarray<float32>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 + float64 */
                return new ndarray<float64>(data + p->data);
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 + int8 */
                return new ndarray<float32>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 + int16 */
                return new ndarray<float32>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 + int32 */
                return new ndarray<float32>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 + int64 */
                return new ndarray<float32>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 + float32 */
                return new ndarray<float32>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 + float64 */
                return new ndarray<float64>(data + p->data);
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 + int8 */
                return new ndarray<float64>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 + int16 */
                return new ndarray<float64>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 + int32 */
                return new ndarray<float64>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 + int64 */
                return new ndarray<float64>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 + float32 */
                return new ndarray<float64>(data + p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 + float64 */
                return new ndarray<float64>(data + p->data);
            }
        }

        const ndarray<T>& other_ = dynamic_cast<const ndarray<T>&>(other);
        return new ndarray<T>(data + other_.data);
    }

    ndarray_base* add_bool(bool_ other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>((data + other).template astype<int8>());
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>((data + other).template astype<int16>());
        } else {
            return new ndarray<T>(data + other);
        }
    }

    ndarray_base* add_int(int_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<int_>(data + other);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>((data + other).template astype<int8>());
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>((data + other).template astype<int16>());
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>((data + other).template astype<int32>());
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int64>(data + other);
        } else if constexpr(std::is_same_v<T, float32>) {
            return new ndarray<float32>(data + other);
        } else if constexpr(std::is_same_v<T, float64>) {
            return new ndarray<float64>(data + other);
        }
    }

    ndarray_base* add_float(float64 other) const override { return new ndarray<float64>(data + other); }

    ndarray_base* sub(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 - int8 */
                return new ndarray<int8>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 - int16 */
                return new ndarray<int16>((data - p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 - int32 */
                return new ndarray<int32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 - int64 */
                return new ndarray<int_>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 - float32 */
                return new ndarray<float32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 - float64 */
                return new ndarray<float64>(data - p->data);
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 - int8 */
                return new ndarray<int16>((data - p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 - int16 */
                return new ndarray<int16>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 - int32 */
                return new ndarray<int32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 - int64 */
                return new ndarray<int_>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 - float32 */
                return new ndarray<float32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 - float64 */
                return new ndarray<float64>(data - p->data);
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 - int8 */
                return new ndarray<int32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 - int16 */
                return new ndarray<int32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 - int32 */
                return new ndarray<int32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 - int64 */
                return new ndarray<int_>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 - float32 */
                return new ndarray<float32>(data - p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 - float64 */
                return new ndarray<float64>(data - p->data);
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 - int8 */
                return new ndarray<int_>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 - int16 */
                return new ndarray<int_>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 - int32 */
                return new ndarray<int_>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 - int64 */
                return new ndarray<int_>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 - float32 */
                return new ndarray<float32>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 - float64 */
                return new ndarray<float64>(data - p->data);
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 - int8 */
                return new ndarray<float32>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 - int16 */
                return new ndarray<float32>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 - int32 */
                return new ndarray<float32>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 - int64 */
                return new ndarray<float32>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 - float32 */
                return new ndarray<float32>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 - float64 */
                return new ndarray<float64>(data - p->data);
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 - int8 */
                return new ndarray<float64>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 - int16 */
                return new ndarray<float64>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 - int32 */
                return new ndarray<float64>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 - int64 */
                return new ndarray<float64>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 - float32 */
                return new ndarray<float64>(data - p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 - float64 */
                return new ndarray<float64>(data - p->data);
            }
        }

        const ndarray<T>& other_ = dynamic_cast<const ndarray<T>&>(other);
        return new ndarray<T>(data - other_.data);
    }

    ndarray_base* sub_int(int_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<int_>(data - other);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>((data - other).template astype<int8>());
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>((data - other).template astype<int16>());
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>((data - other).template astype<int32>());
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int64>(data - other);
        } else if constexpr(std::is_same_v<T, float32>) {
            return new ndarray<float32>(data - other);
        } else if constexpr(std::is_same_v<T, float64>) {
            return new ndarray<float64>(data - other);
        }
    }

    ndarray_base* sub_float(float64 other) const override { return new ndarray<float64>(data - other); }

    ndarray_base* rsub_int(int_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<int_>(other - data);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>((other - data).template astype<int8>());
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>((other - data).template astype<int16>());
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>((other - data).template astype<int32>());
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int64>(other - data);
        } else if constexpr(std::is_same_v<T, float32>) {
            return new ndarray<float32>(other - data);
        } else if constexpr(std::is_same_v<T, float64>) {
            return new ndarray<float64>(other - data);
        }
    }

    ndarray_base* rsub_float(float64 other) const override { return new ndarray<float64>(other - data); }

    ndarray_base* mul(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 * int8 */
                return new ndarray<int8>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 * int16 */
                return new ndarray<int16>((data * p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 * int32 */
                return new ndarray<int32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 * int64 */
                return new ndarray<int_>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 * float32 */
                return new ndarray<float32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 * float64 */
                return new ndarray<float64>(data * p->data);
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 * int8 */
                return new ndarray<int16>((data * p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 * int16 */
                return new ndarray<int16>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 * int32 */
                return new ndarray<int32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 * int64 */
                return new ndarray<int_>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 * float32 */
                return new ndarray<float32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 * float64 */
                return new ndarray<float64>(data * p->data);
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 * int8 */
                return new ndarray<int32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 * int16 */
                return new ndarray<int32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 * int32 */
                return new ndarray<int32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 * int64 */
                return new ndarray<int_>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 * float32 */
                return new ndarray<float32>(data * p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 * float64 */
                return new ndarray<float64>(data * p->data);
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 * int8 */
                return new ndarray<int_>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 * int16 */
                return new ndarray<int_>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 * int32 */
                return new ndarray<int_>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 * int64 */
                return new ndarray<int_>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 * float32 */
                return new ndarray<float32>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 * float64 */
                return new ndarray<float64>(data * p->data);
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 * int8 */
                return new ndarray<float32>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 * int16 */
                return new ndarray<float32>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 * int32 */
                return new ndarray<float32>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 * int64 */
                return new ndarray<float32>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 * float32 */
                return new ndarray<float32>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 * float64 */
                return new ndarray<float64>(data * p->data);
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 * int8 */
                return new ndarray<float64>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 * int16 */
                return new ndarray<float64>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 * int32 */
                return new ndarray<float64>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 * int64 */
                return new ndarray<float64>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 * float32 */
                return new ndarray<float64>(data * p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 * float64 */
                return new ndarray<float64>(data * p->data);
            }
        }

        const ndarray<T>& other_ = dynamic_cast<const ndarray<T>&>(other);
        return new ndarray<T>(data * other_.data);
    }

    ndarray_base* mul_bool(bool_ other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>((data * other).template astype<int8>());
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>((data * other).template astype<int16>());
        } else {
            return new ndarray<T>(data * other);
        }
    }

    ndarray_base* mul_int(int_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<int_>(data * other);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>((data * other).template astype<int8>());
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>((data * other).template astype<int16>());
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>((data * other).template astype<int32>());
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int64>(data * other);
        } else if constexpr(std::is_same_v<T, float32>) {
            return new ndarray<float32>(data * other);
        } else if constexpr(std::is_same_v<T, float64>) {
            return new ndarray<float64>(data * other);
        }
    }

    ndarray_base* mul_float(float64 other) const override { return new ndarray<float64>(data * other); }

    ndarray_base* div(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            if(auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* bool / bool */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* bool / int8 */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* bool / int16 */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* bool / int32 */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* bool / int64 */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* bool / float32 */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* bool / float64 */
                return new ndarray<float64>(data / p->data);
            }
        } else if constexpr(std::is_same_v<T, int8>) {
            if (auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int8 / bool */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 / int8 */
                return new ndarray<int8>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 / int16 */
                return new ndarray<int16>((data / p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 / int32 */
                return new ndarray<int32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 / int64 */
                return new ndarray<int_>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 / float32 */
                return new ndarray<float32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 / float64 */
                return new ndarray<float64>(data / p->data);
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if (auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int16 / bool */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 / int8 */
                return new ndarray<int16>((data / p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 / int16 */
                return new ndarray<int16>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 / int32 */
                return new ndarray<int32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 / int64 */
                return new ndarray<int_>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 / float32 */
                return new ndarray<float32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 / float64 */
                return new ndarray<float64>(data / p->data);
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if (auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int32 / bool */
                return new ndarray<float64>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 / int8 */
                return new ndarray<int32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 / int16 */
                return new ndarray<int32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 / int32 */
                return new ndarray<int32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 / int64 */
                return new ndarray<int_>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 / float32 */
                return new ndarray<float32>(data / p->data);
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 / float64 */
                return new ndarray<float64>(data / p->data);
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int64 / bool */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 / int8 */
                return new ndarray<int_>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 / int16 */
                return new ndarray<int_>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 / int32 */
                return new ndarray<int_>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 / int64 */
                return new ndarray<int_>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 / float32 */
                return new ndarray<float32>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 / float64 */
                return new ndarray<float64>(data / p->data);
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if (auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* float32 / bool */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 / int8 */
                return new ndarray<float32>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 / int16 */
                return new ndarray<float32>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 / int32 */
                return new ndarray<float32>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 / int64 */
                return new ndarray<float32>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 / float32 */
                return new ndarray<float32>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 / float64 */
                return new ndarray<float64>(data / p->data);
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if (auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* float64 / bool */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 / int8 */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 / int16 */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 / int32 */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 / int64 */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 / float32 */
                return new ndarray<float64>(data / p->data);
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 / float64 */
                return new ndarray<float64>(data / p->data);
            }
        }

        const ndarray<float64>& other_ = dynamic_cast<const ndarray<float64>&>(other);
        return new ndarray<float64>(data / other_.data);
    }

    ndarray_base* div_bool(bool_ other) const override { return new ndarray<float64>(data / other); }

    ndarray_base* div_int(int_ other) const override { return new ndarray<float64>(data / other); }

    ndarray_base* div_float(float64 other) const override { return new ndarray<float64>(data / other); }

    ndarray_base* rdiv_bool(bool_ other) const override { return new ndarray<float64>(other / data); }

    ndarray_base* rdiv_int(int_ other) const override { return new ndarray<float64>(other / data); }

    ndarray_base* rdiv_float(float64 other) const override { return new ndarray<float64>(other / data); }

    ndarray_base* matmul(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 @ int8 */
                return new ndarray<int8>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 @ int16 */
                return new ndarray<int16>(pkpy::numpy::matmul(data, p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 @ int32 */
                return new ndarray<int32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 @ int64 */
                return new ndarray<int_>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 @ float32 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 @ float64 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 @ int8 */
                return new ndarray<int16>(pkpy::numpy::matmul(data, p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 @ int16 */
                return new ndarray<int16>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 @ int32 */
                return new ndarray<int32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 @ int64 */
                return new ndarray<int_>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 @ float32 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 @ float64 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 @ int8 */
                return new ndarray<int32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 @ int16 */
                return new ndarray<int32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 @ int32 */
                return new ndarray<int32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 @ int64 */
                return new ndarray<int_>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 @ float32 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 @ float64 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 @ int8 */
                return new ndarray<int_>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 @ int16 */
                return new ndarray<int_>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 @ int32 */
                return new ndarray<int_>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 @ int64 */
                return new ndarray<int_>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 @ float32 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 @ float64 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 @ int8 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 @ int16 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 @ int32 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 @ int64 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 @ float32 */
                return new ndarray<float32>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 @ float64 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 @ int8 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 @ int16 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 @ int32 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 @ int64 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 @ float32 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 @ float64 */
                return new ndarray<float64>(pkpy::numpy::matmul(data, p->data));
            }
        }

        const ndarray<T>& other_ = dynamic_cast<const ndarray<T>&>(other);
        return new ndarray<T>(pkpy::numpy::matmul(data, other_.data));
    }

    ndarray_base* pow(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 ** int8 */
                return new ndarray<int8>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 ** int16 */
                return new ndarray<int16>(data.pow(p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 ** int32 */
                return new ndarray<int32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 ** int64 */
                return new ndarray<int_>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int8 ** float32 */
                return new ndarray<float32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int8 ** float64 */
                return new ndarray<float64>(data.pow(p->data));
            }
        } else if constexpr(std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 ** int8 */
                return new ndarray<int16>(data.pow(p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 ** int16 */
                return new ndarray<int16>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 ** int32 */
                return new ndarray<int32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 ** int64 */
                return new ndarray<int_>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int16 ** float32 */
                return new ndarray<float32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int16 ** float64 */
                return new ndarray<float64>(data.pow(p->data));
            }
        } else if constexpr(std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 ** int8 */
                return new ndarray<int32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 ** int16 */
                return new ndarray<int32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 ** int32 */
                return new ndarray<int32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 ** int64 */
                return new ndarray<int_>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int32 ** float32 */
                return new ndarray<float32>(data.pow(p->data));
            } else if(auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int32 ** float64 */
                return new ndarray<float64>(data.pow(p->data));
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int64 ** int8 */
                return new ndarray<int_>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int64 ** int16 */
                return new ndarray<int_>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int64 ** int32 */
                return new ndarray<int_>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int64 ** int64 */
                return new ndarray<int_>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* int64 ** float32 */
                return new ndarray<float32>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* int64 ** float64 */
                return new ndarray<float64>(data.pow(p->data));
            }
        } else if constexpr(std::is_same_v<T, float32>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float32 ** int8 */
                return new ndarray<float32>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float32 ** int16 */
                return new ndarray<float32>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float32 ** int32 */
                return new ndarray<float32>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float32 ** int64 */
                return new ndarray<float32>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float32 ** float32 */
                return new ndarray<float32>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float32 ** float64 */
                return new ndarray<float64>(data.pow(p->data));
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if (auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* float64 ** int8 */
                return new ndarray<float64>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* float64 ** int16 */
                return new ndarray<float64>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* float64 ** int32 */
                return new ndarray<float64>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* float64 ** int64 */
                return new ndarray<float64>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<float32>*>(&other)) { /* float64 ** float32 */
                return new ndarray<float64>(data.pow(p->data));
            } else if (auto p = dynamic_cast<const ndarray<float64>*>(&other)) { /* float64 ** float64 */
                return new ndarray<float64>(data.pow(p->data));
            }
        }

        const ndarray<T>& other_ = dynamic_cast<const ndarray<T>&>(other);
        return new ndarray<T>(data.pow(other_.data));
    }

    ndarray_base* pow_int(int_ other) const override { return new ndarray<float64>(data.pow(other)); }

    ndarray_base* pow_float(float64 other) const override { return new ndarray<float64>(data.pow(other)); }

    ndarray_base* rpow_int(int_ other) const override { return new ndarray<float64>(pkpy::numpy::pow(other, data)); }

    ndarray_base* rpow_float(float64 other) const override {
        return new ndarray<float64>(pkpy::numpy::pow(other, data));
    }

    int len() const override { return data.shape()[0]; }

    ndarray_base* and_array(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            if(auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* bool & bool */
                return new ndarray<bool_>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* bool & int8 */
                return new ndarray<int8>((data & p->data).template astype<int8>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* bool & int16 */
                return new ndarray<int16>((data & p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* bool & int32 */
                return new ndarray<int32>((data & p->data).template astype<int32>());
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* bool & int64 */
                return new ndarray<int_>((data & p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int8 & bool */
                return new ndarray<int8>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 & int8 */
                return new ndarray<int8>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 & int16 */
                return new ndarray<int16>((data & p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 & int32 */
                return new ndarray<int32>((data & p->data).template astype<int32>());
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 & int64 */
                return new ndarray<int_>((data & p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int16>) {
            if(auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int16 & bool */
                return new ndarray<int16>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int16 & int8 */
                return new ndarray<int16>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int16 & int16 */
                return new ndarray<int16>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int16 & int32 */
                return new ndarray<int32>((data & p->data).template astype<int32>());
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int16 & int64 */
                return new ndarray<int_>((data & p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int32>) {
            if(auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int32 & bool */
                return new ndarray<int32>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int32 & int8 */
                return new ndarray<int32>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int32 & int16 */
                return new ndarray<int32>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int32 & int32 */
                return new ndarray<int32>(data & p->data);
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int32 & int64 */
                return new ndarray<int_>((data & p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int64 & bool */
                return new ndarray<int_>(data & p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int64 & int8 */
                return new ndarray<int_>(data & p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int64 & int16 */
                return new ndarray<int_>(data & p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int64 & int32 */
                return new ndarray<int_>(data & p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int64 & int64 */
                return new ndarray<int_>(data & p->data);
            }
        }

        throw std::runtime_error("& operator is not compatible with floating types");
    }

    ndarray_base* and_bool(bool_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<bool_>(data & other);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>(data & other);
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>(data & other);
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>(data & other);
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int_>(data & other);
        }

        throw std::runtime_error("& operator is not compatible with floating types");
    }

    ndarray_base* and_int(int_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<int_>((data & other).template astype<int_>());
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>(data & other);
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>(data & other);
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>(data & other);
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int_>(data & other);
        }

        throw std::runtime_error("& operator is not compatible with floating types");
    }

    ndarray_base* or_array(const ndarray_base& other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            if(auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* bool | bool */
                return new ndarray<bool_>(data | p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* bool | int8 */
                return new ndarray<int8>((data | p->data).template astype<int8>());
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* bool | int16 */
                return new ndarray<int16>((data | p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* bool | int32 */
                return new ndarray<int32>((data | p->data).template astype<int32>());
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* bool | int64 */
                return new ndarray<int_>((data | p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int8>) {
            if(auto p = dynamic_cast<const ndarray<bool_>*>(&other)) { /* int8 | bool */
                return new ndarray<int8>(data | p->data);
            } else if(auto p = dynamic_cast<const ndarray<int8>*>(&other)) { /* int8 | int8 */
                return new ndarray<int8>(data | p->data);
            } else if(auto p = dynamic_cast<const ndarray<int16>*>(&other)) { /* int8 | int16 */
                return new ndarray<int16>((data | p->data).template astype<int16>());
            } else if(auto p = dynamic_cast<const ndarray<int32>*>(&other)) { /* int8 | int32 */
                return new ndarray<int32>((data | p->data).template astype<int32>());
            } else if(auto p = dynamic_cast<const ndarray<int_>*>(&other)) { /* int8 | int64 */
                return new ndarray<int_>((data | p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int16>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int16 | bool */
                return new ndarray<int16>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int16 | int8 */
                return new ndarray<int16>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int16 | int16 */
                return new ndarray<int16>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int16 | int32 */
                return new ndarray<int32>((data | p->data).template astype<int32>());
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int16 | int64 */
                return new ndarray<int_>((data | p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int32>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int32 | bool */
                return new ndarray<int32>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int32 | int8 */
                return new ndarray<int32>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int32 | int16 */
                return new ndarray<int32>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int32 | int32 */
                return new ndarray<int32>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int32 | int64 */
                return new ndarray<int_>((data | p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int64 | bool */
                return new ndarray<int_>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int64 | int8 */
                return new ndarray<int_>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int64 | int16 */
                return new ndarray<int_>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int64 | int32 */
                return new ndarray<int_>(data | p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int64 | int64 */
                return new ndarray<int_>(data | p->data);
            }
        }

        throw std::runtime_error("| operator is not compatible with floating types");
    }

    ndarray_base* or_bool(bool_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<bool_>(data | other);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>(data | other);
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>(data | other);
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>(data | other);
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int_>(data | other);
        }

        throw std::runtime_error("| operator is not compatible with floating types");
    }

    ndarray_base* or_int(int_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<int_>((data | other).template astype<int_>());
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>(data | other);
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>(data | other);
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>(data | other);
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int_>(data | other);
        }

        throw std::runtime_error("| operator is not compatible with floating types");
    }

    ndarray_base* xor_array(const ndarray_base& other) const override {
        if constexpr (std::is_same_v<T, bool_>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* bool ^ bool */
                return new ndarray<bool_>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* bool ^ int8 */
                return new ndarray<int8>((data ^ p->data).template astype<int8>());
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* bool ^ int16 */
                return new ndarray<int16>((data ^ p->data).template astype<int16>());
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* bool ^ int32 */
                return new ndarray<int32>((data ^ p->data).template astype<int32>());
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* bool ^ int64 */
                return new ndarray<int_>((data ^ p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int8>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int8 ^ bool */
                return new ndarray<int8>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int8 ^ int8 */
                return new ndarray<int8>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int8 ^ int16 */
                return new ndarray<int16>((data ^ p->data).template astype<int16>());
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int8 ^ int32 */
                return new ndarray<int32>((data ^ p->data).template astype<int32>());
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int8 ^ int64 */
                return new ndarray<int_>((data ^ p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int16>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int16 ^ bool */
                return new ndarray<int16>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int16 ^ int8 */
                return new ndarray<int16>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int16 ^ int16 */
                return new ndarray<int16>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int16 ^ int32 */
                return new ndarray<int32>((data ^ p->data).template astype<int32>());
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int16 ^ int64 */
                return new ndarray<int_>((data ^ p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int32>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int32 ^ bool */
                return new ndarray<int32>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int32 ^ int8 */
                return new ndarray<int32>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int32 ^ int16 */
                return new ndarray<int32>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int32 ^ int32 */
                return new ndarray<int32>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int32 ^ int64 */
                return new ndarray<int_>((data ^ p->data).template astype<int_>());
            }
        } else if constexpr (std::is_same_v<T, int_>) {
            if (auto p = dynamic_cast<const ndarray<bool_> *>(&other)) { /* int64 ^ bool */
                return new ndarray<int_>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int8> *>(&other)) { /* int64 ^ int8 */
                return new ndarray<int_>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int16> *>(&other)) { /* int64 ^ int16 */
                return new ndarray<int_>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int32> *>(&other)) { /* int64 ^ int32 */
                return new ndarray<int_>(data ^ p->data);
            } else if (auto p = dynamic_cast<const ndarray<int_> *>(&other)) { /* int64 ^ int64 */
                return new ndarray<int_>(data ^ p->data);
            }
        }

        throw std::runtime_error("^ operator is not compatible with floating types");
    }

    ndarray_base* xor_bool(bool_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<bool_>(data ^ other);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>(data ^ other);
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>(data ^ other);
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>(data ^ other);
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int_>(data ^ other);
        }

        throw std::runtime_error("^ operator is not compatible with floating types");
    }

    ndarray_base* xor_int(int_ other) const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<int_>((data ^ other).template astype<int_>());
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>(data ^ other);
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>(data ^ other);
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>(data ^ other);
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int_>(data ^ other);
        }

        throw std::runtime_error("^ operator is not compatible with floating types");
    }

    ndarray_base* invert() const override {
        if constexpr(std::is_same_v<T, bool_>) {
            return new ndarray<bool_>(!data);
        } else if constexpr(std::is_same_v<T, int8>) {
            return new ndarray<int8>(!data);
        } else if constexpr(std::is_same_v<T, int16>) {
            return new ndarray<int16>(!data);
        } else if constexpr(std::is_same_v<T, int32>) {
            return new ndarray<int32>(!data);
        } else if constexpr(std::is_same_v<T, int_>) {
            return new ndarray<int_>(!data);
        }

        throw std::runtime_error("~ operator is not compatible with floating types");
    }

    py::object get_item_int(int index) const override {
        if(index < 0) index += data.shape()[0];
        if(data.ndim() == 1) {
            if constexpr(std::is_same_v<T, bool_>) {
                return py::bool_(data(index));
            } else if constexpr(std::is_same_v<T, int_>) {
                return py::int_(data(index));
            } else if constexpr(std::is_same_v<T, float64>) {
                return py::float_(data(index));
            }
        } 
        return py::cast(ndarray<T>(data[index]));
    }

    py::object get_item_tuple(py::tuple args) const override {
        pkpy::numpy::ndarray<T> store = data;
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        for(int i = 0; i < indices.size() - 1; i++) {
            if(indices[i] < 0) indices[i] += store.shape()[0];
            pkpy::numpy::ndarray<T> temp = store[indices[i]];
            store = temp;
        }

        if(indices[indices.size() - 1] < 0) indices[indices.size() - 1] += store.shape()[0];
        if(store.ndim() == 1) {
            if constexpr(std::is_same_v<T, bool_>) {
                return py::bool_(store(indices[indices.size() - 1]));
            } else if constexpr(std::is_same_v<T, int_>) {
                return py::int_(store(indices[indices.size() - 1]));
            } else if constexpr(std::is_same_v<T, float64>) {
                return py::float_(store(indices[indices.size() - 1]));
            }
        } 
        return py::cast(ndarray<T>(store[indices[indices.size() - 1]]));
    }

    ndarray_base* get_item_vector(const std::vector<int>& indices) const override {
        return new ndarray<T>(data[indices]);
    }

    ndarray_base* get_item_slice(py::slice slice) const override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }

        return new ndarray<T>(data[std::make_tuple(start, stop, step)]);
    }

    void set_item_int(int index, int_ value) override {
        if constexpr(std::is_same_v<T, int_>) {
            if (data.ndim() == 1) {
                data.set_item(index, value);
            } else {
                data.set_item(index, pkpy::numpy::adapt<int_>(std::vector{value}));
            }
        } else if constexpr(std::is_same_v<T, float64>) {
            if (data.ndim() == 1) {
                data.set_item(index, static_cast<T>(value));
            } else {
                data.set_item(index, (pkpy::numpy::adapt<int_>(std::vector{value})).astype<float64>());
            }
        }
    }

    void set_item_index_int(int index, const std::vector<int_>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_index_int_2d(int index, const std::vector<std::vector<int_>>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_index_int_3d(int index, const std::vector<std::vector<std::vector<int_>>>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_index_int_4d(int index, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_float(int index, float64 value) override {
        if constexpr(std::is_same_v<T, float64>) {
            if (data.ndim() == 1) {
                data.set_item(index, value);
            } else {
                data.set_item(index, pkpy::numpy::adapt<float64>(std::vector{value}));
            }
        } else if constexpr(std::is_same_v<T, int_>) {
            if (data.ndim() == 1) {
                data.set_item(index, static_cast<T>(value));
            } else {
                data.set_item(index, (pkpy::numpy::adapt<float64>(std::vector{value})).astype<int_>());
            }
        }
    }

    void set_item_index_float(int index, const std::vector<float64>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_index_float_2d(int index, const std::vector<std::vector<float64>>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_index_float_3d(int index, const std::vector<std::vector<std::vector<float64>>>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_index_float_4d(int index, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(index, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_tuple_int1(py::tuple args, int_ value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, pkpy::numpy::adapt<int_>(std::vector{value}));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, (pkpy::numpy::adapt<int_>(std::vector{value})).astype<float64>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], static_cast<T>(value));
        else if(indices.size() == 3 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], indices[2], static_cast<T>(value));
        else if(indices.size() == 4 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], indices[2], indices[3], static_cast<T>(value));
        else if(indices.size() == 5 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], static_cast<T>(value));
    }

    void set_item_tuple_int2(py::tuple args, const std::vector<int_>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        }
    }

    void set_item_tuple_int3(py::tuple args, const std::vector<std::vector<int_>>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        }
    }

    void set_item_tuple_int4(py::tuple args, const std::vector<std::vector<std::vector<int_>>>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        }
    }

    void set_item_tuple_int5(py::tuple args, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        } else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr(std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<int_>(value));
            } else if constexpr(std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<int_>(value)).astype<float64>());
            }
        }
    }

    void set_item_tuple_float1(py::tuple args, float64 value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, pkpy::numpy::adapt<float64>(std::vector{value}));
            } else if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, (pkpy::numpy::adapt<float64>(std::vector{value})).astype<int_>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], static_cast<T>(value));
        else if(indices.size() == 3 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], indices[2], static_cast<T>(value));
        else if(indices.size() == 4 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], indices[2], indices[3], static_cast<T>(value));
        else if(indices.size() == 5 && indices.size() <= data.ndim())
            data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], static_cast<T>(value));
    }

    void set_item_tuple_float2(py::tuple args, const std::vector<float64>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, pkpy::numpy::adapt<float64>(value));
            } else if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
    }

    void set_item_tuple_float3(py::tuple args, const std::vector<std::vector<float64>>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, pkpy::numpy::adapt<float64>(value));
            } else if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
    }

    void set_item_tuple_float4(py::tuple args, const std::vector<std::vector<std::vector<float64>>>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, pkpy::numpy::adapt<float64>(value));
            } else if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
    }

    void set_item_tuple_float5(py::tuple args, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) override {
        std::vector<int> indices;
        for(auto item: args) {
            indices.push_back(py::cast<int>(item));
        }
        if(indices.size() == 1) {
            int index = indices[0];
            if constexpr(std::is_same_v<T, float64>) {
                data.set_item(index, pkpy::numpy::adapt<float64>(value));
            } else if constexpr(std::is_same_v<T, int_>) {
                data.set_item(index, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        } else if(indices.size() == 2 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 3 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 4 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
        else if(indices.size() == 5 && indices.size() <= data.ndim()) {
            if constexpr (std::is_same_v<T, float64>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], pkpy::numpy::adapt<float64>(value));
            } else if constexpr (std::is_same_v<T, int_>) {
                data.set_item(indices[0], indices[1], indices[2], indices[3], indices[4], (pkpy::numpy::adapt<float64>(value)).astype<int_>());
            }
        }
    }

    void set_item_vector_int1(const std::vector<int>& indices, int_ value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, pkpy::numpy::adapt<int_>(std::vector{value}));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, (pkpy::numpy::adapt<int_>(std::vector{value})).astype<float64>());
        }
    }

    void set_item_vector_int2(const std::vector<int>& indices, const std::vector<int_>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_vector_int3(const std::vector<int>& indices, const std::vector<std::vector<int_>>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_vector_int4(const std::vector<int>& indices, const std::vector<std::vector<std::vector<int_>>>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_vector_int5(const std::vector<int>& indices, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) override {
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_vector_float1(const std::vector<int>& indices, float64 value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, pkpy::numpy::adapt<float64>(std::vector{value}));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, (pkpy::numpy::adapt<float64>(std::vector{value})).astype<int_>());
        }
    }

    void set_item_vector_float2(const std::vector<int>& indices, const std::vector<float64>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_vector_float3(const std::vector<int>& indices, const std::vector<std::vector<float64>>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_vector_float4(const std::vector<int>& indices, const std::vector<std::vector<std::vector<float64>>>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_vector_float5(const std::vector<int>& indices, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) override {
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(indices, pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(indices, (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_slice_int1(py::slice slice, int_ value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<int_>(std::vector{value}));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step),
                          (pkpy::numpy::adapt<int_>(std::vector{value})).astype<float64>());
        }
    }

    void set_item_slice_int2(py::slice slice, const std::vector<int_>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_slice_int3(py::slice slice, const std::vector<std::vector<int_>>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_slice_int4(py::slice slice, const std::vector<std::vector<std::vector<int_>>>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_slice_int5(py::slice slice, const std::vector<std::vector<std::vector<std::vector<int_>>>>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<int_>(value));
        } else if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<int_>(value)).astype<float64>());
        }
    }

    void set_item_slice_float1(py::slice slice, float64 value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<float64>(std::vector{value}));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step),
                          (pkpy::numpy::adapt<float64>(std::vector{value})).astype<int_>());
        }
    }

    void set_item_slice_float2(py::slice slice, const std::vector<float64>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_slice_float3(py::slice slice, const std::vector<std::vector<float64>>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_slice_float4(py::slice slice, const std::vector<std::vector<std::vector<float64>>>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    void set_item_slice_float5(py::slice slice, const std::vector<std::vector<std::vector<std::vector<float64>>>>& value) override {
        int start = parseAttr(getattr(slice, "start"));
        int stop = parseAttr(getattr(slice, "stop"));
        int step = parseAttr(getattr(slice, "step"));

        if(step == INT_MAX) step = 1;
        if(step > 0) {
            if(start == INT_MAX) start = 0;
            if(stop == INT_MAX) stop = data.shape()[0];
        } else if(step < 0) {
            if(start == INT_MAX) start = data.shape()[0] - 1;
            if(stop == INT_MAX) stop = -(1 + data.shape()[0]);
        }
        if constexpr(std::is_same_v<T, float64>) {
            data.set_item(std::make_tuple(start, stop, step), pkpy::numpy::adapt<float64>(value));
        } else if constexpr(std::is_same_v<T, int_>) {
            data.set_item(std::make_tuple(start, stop, step), (pkpy::numpy::adapt<float64>(value)).astype<int_>());
        }
    }

    std::string to_string() const override {
        std::ostringstream os;
        os << data;
        std::string result = os.str();

        size_t pos = 0;
        while ((pos = result.find('{', pos)) != std::string::npos) {
            result.replace(pos, 1, "[");
            pos += 1;
        }
        pos = 0;
        while ((pos = result.find('}', pos)) != std::string::npos) {
            result.replace(pos, 1, "]");
            pos += 1;
        }

        if constexpr(std::is_same_v<T, bool_>) {
            size_t pos = 0;
            while ((pos = result.find("true", pos)) != std::string::npos) {
                result.replace(pos, 4, "True");
                pos += 4;
            }
            pos = 0;
            while ((pos = result.find("false", pos)) != std::string::npos) {
                result.replace(pos, 5, "False");
                pos += 5;
            }
        }

        for(int i = 0; i < result.size(); i++) {
            if(result[i] == '\n') {
                result.insert(i + 1, "      ");
            }
        }
        return result;
    }
};

