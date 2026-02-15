#pragma once

#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <ndarray_binding.hpp>

namespace pkbind {
namespace detail {

// --- A) Eigen type detection traits ---

template <typename T, typename = void>
struct is_eigen_dense : std::false_type {};

template <typename T>
struct is_eigen_dense<T, std::enable_if_t<std::is_base_of_v<Eigen::PlainObjectBase<T>, T>>>
    : std::true_type {};

template <typename T>
constexpr bool is_eigen_dense_v = is_eigen_dense<T>::value;

template <typename T>
struct is_eigen_ref : std::false_type {};

template <typename PlainType, int Options, typename StrideType>
struct is_eigen_ref<Eigen::Ref<PlainType, Options, StrideType>> : std::true_type {};

template <typename T>
constexpr bool is_eigen_ref_v = is_eigen_ref<T>::value;

// --- B) Helper: try to load Eigen from ndarray_base ---

template <typename EigenType, typename NdarrayScalar>
bool try_load_eigen_from_ndarray(EigenType& out, const ndarray_base& base) {
    using Scalar = typename EigenType::Scalar;
    constexpr int RowsAtCompile = EigenType::RowsAtCompileTime;
    constexpr int ColsAtCompile = EigenType::ColsAtCompileTime;
    constexpr bool IsVector = EigenType::IsVectorAtCompileTime;

    auto* typed = dynamic_cast<const ::ndarray<NdarrayScalar>*>(&base);
    if (!typed) return false;

    const auto& xarr = typed->data.get_array();
    const NdarrayScalar* ptr = xarr.data();
    auto shape = xarr.shape();
    int ndim = (int)xarr.dimension();

    if constexpr (IsVector) {
        // Accept 1D ndarray for Eigen vectors
        if (ndim != 1) return false;
        int size = (int)shape[0];

        // Check compile-time size constraint
        if constexpr (RowsAtCompile != Eigen::Dynamic) {
            if (size != RowsAtCompile) return false;
        }
        if constexpr (ColsAtCompile != Eigen::Dynamic) {
            if (ColsAtCompile != 1 && size != ColsAtCompile) return false;
        }

        out.resize(size, 1);
        for (int i = 0; i < size; ++i) {
            out(i) = static_cast<Scalar>(static_cast<double>(ptr[i]));
        }
        return true;
    } else {
        // Accept 2D ndarray for Eigen matrices
        if (ndim != 2) return false;
        int rows = (int)shape[0];
        int cols = (int)shape[1];

        // Check compile-time dimension constraints (critical for overload resolution)
        if constexpr (RowsAtCompile != Eigen::Dynamic) {
            if (rows != RowsAtCompile) return false;
        }
        if constexpr (ColsAtCompile != Eigen::Dynamic) {
            if (cols != ColsAtCompile) return false;
        }

        out.resize(rows, cols);
        // Copy element by element with cast (xtensor is row-major)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out(i, j) = static_cast<Scalar>(static_cast<double>(ptr[i * cols + j]));
            }
        }
        return true;
    }
}

// --- C) Helper: cast Eigen to Python ndarray ---

template <typename EigenType>
pkbind::object eigen_to_python(const EigenType& mat) {
    using Scalar = typename EigenType::Scalar;
    constexpr bool IsVector = EigenType::IsVectorAtCompileTime;

    if constexpr (IsVector) {
        int size = (int)mat.size();
        std::vector<Scalar> flat(size);
        for (int i = 0; i < size; ++i) {
            flat[i] = mat(i);
        }
        auto* result = new ::ndarray<Scalar>(flat);
        return pkbind::cast(std::unique_ptr<ndarray_base>(result));
    } else {
        int rows = (int)mat.rows();
        int cols = (int)mat.cols();
        std::vector<std::vector<Scalar>> data(rows, std::vector<Scalar>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = mat(i, j);
            }
        }
        auto* result = new ::ndarray<Scalar>(data);
        return pkbind::cast(std::unique_ptr<ndarray_base>(result));
    }
}

}  // namespace detail

// --- D) type_caster<T> for Eigen dense plain types (Matrix, Vector, Array) ---

template <typename T>
struct type_caster<T, std::enable_if_t<detail::is_eigen_dense_v<T>>> {
    T data;

    template <typename U>
    static object cast(U&& value, return_value_policy, handle) {
        return detail::eigen_to_python(std::forward<U>(value));
    }

    bool load(handle src, bool convert) {
        // Use pkbind's built-in type_caster for ndarray_base
        type_caster<ndarray_base> base_caster;
        if (!base_caster.load(src, false)) return false;
        auto& base = base_caster.value();

        // Try exact scalar match first (use ::qualified names to avoid pkbind:: types)
        using Scalar = typename T::Scalar;
        if constexpr (std::is_same_v<Scalar, double>) {
            if (detail::try_load_eigen_from_ndarray<T, pkpy::float64>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::float32>(data, base)) return true;
        } else if constexpr (std::is_same_v<Scalar, float>) {
            if (detail::try_load_eigen_from_ndarray<T, pkpy::float32>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::float64>(data, base)) return true;
        } else if constexpr (std::is_same_v<Scalar, int>) {
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int32>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int_>(data, base)) return true;
        } else if constexpr (std::is_same_v<Scalar, int64_t>) {
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int_>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int32>(data, base)) return true;
        }

        // Try type-converting matches if convert is allowed
        if (convert) {
            if (detail::try_load_eigen_from_ndarray<T, pkpy::float64>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::float32>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int_>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int32>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int16>(data, base)) return true;
            if (detail::try_load_eigen_from_ndarray<T, pkpy::int8>(data, base)) return true;
        }

        return false;
    }

    T& value() { return data; }

    constexpr inline static bool is_temporary_v = true;
};

// --- E) type_caster for Eigen::Ref<PlainType> ---

template <typename PlainType, int Options, typename StrideType>
struct type_caster<Eigen::Ref<PlainType, Options, StrideType>> {
    using EigenRef = Eigen::Ref<PlainType, Options, StrideType>;
    using PlainNoConst = std::remove_const_t<PlainType>;

    type_caster<PlainNoConst> inner;
    std::unique_ptr<EigenRef> ref_storage;

    template <typename U>
    static object cast(U&& value, return_value_policy policy, handle parent) {
        PlainNoConst plain = value;
        return type_caster<PlainNoConst>::cast(std::move(plain), policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!inner.load(src, convert)) return false;
        ref_storage = std::make_unique<EigenRef>(inner.value());
        return true;
    }

    EigenRef& value() {
        return *ref_storage;
    }

    constexpr inline static bool is_temporary_v = false;
};

}  // namespace pkbind
