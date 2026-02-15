# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pocket_numpy** (repo: xtensor-numpy) is a C++ project that integrates the pocketpy lightweight Python interpreter with numpy-like array functionality powered by xtensor and Eigen. It produces:
- A native CLI (`pocketpy.exe`) that runs Python scripts with numpy support
- Python wheels via pybind11 (package name: `pocket_numpy`)
- A WebAssembly build for browser deployment

## Build & Test Commands

```bash
# Build (installs in editable mode with scikit-build-core)
make build

# Run numpy tests via pocketpy CLI
make test                              # runs: build/pocketpy.exe tests/test_numpy.py

# Run a single test file
build/pocketpy.exe tests/test_numpy.py

# Run tests via pytest (after pip install)
python3 -m pytest tests/

# Install as pip package
make python_install

# Build WASM
make build_wasm

# Serve WASM demo locally
make serve_wasm
```

## Architecture

### Core Type System (`include/numpy.hpp`)

Defines C++ type aliases (int8–64, uint8–64, float32, float64, bool_, complex64/128) and `dtype_traits<T>` template for runtime dtype identification. Contains the `ndarray<T>` template backed by `xt::xarray<T>`.

### Polymorphic ndarray (`include/ndarray_binding.hpp`)

`ndarray_base` is an abstract base class with ~80+ virtual methods (shape, reductions, slicing, arithmetic, sorting, etc.). `ndarray<T>` is the concrete template implementation. This virtual dispatch pattern enables a single Python-facing interface that works across all numeric types via `std::unique_ptr<ndarray_base>`.

### Module Bindings (`src/numpy.cpp`)

Registers the `numpy` module into pocketpy's runtime. Binds all ndarray methods, array creation functions (ones, zeros, full, arange, linspace, identity), random number generation, and dtype attributes. This is the largest source file.

### RDP Bindings (`src/pybind.cpp` + `src/rdp.hpp`)

Ramer-Douglas-Peucker polyline simplification exposed via pybind11 as the `_core` module. Supports 2D and 3D point arrays using Eigen matrix types.

### CLI Entry Point (`src/main.cpp`)

Initializes pocketpy, registers the numpy module via `py_module_initialize()`, then executes a Python script or enters REPL mode.

### Dependencies

- **pocketpy** — git submodule at `pocketpy/`
- **xtensor, xtl, Eigen** — vendored headers in `include/`
- **pybind11** — required via pip/CMake for Python wheel builds

### Build Targets (CMake)

- `numpy` — static library (numpy.cpp + pocketpy)
- `pocketpy_cli` — CLI executable linked against numpy
- `_core` — pybind11 module for RDP bindings

## Git Policy

- **NEVER use `git commit --amend`**. Always create a new commit instead. Amending rewrites history and can destroy previous work, especially after a failed pre-commit hook where `--amend` would silently modify the wrong commit.

## Key Conventions

- C++17 required; C11 for pocketpy C sources
- Default build type is Release with `-O3`; debug uses `-O0 -ggdb`
- xtensor warnings are suppressed by default (`SHOW_XTENSOR_WARNINGS=OFF`)
- Tests are standalone Python scripts (not pytest-based) so they can also run under pocketpy and WASM
- macOS deployment target is 10.14+ (required for `std::visit`)
