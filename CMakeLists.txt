cmake_minimum_required(VERSION 3.15...3.26)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME "pocket_numpy")
endif()
if(NOT DEFINED PROJECT_VERSION)
  set(PROJECT_VERSION "dev")
endif()

# https://scikit-build-core.readthedocs.io/en/latest/cmakelists.html#accessing-information
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# std::visit requires macOS 10.14+
if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "10.14")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum macOS deployment version" FORCE)
endif()

# set(CMAKE_BUILD_TYPE "Debug")
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    message(STATUS "Set build type to default: ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Your build type: ${CMAKE_BUILD_TYPE}")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(PK_BUILD_MODULE_MSGPACK ON)
set(PK_BUILD_MODULE_LZ4 ON)
set(PK_BUILD_STATIC_LIB ON)
add_subdirectory(pocketpy ${CMAKE_CURRENT_BINARY_DIR}/__pocketpy__)

include_directories(SYSTEM pocketpy/include include)

# Control xtensor warnings (OFF by default)
option(SHOW_XTENSOR_WARNINGS "Show warnings from xtensor" OFF)

if(MSVC)
    add_compile_options(/wd4018 /wd4172 /wd4819)
    add_compile_options(/bigobj)
endif()

# Suppress xtensor warnings if SHOW_XTENSOR_WARNINGS is OFF
if(NOT SHOW_XTENSOR_WARNINGS)
    add_compile_definitions(SUPPRESS_XTENSOR_WARNINGS)
    if(MSVC)
        add_compile_options(/wd4244 /wd4267 /wd4018)
    else()
        add_compile_options(-Wno-sign-compare -Wno-conversion -Wno-unused-variable -Wno-unused-parameter)
    endif()
endif()

# Create numpy static library
add_library(numpy STATIC src/numpy.cpp)

# Define PY_DYNAMIC_MODULE for numpy
target_compile_definitions(numpy PRIVATE PY_DYNAMIC_MODULE)

# Link numpy with pocketpy
target_link_libraries(numpy PUBLIC pocketpy)

add_executable(pocketpy_cli src/main.cpp)
set_target_properties(pocketpy_cli PROPERTIES OUTPUT_NAME pocketpy)
set_target_properties(pocketpy_cli PROPERTIES SUFFIX ".exe")
target_link_libraries(pocketpy_cli numpy)
install(TARGETS pocketpy_cli DESTINATION ${PROJECT_NAME})

# https://scikit-build-core.readthedocs.io/en/latest/getting_started.html
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_core MODULE src/pybind.cpp WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _core DESTINATION ${PROJECT_NAME})