cmake_minimum_required(VERSION 3.10)
project(numpy)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

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
add_library(${PROJECT_NAME} STATIC src/numpy.cpp)

# Define PY_DYNAMIC_MODULE for numpy
target_compile_definitions(${PROJECT_NAME} PRIVATE PY_DYNAMIC_MODULE)

# Link numpy with pocketpy
target_link_libraries(${PROJECT_NAME} PUBLIC pocketpy)

add_executable(pocketpy_cli src/main.cpp)
set_target_properties(pocketpy_cli PROPERTIES OUTPUT_NAME pocketpy)
target_link_libraries(pocketpy_cli ${PROJECT_NAME})