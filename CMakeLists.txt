cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
project(spc VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
include(FetchContent)

# Fetch Pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Fetch GoogleTest
enable_testing()
find_package(GTest)
if(NOT GTEST_FOUND)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Fetch MuJoCo properly (Build from source for maximum portability)
FetchContent_Declare(
  mujoco
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
  GIT_TAG        3.3.7
)
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)

# Do not apply global flags that mess with third-party libraries

FetchContent_MakeAvailable(mujoco)

# Fetch ONNX Runtime (pre-built binary for Linux x64)
FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
)
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include
)

# -----------------------------------------------------------------------------
# Core C++ Library
# -----------------------------------------------------------------------------
add_library(spc_core SHARED
    src/utils/spline.cc
    src/algs/optimizer.cc
    src/algs/cem.cc
    src/tasks/particle.cc
    src/tasks/franka_push.cc
    src/tasks/g1_navigation.cc
    src/core/onnx_policy.cc
    src/core/task_factory.cc
)
target_include_directories(spc_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(spc_core PUBLIC OpenMP::OpenMP_CXX mujoco::mujoco onnxruntime)
target_compile_options(spc_core PRIVATE -O3 -march=native -flto)
target_link_options(spc_core PRIVATE -O3 -march=native -flto)
set_target_properties(spc_core PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib")

# -----------------------------------------------------------------------------
# Python Bindings
# -----------------------------------------------------------------------------
pybind11_add_module(spc_py src/bindings/pybind.cc)
target_link_libraries(spc_py PRIVATE spc_core)
target_compile_options(spc_py PRIVATE -O3 -march=native -flto)
target_link_options(spc_py PRIVATE -O3 -march=native -flto)
set_target_properties(spc_py PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib")

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
if(NOT SKBUILD)
enable_testing()

add_executable(test_cem tests/test_cem.cc)
target_link_libraries(test_cem PRIVATE spc_core GTest::gtest_main)
add_test(NAME CEMTest COMMAND test_cem)

add_executable(test_spline tests/test_spline.cc)
target_link_libraries(test_spline PRIVATE spc_core GTest::gtest_main)
add_test(NAME SplineTest COMMAND test_spline)

add_executable(bench_mj_step tests/bench_mj_step.cc)
target_link_libraries(bench_mj_step PRIVATE spc_core)

add_executable(test_particle tests/test_particle.cc)
target_link_libraries(test_particle PRIVATE spc_core GTest::gtest_main)
target_compile_definitions(test_particle PRIVATE MODELS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/models)
add_test(NAME ParticleTest COMMAND test_particle)

add_executable(test_onnx tests/test_onnx.cc)
target_link_libraries(test_onnx PRIVATE spc_core gtest_main onnxruntime)
add_test(NAME ONNXTest COMMAND test_onnx)

include(GoogleTest)
gtest_discover_tests(test_cem)
gtest_discover_tests(test_spline)
gtest_discover_tests(test_particle)
gtest_discover_tests(test_onnx)
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install(TARGETS spc_core spc_py DESTINATION .)
install(DIRECTORY ${onnxruntime_SOURCE_DIR}/lib/ DESTINATION lib)
