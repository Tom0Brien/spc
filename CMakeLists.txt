cmake_minimum_required(VERSION 3.20)
project(spc VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)

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
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Link against the Python `mujoco` package instead of building from source
find_package(Python REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import mujoco; import os; print(os.path.dirname(mujoco.__file__))"
    OUTPUT_VARIABLE MUJOCO_PY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import mujoco; import os; print(os.path.dirname(mujoco.HEADERS_DIR))"
    OUTPUT_VARIABLE MUJOCO_HEADERS_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(GLOB MUJOCO_PY_LIB "${MUJOCO_PY_DIR}/libmujoco.so*")
list(GET MUJOCO_PY_LIB 0 MUJOCO_PY_LIB_PATH)

add_library(mujoco::mujoco SHARED IMPORTED)
set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_PY_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_HEADERS_DIR}"
)
# Ensure scikit-build-core installs this library if needed, though it's already in the environment
install(FILES ${MUJOCO_PY_LIB_PATH} DESTINATION lib)

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
file(GLOB_RECURSE SPC_CORE_SOURCES CONFIGURE_DEPENDS "src/*.cc")
list(FILTER SPC_CORE_SOURCES EXCLUDE REGEX "src/bindings/.*")

add_library(spc_core SHARED ${SPC_CORE_SOURCES})
target_include_directories(spc_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(spc_core PUBLIC OpenMP::OpenMP_CXX mujoco::mujoco onnxruntime)
set_target_properties(spc_core PROPERTIES 
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# -----------------------------------------------------------------------------
# Python Bindings
# -----------------------------------------------------------------------------
pybind11_add_module(spc_py src/bindings/pybind.cc)
target_link_libraries(spc_py PRIVATE spc_core)
set_target_properties(spc_py PROPERTIES 
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

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
