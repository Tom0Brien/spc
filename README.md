# SPC (Sampling-based Predictive Control)

A high-performance C++ backend for Policy-Guided Sampling-based Predictive Control (SPC).

## Architecture

This project is a native C++ reimplementation of the core optimization loop (e.g. Cross-Entropy Method), designed to achieve extreme performance for real-time robotic control. By leveraging native MuJoCo CPU execution, thread-local simulation states, and OpenMP multi-threading, this library completely bypasses Python's GIL and thread-pool overhead.

### Key Components

- **C++ Core Engine:** Parallel physics rollouts utilizing the MuJoCo C API.
- **Native Policy Inference:** The base RL policy is evaluated natively in C++ via ONNX Runtime to avoid GPU latency during planning.
- **Python Bindings:** Exposed seamlessly to Python via `pybind11` for high-level environment orchestration and training.

## Building

Requires a C++20 compiler and CMake.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Testing

Uses GoogleTest framework.

```bash
make test_cem
ctest
```
