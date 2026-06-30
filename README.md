# SPC (Sampling-based Predictive Control)

A high-performance C++ backend for Policy-Guided Sampling-based Predictive Control.

This project implements a highly optimized Cross-Entropy Method (CEM) planner designed for real-time robotic control. By leveraging native MuJoCo C APIs, thread-local simulation states, and OpenMP multi-threading, this library completely bypasses Python's GIL and thread-pool overhead.

## Features

- **C++ Core Engine:** Parallel physics rollouts utilizing the native MuJoCo C API.
- **Native Policy Inference:** Fast evaluation of ONNX policies directly inside the C++ planner.
- **Python Bindings:** Exposed seamlessly to Python (`spc_py`) via `pybind11` for interactive orchestration.
- **Zero-Boilerplate Tasks:** Dynamically pass parameters (e.g. reward weights, target names) from Python to C++ tasks without recompiling.

## Setup

The project uses `uv` for fast package management and `scikit-build-core` for the C++ backend. You'll need a C++20 compiler and CMake installed.

```bash
# Clone the repository
git clone <repo-url> spc && cd spc

# Create a virtual environment
uv venv

# Install the Python package and compile the C++ backend
uv pip install -e .
```

## Running Examples

Examples demonstrate the interactive planner loop for various tasks. They launch a native MuJoCo viewer where you can see the optimizer at work.

```bash
uv run examples/run_particle.py
uv run examples/run_franka_push.py
uv run examples/run_g1_navigation.py
```

## Development

To contribute to the project, install the development dependencies which include automated formatting tools (`ruff` for Python, `clang-format` for C++).

```bash
# Install development tools
make install-dev

# Format all Python and C++ code
make format
```

> **Note**: CMake is configured to automatically discover new source files. When you add a new `.cc` file to the `src/` directory, simply run `uv pip install -e .` and it will automatically be compiled into the core library!
