# SPC (Sampling-based Predictive Control)

A high-performance C++ backend for Policy-Guided Sampling-based Predictive Control.

## Requirements

- **Linux x86-64.** The build downloads a prebuilt ONNX Runtime for `linux-x64`, so macOS and Windows are not supported out of the box.
- **A C++20 compiler** (GCC or Clang).
- **CMake** ≥ 3.20.
- **OpenMP** (e.g. `libgomp` / `libomp`) — required for the parallel rollouts.
- **[`uv`](https://docs.astral.sh/uv/)** for package management. Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Internet access at build time** — CMake fetches pybind11, GoogleTest, and ONNX Runtime.

## Setup

The project uses `uv` for fast package management and `scikit-build-core` for the C++ backend.

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
uv run examples/run_g1_soccer.py
```

## Development

To contribute to the project, install the development dependencies which include automated formatting tools (`ruff` for Python, `clang-format` for C++).

```bash
# Install development tools
make install-dev

# Format all Python and C++ code
make format
```

> **Note**: CMake is configured to automatically discover new source files (`GLOB_RECURSE`), so adding a new `.cc` file under `src/` requires no CMake edits. However, a plain `uv pip install -e .` will **not** rebuild after you edit C++ sources — it reports `Audited 1 package` and exits. To force a recompile, run:
>
> ```bash
> make rebuild
> ```
