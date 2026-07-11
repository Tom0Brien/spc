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
# Particle reaching and Franka push
uv run examples/particle.py
uv run examples/franka_push.py

# Unitree G1 humanoid: navigation, pass, and shoot
uv run examples/g1_navigation.py
uv run examples/g1_pass.py
uv run examples/g1_pass_augmented.py

# Booster T1 humanoid: navigation, pass, and shoot
uv run examples/t1_navigation.py
uv run examples/t1_pass.py
uv run examples/t1_pass_augmented.py
uv run examples/t1_shoot.py
uv run examples/t1_shoot_augmented.py
```

Pass `--record` to save an mp4 of the viewer to `recordings/` (requires `ffmpeg` on the `PATH`):

```bash
uv run examples/g1_pass.py --record
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
>
> To remove the installed extension and local caches for a clean slate (run `make build` afterwards to recompile from scratch):
>
> ```bash
> make clean
> ```

## Testing

The C++ unit tests use GoogleTest and are driven by CTest. They are only built
from a standalone CMake configuration (they are skipped during the
`scikit-build-core` wheel build), so configure a separate build directory:

```bash
# Configure and build the tests (and the rest of the C++ backend)
uv run cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
uv run cmake --build build -j 8

# Run the full test suite
cd build && uv run ctest --output-on-failure
```

The suite covers the CEM optimizer (convergence, control bounds), the
control spline interpolation, the `Particle` task cost, the task
factory, and both policy backends — including a check that the hand-rolled
`MLPPolicy` inference matches the ONNX reference.
