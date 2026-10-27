# PyFWI-CUDA

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **This is a modified version of PyFWI.**
> It is a fork of [PyFWI](https://github.com/AmirMardan/PyFWI) by Amir Mardan, Bernard Giroux and
> Gabriel Fabien-Ouellet, based on release 0.1.10 (commit `1434a16`). It has been modified by
> Gabriel Carvalho since October 24, 2026, to add a CUDA backend for NVIDIA GPUs. Every change is
> listed in [CHANGELOG.md](CHANGELOG.md). Like the original, it is distributed under the GNU General
> Public License, version 3 or (at your option) any later version; see [LICENSE](LICENSE).

## What PyFWI is

PyFWI is a Python package for elastic seismic full-waveform inversion (FWI) and time-lapse FWI. It
models 2D elastic wave propagation in the time domain with staggered-grid finite differences
(order 4 or 8 in space), computes gradients with the adjoint-state method, and runs multi-scale
inversions with SciPy optimisers or through PyTorch. Its computational core is written in OpenCL
and driven from Python with PyOpenCL. The original documentation is at
[pyfwi.readthedocs.io](https://pyfwi.readthedocs.io/en/latest/index.html).

![PyFWI structure](docs/pyfwi_structure.png)

## What this fork changes

The goal is to make PyFWI much faster on NVIDIA GeForce RTX cards **without changing its public
API**: a script written for PyFWI 0.1.10 must run unchanged, with the same imports, classes and
results. Only the installation command changes, and the backend (OpenCL or CUDA) is chosen outside
the user's code.

- [Analysis and migration plan](docs/cuda-migration/analysis-and-plan.md): where the time goes in
  0.1.10, why CUDA (through CuPy) helps, the target architecture and the risks.
- [Execution and verification roadmap](docs/cuda-migration/execution-roadmap.md): every step from
  the current state to version 0.4, with the tests and checks of each step.
- [Overview of release 0.3.0](docs/cuda-migration/overview.md): what the fork changed and what it
  gained so far, explained with figures.

**Status:** release 0.4.0, the end of Phase 4 ([gate G4](docs/cuda-migration/gates/G4.md)). On an NVIDIA
GPU the CUDA backend runs kernels written for CUDA, in the `fixed` numeric mode that corrects the numerical
bugs of 0.1.10, 17 to 77 times faster than 0.1.10 for the forward modelling and 12 to 32 times for the
gradient ([benchmark](docs/cuda-migration/reports/benchmark-0.4.0.md)), within recorded tolerances of
OpenCL ([parity report](docs/cuda-migration/reports/parity-0.4.0.md)); complete inversions give the same
estimates on both backends ([geophysical validation](docs/cuda-migration/reports/geophysical-validation-0.4.0.dev0.md),
awaiting the review of a geophysicist). The default backend is `auto`: CUDA when CuPy is installed (a `cuda`
extra) and the GPU is NVIDIA; otherwise the OpenCL engine, in the `legacy` mode, with exactly the results of
0.1.10 (the golden masters in `test/golden/` check it). Release 0.4.0 adds opt-in features, all off by default
(see [Tuning](#tuning-the-cuda-backend)): the cost function on the GPU, boundary saving (4 to 11 times less GPU memory), FP16
fields (half the memory; for modelling and gradients, not inversions), the shots in several processes and CUDA
tensors in `PyFWI.torchfwi`; their tolerances are in [CHANGELOG.md](CHANGELOG.md), which lists every change.
Next: the maintenance steps of the roadmap (documentation, merges of the original repository).

## Repository layout

| Path | Content |
| --- | --- |
| `src/PyFWI/` | The PyFWI package (OpenCL kernels in `*.cl`) |
| `src/PyFWI/engine/`, `src/PyFWI/backends/` | The time loops of the propagation, and the backends that run their kernels (OpenCL; CUDA in Phase 2) |
| `example/` | The original examples (notebooks and one script) |
| `test/` | Test suite (`pytest`); hardware-dependent tests skip themselves when the hardware is missing |
| `tools/check_env.py` | Checks that an environment can run PyFWI and the CUDA tool chain |
| `tools/check_install.py` | Checks an installed copy of the package (files, metadata, every module imports) |
| `tools/setup_envs.sh` | Re-creates the reference and development environments |
| `requirements/` | Pinned package versions of each environment |
| `docs/cuda-migration/` | Analysis, roadmap, overview, gate decisions (`gates/`) and reports (`reports/`) |

## Development setup

The project uses four Python environments, created next to the repository (in `../envs/`) by
[uv](https://docs.astral.sh/uv/), without sudo:

- `ref-0.1.10`: Python 3.10 with PyFWI 0.1.10 exactly as published on PyPI and the package versions
  of its original `requirements.txt`. It produces the reference results the fork is compared with.
- `dev`: Python 3.12 with this repository installed in editable mode, CuPy for CUDA 13, PyOpenCL
  and the test tools.
- `cuda12`: the same with CuPy for CUDA 12, to check the CUDA backend with a CUDA 12 runtime.
- `torch`: the same as `dev` with PyTorch built for CUDA 13, for `PyFWI.torchfwi` with tensors on the GPU.

```bash
pipx install uv                                  # once
tools/setup_envs.sh all                          # creates ../envs/ref-0.1.10, dev, cuda12 and torch
../envs/dev/bin/python tools/check_env.py        # checks GPU, CUDA, OpenCL and a PyFWI run
../envs/dev/bin/python -m pytest -m "not slow"   # fast test suite
```

A CUDA backend needs an NVIDIA GPU of the Turing generation or newer (RTX 20 series and up) with a
driver of the 580 series or newer. Without a GPU, `check_env.py --no-gpu` and the test suite still
run, using OpenCL on the CPU (POCL).

## Tests

```bash
../envs/dev/bin/python -m pytest -m "not slow"     # fast suite, run before every commit (a few minutes)
../envs/dev/bin/python -m pytest                   # everything, including the slow characterisation
../envs/ref-0.1.10/bin/python -m pytest            # the same tests against the published PyFWI 0.1.10
```

How to read the results:

- **Markers.** `opencl` and `gpu` tests need the NVIDIA OpenCL platform or a CUDA GPU and skip
  themselves elsewhere; `slow` tests take minutes (inversions, examples against the golden masters,
  the mathematical baseline).
- **Expected failures.** `test/test_known_bugs.py` states the correct behaviour for each bug inherited
  from PyFWI 0.1.10 and marks it `xfail`; a bug fix turns it into a pass, and the marker must then be
  removed. `xfailed` in the summary is therefore normal.
- **Golden masters.** `test/golden/` holds outputs of the published 0.1.10 (16 forward + gradient
  cases, 10 inversions, every example script). `test/test_characterization.py` compares the installed
  PyFWI with them, bit for bit where possible.
- **Examples.** Every script in `example/scripts/` is run by `test/test_examples.py` with a physical
  check, and must keep working.

## Installing the fork

The fork is distributed as `pyfwi-cuda` and imported as `PyFWI`, like the original. It is not on
PyPI yet; install it from a checkout of this repository:

```bash
python3 -m pip install .                  # PyFWI with its OpenCL engine (Python 3.10 or newer)
python3 -m pip install ".[segy]"          # also reads SEG-Y files (segyio)
python3 -m pip install ".[torch]"         # also PyFWI.torchfwi, gradients through PyTorch
python3 -m pip install ".[cuda13]"        # also the CUDA backend, NVIDIA driver 580 or newer
python3 -m pip install ".[cuda12]"        # the CUDA backend with CUDA 12, NVIDIA driver 525 or newer
python3 tools/check_install.py            # checks the installed copy (run it outside src/)
```

Optional dependencies (extras):

| Extra | Installs | Needed by |
| --- | --- | --- |
| `segy` | `segyio` | `seismic_io.read_segy` and the Marmousi model of `model_dataset` |
| `torch` | `torch` | `PyFWI.torchfwi` and `PyFWI.torch` |
| `cuda13` | `cupy-cuda13x` with the CUDA 13 runtime and NVRTC from PyPI | the CUDA backend (driver 580 or newer) |
| `cuda12` | `cupy-cuda12x` with the CUDA 12 runtime and NVRTC from PyPI | the CUDA backend (driver 525 or newer) |

The CUDA extras need no CUDA Toolkit on the system, only the NVIDIA driver. Install one of them,
not both.

The original PyFWI (`python3 -m pip install PyFWI`) and the fork both provide the `PyFWI` package,
so they cannot be installed in the same environment: uninstall one before installing the other.

## What is running

`PyFWI.info()` says which PyFWI runs and on what: the fork and its git commit (when it runs from a
checkout), the backend (for example `cuda device 0 (auto, the default)`) and the numeric mode (`legacy`:
the results of PyFWI 0.1.10; `fixed`), the OpenCL device a propagator would use, the OpenCL platforms
and the versions of the libraries. Attach it to any bug report.

```python
import PyFWI
print(PyFWI.info())                        # inpa-like arguments: PyFWI.info(platform=1, device=0)
```

The same summary goes to the `"PyFWI"` logger, at level INFO, at the first forward modelling of
the process; Python shows it with `logging.basicConfig(level=logging.INFO)`.

## Choosing the backend

The backend that runs the kernels is chosen by `inpa['backend']`, else by the environment variable
`PYFWI_BACKEND`, else it is `auto`:

| Value | Meaning |
| --- | --- |
| `auto` | CUDA when the OpenCL device that would be used is an NVIDIA GPU (or when there is no OpenCL device) and CUDA can run, otherwise OpenCL (default since 0.3.0) |
| `opencl` | The OpenCL engine of PyFWI 0.1.10 (the default of 0.2.0) |
| `cuda` | CUDA when it can run here, otherwise OpenCL with the reason in the log |
| `cuda:strict` | CUDA, or an error that says why it cannot run |

CUDA cannot run without CuPy, so a plain installation (no `cuda13` or `cuda12` extra) keeps the OpenCL
engine and the results of PyFWI 0.1.10. With one of those extras on an NVIDIA GPU, `auto` runs CUDA, in
the `fixed` numeric mode; `PYFWI_NUMERICS=legacy` keeps the results of 0.1.10 on CUDA (with the slower
parity kernels), and `PYFWI_BACKEND=opencl` keeps the OpenCL engine.

The numeric mode is chosen the same way, with `inpa['numerics']` or `PYFWI_NUMERICS`: `legacy` gives the
results of PyFWI 0.1.10 (default on OpenCL), `fixed` corrects its numerical bugs (default on CUDA;
see [CHANGELOG.md](CHANGELOG.md) for what changes and by how much). The largest change: `fixed` injects
the source at every internal time step instead of once per recorded sample, which removes a spurious
grid-scale noise that 0.1.10 creates at grid spacings of a few metres.

CUDA needs CuPy for the CUDA version of the driver: the `cuda13` extra for drivers of the 580 series
or newer, `cuda12` for older ones (see [Installing the fork](#installing-the-fork)). In the `fixed` mode
the CUDA backend runs kernels written for CUDA (Phase 3), in the `legacy` mode the kernel files of
OpenCL compiled for CUDA (its results agree with OpenCL to about 1e-6 for seismograms and 2e-5 for
gradients).

## Tuning the CUDA backend

These environment variables tune the CUDA backend; none is needed. The results do not depend on them,
except where the table says so.

| Variable | Default | Effect |
| --- | --- | --- |
| `PYFWI_CUDA_KERNELS` | `native` | In the `fixed` mode, `parity` runs the kernel files of OpenCL compiled for CUDA instead of the kernels written for CUDA (slower; results differ by rounding) |
| `PYFWI_CUDA_BATCH` | as many shots as fit in 2^20 grid cells, in balanced batches | Shots run at once by the same launches (the gradients of a batch are added at the end: they differ by rounding with the batch size) |
| `PYFWI_CUDA_GRAPHS` | `1` | `0` launches the kernels of every time step one by one instead of replaying CUDA graphs |
| `PYFWI_CUDA_CHECKPOINT_BUDGET` | 80 % of the free GPU memory | Bytes of checkpoints (the forward field stored for the gradient) kept on the GPU; the others stay in host memory |
| `PYFWI_CUDA_BLOCK` | 32 columns by 8, 4 or 2 rows, to fill the GPU | The CUDA block of every launch, `<columns>x<rows>` (for example `64x4`) |
| `PYFWI_BOUNDARY_SAVING` | `0` | `1` stores, at every step, only a band of cells around the model instead of checkpoints of the whole field (`chpr`), and rebuilds the forward field exactly for the gradient: 5 to 12 times less memory; the `fixed` mode on CUDA only |
| `PYFWI_FP16` | `0` | `1` stores the fields in half precision (scaled): half the memory, 1.3 to 1.4 times faster on large grids; seismograms within about 45 dB of float32. For modelling and gradients, not for inversions (see the CHANGELOG); the `fixed` mode on CUDA only |
| `PYFWI_CUDA_DEVICES` | unset | A list of CUDA devices (`0,1`): the shots are shared between one process per device, with the same results as one process |
| `PYFWI_CUDA_COST` | `0` | `1` runs the L2 cost function, its multi-scale filters and the smoothing of the gradient on the GPU (an inversion step faster when the propagation is fast; results within 1e-6, the filters in float32) |

## Citing PyFWI

If you use this software, please cite the original PyFWI paper:

```
@article{mardan2023pyfwi,
  title = {PyFWI: {A Python} package for full-waveform inversion and reservoir monitoring},
  author = {Mardan, Amir and Giroux, Bernard and Fabien-Ouellet, Gabriel},
  journal = {SoftwareX},
  volume = {22},
  pages = {101384},
  year = {2023},
  publisher = {Elsevier},
  doi = {10.1016/j.softx.2023.101384}
}
```

## License

GNU General Public License v3.0 or later, as the original PyFWI. The full text is in
[LICENSE](LICENSE), unchanged from the original repository.
