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

**Status:** Phase 1 (host-side fixes, no change to the numerical results). The package code under
`src/PyFWI/` differs from PyFWI 0.1.10 only by the changes listed in [CHANGELOG.md](CHANGELOG.md);
the golden masters in `test/golden/` check that the results stay the same.

## Repository layout

| Path | Content |
| --- | --- |
| `src/PyFWI/` | The PyFWI package (OpenCL kernels in `*.cl`) |
| `example/` | The original examples (notebooks and one script) |
| `test/` | Test suite (`pytest`); hardware-dependent tests skip themselves when the hardware is missing |
| `tools/check_env.py` | Checks that an environment can run PyFWI and the CUDA tool chain |
| `tools/check_install.py` | Checks an installed copy of the package (files, metadata, every module imports) |
| `tools/setup_envs.sh` | Re-creates the reference and development environments |
| `requirements/` | Pinned package versions of each environment |
| `docs/cuda-migration/` | Analysis, roadmap and, later, gate reports |

## Development setup

The project uses two Python environments, created next to the repository (in `../envs/`) by
[uv](https://docs.astral.sh/uv/), without sudo:

- `ref-0.1.10`: Python 3.10 with PyFWI 0.1.10 exactly as published on PyPI and the package versions
  of its original `requirements.txt`. It produces the reference results the fork is compared with.
- `dev`: Python 3.12 with this repository installed in editable mode, CuPy for CUDA 13, PyOpenCL
  and the test tools.

```bash
pipx install uv                                  # once
tools/setup_envs.sh all                          # creates ../envs/ref-0.1.10 and ../envs/dev
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
python3 tools/check_install.py            # checks the installed copy (run it outside src/)
```

Optional dependencies (extras):

| Extra | Installs | Needed by |
| --- | --- | --- |
| `segy` | `segyio` | `seismic_io.read_segy` and the Marmousi model of `model_dataset` |
| `torch` | `torch` | `PyFWI.torchfwi` and `PyFWI.torch` |

The original PyFWI (`python3 -m pip install PyFWI`) and the fork both provide the `PyFWI` package,
so they cannot be installed in the same environment: uninstall one before installing the other.

## What is running

`PyFWI.info()` says which PyFWI runs and on what: the fork and its git commit (when it runs from a
checkout), the backend (`opencl`) and numeric mode (`legacy`: the results of PyFWI 0.1.10), the
OpenCL device a propagator would use, the OpenCL platforms and the versions of the libraries. Attach
it to any bug report.

```python
import PyFWI
print(PyFWI.info())                        # inpa-like arguments: PyFWI.info(platform=1, device=0)
```

The same summary goes to the `"PyFWI"` logger, at level INFO, at the first forward modelling of
the process; Python shows it with `logging.basicConfig(level=logging.INFO)`.

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
