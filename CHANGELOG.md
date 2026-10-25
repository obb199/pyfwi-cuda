# Changelog

All notable changes of the PyFWI-CUDA fork are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions are those of the fork; the
starting point is PyFWI 0.1.10 (commit `1434a16`).

Numerical changes are always flagged: if an entry changes the results of the package, it says by
how much and in which numeric mode.

## [Unreleased]

### Added

- `tools/check_env.py`: checks versions, OpenCL devices, the CUDA tool chain (NVRTC, CUDA Graphs,
  cuFFT, cupyx) and runs forward + gradient of PyFWI on every OpenCL device; `--no-gpu` for
  machines without NVIDIA.
- `tools/setup_envs.sh` and `requirements/`: reproducible reference (`ref-0.1.10`) and
  development (`dev`) environments.
- Test suite for the tools, the environment files and the documentation.
- `docs/cuda-migration/`: analysis and migration plan, and the execution and verification roadmap.
- Fork README with the GPLv3 modification notice, and this changelog.
- `test/test_known_bugs.py`: the known bugs of 0.1.10 as strict expected failures, and
  `test/oob_guard.py`, guard bands around OpenCL buffers that make out-of-bounds reads and
  writes deterministic and detectable on any OpenCL device.
- `test/test_reproducibility.py` and race-free geometry helpers: under zero guard bands and a
  race-free acquisition, PyFWI 0.1.10 is deterministic and NVIDIA OpenCL and POCL agree bit for
  bit, the basis of the golden masters.
- `example/scripts/`: user-facing example scripts that use only the public API, each with a
  `main(size, show)` function, run by `test/test_examples.py` with a physical check
  (`forward_surface.py`: forward modelling with a surface acquisition; `forward_acquisitions.py`:
  crosswell, surface and combined geometries, and how traces map to receivers; `gradient.py`: the
  adjoint-state gradient, the effect of `chpr` and the two parameterisations; `fwi.py`: multi-scale
  FWI with L-BFGS, and why the source amplitude matters with the absolute tolerances of 0.1.10;
  `tlfwi.py`: cascaded time-lapse FWI; `torch_gradient.py`: the gradient through PyTorch with
  `torchfwi.Fwi`, equal to PyFWI's own; `models.py`: the synthetic model library).
- Golden masters of PyFWI 0.1.10 (`test/golden/`, 16 pairwise cases) with their generator
  (`tools/make_golden.py`), and characterisation tests that compare the installed PyFWI with them
  (`test/test_characterization.py`).
- Mathematical baseline of 0.1.10 (`tools/math_baseline.py`, `test/golden/math_baseline.json`):
  gradient against finite differences, analytic solution, boundary reflection, residual energy; and
  `test/test_math.py`, which requires later versions not to be worse.
- GitHub Actions workflow `ci.yml` (CPU job: two lint passes, GPU-less environment check, test
  suite) with pinned dependencies in `requirements/ci-cpu.txt`, and a `[tool.ruff]` configuration.

### Removed

- The inherited workflows `pr.yml` (tested Python 3.8–3.10 with the old requirements) and
  `publish.yml` (uploaded to PyPI under the name of the original package).

### Changed

- `PyFWI.seismic_io`, `PyFWI.model_dataset`, `PyFWI.rock_physics`, `PyFWI.seiplot` and
  `PyFWI.fwi_tools` import with NumPy 2: unused imports of `numpy.lib.function_base` and
  `numpy.core` were removed, and `gaussian_filter` is imported from `scipy.ndimage` (public module)
  instead of the deprecated `scipy.ndimage.filters`.
- `fwi_tools.pml_delta_calculation` computes the PML damping profile in float64 explicitly. Under
  NumPy 2 (NEP 50 type promotion) part of it was computed in float32, which changed the last bit of
  the profile and moved seismograms by up to 6e-7 and gradients by up to 3e-5 (relative L2). Results
  are now bit-for-bit those of PyFWI 0.1.10 with any NumPy version.
- `fwi_tools.CostFunction` (L2) accumulates the misfit in float64 explicitly, as NumPy 1.x did
  implicitly; under NumPy 2 it was a float32 value (same value, different type).

### Numerical changes

- None.
