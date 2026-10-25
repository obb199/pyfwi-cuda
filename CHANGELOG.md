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
  `test/oob_guard.py`, NaN guard bands that make out-of-bounds buffer reads deterministic and
  detectable on any OpenCL device.
- GitHub Actions workflow `ci.yml` (CPU job: two lint passes, GPU-less environment check, test
  suite) with pinned dependencies in `requirements/ci-cpu.txt`, and a `[tool.ruff]` configuration.

### Removed

- The inherited workflows `pr.yml` (tested Python 3.8–3.10 with the old requirements) and
  `publish.yml` (uploaded to PyPI under the name of the original package).

### Changed

- None in the package: `src/PyFWI/` is identical to PyFWI 0.1.10.

### Numerical changes

- None.
