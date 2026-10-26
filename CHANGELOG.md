# Changelog

All notable changes of the PyFWI-CUDA fork are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions are those of the fork; the
starting point is PyFWI 0.1.10 (commit `1434a16`).

Numerical changes are always flagged: if an entry changes the results of the package, it says by
how much and in which numeric mode.

## [Unreleased]

### Added

- Choice of the backend (step 2.2, `PyFWI.backends.selection`): `inpa['backend']`, else the
  `PYFWI_BACKEND` environment variable, else `opencl`. `cuda` falls back to OpenCL with the reason in
  the log, `cuda:strict` fails with the reason, `auto` takes CUDA only for an NVIDIA GPU chosen as
  OpenCL device (matched by PCI bus ID). Until the CUDA backend exists (step 2.3) everything runs on
  OpenCL. `W.backend_selection` and `PyFWI.info()` say what was chosen and why.
- CUDA backend (step 2.3, `PyFWI.backends.cuda`, needs CuPy): the same kernel files as OpenCL,
  compiled as CUDA with NVRTC after the portability header `PyFWI/kernels/opencl_on_cuda.h`, in the
  `legacy` numeric mode; a guard at the start of every kernel runs exactly the work items of the
  OpenCL launch; buffers with margins as in step 1.7. Chosen with `PYFWI_BACKEND=cuda` (or
  `inpa['backend']`); OpenCL stays the default. With it `W.ctx` and `W.queue` are None.

### Changed

- Every OpenCL call of `PyFWI.wave_propagation` goes through a backend object (step 2.1):
  `PyFWI.backends.opencl.OpenCLBackend` (allocate with margins, copy, compile, set arguments,
  launch, synchronise), the interface that the CUDA backend will also implement. No change in what
  runs; `W.ctx`, `W.queue`, `W.prg`, `W.knl` and the buffer attributes are unchanged.
- The time loops move to `PyFWI.engine.propagation.PropagationEngine` (step 2.1): device buffers
  with margins, kernels and their arguments, the forward loop, the checkpoints and the gradient loop,
  on any backend. `WavePropagator` prepares the inputs and the outputs and shows the engine's
  attributes as its own (`W.vx_b`, `W.knl`, `W.margins_changed()`...). `MARGIN_FILL` and
  `MIN_MARGIN_BYTES` are now in `PyFWI.engine.propagation`; the checkpoint storage is
  `W._engine.checkpoints` (`W.W` is unchanged). Same results bit for bit.

### Numerical changes

- None for the default backend (OpenCL). The CUDA backend agrees with OpenCL on the same GPU to a
  relative L2 of 1.1e-6 (seismograms) and 1.9e-5 (gradients) on the propagation cases of the golden
  masters: the two compilers emit different instruction sequences.

## [0.1.11] - 2026-10-25

First release of the fork (Phase 1 of the roadmap, gate [G1](docs/cuda-migration/gates/G1.md)): PyFWI
0.1.10 on its OpenCL engine, with exactly its results, installable with current NumPy and SciPy on
Python 3.10 to 3.13, and faster in every scenario of the benchmark (forward / gradient against
0.1.10: 3.5× / 2.5× small, 2.6× / 1.5× medium, 1.5× / 1.14× large;
[report](docs/cuda-migration/reports/benchmark-0.1.11.md)).

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
- `tools/check_install.py`: checks an installed copy of the package (distribution metadata,
  kernels and data files present, every module imports, the `dupuy` model reads its data); the CI
  runs it on the wheel installed in a fresh environment.
- Optional dependencies (extras) `segy` (`segyio`) and `torch` (`torch`).
- `PyFWI.info()` (step 1.8): which PyFWI runs (the fork, its git commit when it runs from a
  checkout), its backend and numeric mode, the OpenCL device a propagator would use, the OpenCL
  platforms and the library versions; works without a GPU or OpenCL. The same summary goes to the
  `"PyFWI"` logger (INFO) at the first forward modelling of the process. `import PyFWI` still loads
  nothing else.
- The test suite reports OpenCL tests as skipped, with a warning annotation on GitHub, when PoCL
  cannot compile kernels for the CPU of the machine (PoCL 3.0 with LLVM 14 does not know the AMD
  EPYC 9V45 of some GitHub-hosted runners); any other failure still fails.

### Removed

- The inherited workflows `pr.yml` (tested Python 3.8–3.10 with the old requirements) and
  `publish.yml` (uploaded to PyPI under the name of the original package).
- `setup.py` and `requirements.txt`: the package metadata is in `pyproject.toml`, and the pinned
  0.1.10 stack is in `requirements/ref-0.1.10.txt`.

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
- `PyFWI.fwi`, `PyFWI.tl_fwi` and `PyFWI.optimization` import and run with current SciPy: the new
  module `PyFWI._scipy_compat` provides `MemoizeJac` (removed from SciPy's public names in 1.8)
  and an `fmin_l_bfgs_b` that drops `iprint` and `disp` when the installed SciPy no longer accepts
  them (they only control printing). With such a SciPy the optimiser no longer prints its progress.
- `segyio` is now an optional dependency: `PyFWI.seismic_io` and `PyFWI.model_dataset` import it
  only inside the functions that read SEG-Y files (`seismic_io.read_segy` and the Marmousi model of
  `model_dataset`). Without it the rest of PyFWI works, and reading SEG-Y raises an `ImportError`
  that says how to install it. In 0.1.10 importing either module failed without `segyio`, which was
  never a declared dependency.
- Packaging (`pyproject.toml`): the distribution is named `pyfwi-cuda` (the import name stays
  `PyFWI`), version `0.1.11.dev0`, Python 3.10 or newer. `pyopencl` is now a declared dependency
  (0.1.10 listed it only as a build requirement, so `pip install PyFWI` did not install the
  engine); `setuptools`, `wheel` and `datetime` are no longer runtime dependencies (`datetime` on
  PyPI is an unrelated package). The wheel now contains `PyFWI.torch` and `PyFWI/data/` (the
  `dupuy` model failed with `FileNotFoundError` with the wheel of 0.1.10), and the licence file.
- The version is written only in `src/PyFWI/version.py`; the package metadata and the Sphinx
  configuration read it from there.
- The OpenCL device is chosen by PyFWI itself (`PyFWI._opencl_device.choose_devices`) instead of
  `pyopencl.create_some_context()`, with the same rules: `inpa['platform']` and `inpa['device']`,
  otherwise `PYOPENCL_CTX` (index or part of the name, as in PyOpenCL), otherwise the first device of
  the first platform. Fixed: `inpa['device']` failed on a platform with a single device with
  PyOpenCL 2022; the range of `inpa['device']` was checked against the devices of the first
  platform; `inpa['platform']` without `inpa['device']` was ignored (it now selects device 0 of
  that platform). PyFWI no longer asks in the terminal which device to use when nothing is
  configured: it takes the default answer of that question (first platform, first device). A
  device that does not exist raises a `ValueError` that lists the available ones. As before, a
  choice made with `inpa['device']` is also written to `PYOPENCL_CTX`.
- Kernel launches (step 1.3): `WavePropagator` retrieves each OpenCL kernel once (`W.knl`), sets
  the arguments that stay the same during a time loop once per forward modelling or gradient, and
  launches the time-loop kernels with `pyopencl.enqueue_nd_range_kernel`. Same kernels, order and
  arguments, so the same results bit for bit; no more `RepeatedKernelRetrieval` warnings. Small
  scenario: forward 3.0x and gradient 2.4x faster than 0.1.10; PyOpenCL 2022 and 2026 now run at the
  same speed ([report](docs/cuda-migration/reports/step-1.3-kernel-launches.md)).
- `tools/benchmark.py` and `tools/opencl_profile.py` also count the launches made with
  `pyopencl.enqueue_nd_range_kernel`.
- Residual on the device (step 1.4): the gradient copies the residual of each shot to the device
  once (`WavePropagator.residual_to_device`) and copies each time sample on the device before the
  adjoint injection (`residual_sample`), instead of five blocking host-to-device copies before each
  injection (750 per shot in the example of the documentation; now 5). The seismograms of a shot are
  read with non-blocking copies and one wait. Same values, same results bit for bit.
  `make_residual` is kept, unused. The copy counter of `tools/benchmark.py` no longer counts copies
  between two device buffers as blocking.
- Contiguous checkpoints (step 1.5): the checkpoints of the forward field are stored as
  `(ns, nchp, tnz, tnx)` per component, so that each field is copied straight from and to the
  device without blocking (0.1.10: `(tnz, tnx, ns, nchp)`, a temporary array and a blocking copy
  for each of them). `W.W` becomes a property that shows them in the 0.1.10 layout, as views: same
  shape, dtype, axis order and values; writing into it changes the checkpoints, and assigning it
  replaces them, as before. Replacing an entry of the dictionary it returns no longer has an effect.
  Same results bit for bit; peak memory unchanged.
- Low-pass filter of the multi-scale inversion (step 1.6): `fwi_tools.lowpass` and `adj_lowpass`
  multiply the spectrum element by element by `|h|²`, with the real FFT of `scipy.fft` on every
  CPU core, instead of two products with `(n, n)` diagonal matrices. Filtering per FWI iteration:
  22x faster in the small scenario of the benchmark, 11x in the medium one. They now work along any
  axis (0.1.10 only along axis 1).
- Margins around every device buffer (step 1.7): each buffer is a sub-buffer of an allocation
  `[margin | data | margin]`, the margins holding zeros (`wave_propagation.MARGIN_FILL`), at least
  64 KiB and enough for the farthest out-of-bounds access of the kernels, aligned for sub-buffers,
  and written whole by an explicit copy (one more copy per new buffer). The out-of-bounds reads and writes of the kernels (still
  there, as in 0.1.10) now stay inside the allocation and read zeros, as inside the guard bands of
  the golden masters: plain runs reproduce them bit for bit on NVIDIA and on POCL, where 0.1.10
  depended on the memory layout (crosswell and combined acquisitions) and returned NaN on POCL.
  `WavePropagator.margins_changed()` lists the buffers written outside.
- One OpenCL context per set of devices for the whole process (step 1.7,
  `PyFWI._opencl_device.shared_context`), instead of one per `WavePropagator`. Fixed: on NVIDIA,
  building propagators in a loop while the previous one is still referenced failed at the fourth
  with `CL_OUT_OF_RESOURCES`, because releasing a context while another one exists on the same device
  breaks the driver. Each propagator keeps its own command queue; `W.ctx` is now the same object for
  every propagator on the same devices.

### Numerical changes

- `fwi_tools.lowpass` and `adj_lowpass` (step 1.6): results differ from those of 0.1.10 by at most
  about 1e-15 relative (rounding of a different but equivalent computation, float64). The inversions
  and examples of the golden masters are unchanged: float32 outputs bit for bit, float64 within 1e-12.
