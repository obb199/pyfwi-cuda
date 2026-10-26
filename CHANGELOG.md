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
  `inpa['backend']`); OpenCL stays the default. With it `W.ctx` and `W.queue` are None. The public
  API is checked on both backends (step 2.5): sections and gradients, `W.W` and `W.seismogram`, the
  `show=True` plots, the example scripts (with `torchfwi.Fwi` on CPU tensors) and the original
  example script of 0.1.10.
- Numeric modes (step 2.4): `inpa['numerics']`, else `PYFWI_NUMERICS`, else the default of the
  backend (OpenCL: `legacy`, the results of 0.1.10; CUDA: `fixed`). `W.numerics` and `PyFWI.info()`
  say which. The `fixed` mode writes the floating-point constants as float literals with 9 significant
  digits: the kernels compute in FP32 only, and `dt` is no longer truncated to 6 decimals. With it the
  CUDA backend runs the medium scenario of the benchmark in 4.5 s (forward) and 6.3 s (gradient),
  against 47.3 s and 57.3 s for 0.1.10. The other corrections of `fixed` are listed under
  Numerical changes.

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
- The `fixed` mode (default of the CUDA backend) differs from `legacy` by up to 6 % on seismograms
  and on gradients (relative L2, propagation cases of the golden masters). Two corrections make
  almost all of it: the source injection (below) and the full `dt` (about 2 % on seismograms):
  0.1.10 samples the source with the true step but propagates with a step truncated to 6 decimals
  (0.075 % shorter for a user `dt` of 4 ms). FP32 arithmetic alone changes results by rounding
  (1e-6 on seismograms, 5e-5 on gradients).
- In the `fixed` mode the source is resampled to the internal time step (windowed-sinc
  interpolation, `PyFWI.engine.propagation.internal_source`) and injected at every step, divided by
  `dt_scale`. 0.1.10 adds one sample every `dt_scale` steps: a train of impulses, whose spectral
  images excite grid-scale noise and alias into the recorded band. With `dt` = 4 ms the seismograms
  of `legacy` differ by 3 to 6 % from those of the same wavelet sampled at the internal step (10 m
  grid), and the noise dominates them at 5 m; in `fixed` the difference is 3e-4 to 5e-4, the error of
  interpolating the truncated Ricker wavelet. Amplitudes are unchanged: every recorded interval
  receives the same amount of source as in 0.1.10. With `dt` of 0.6 ms or less nothing changes.
- In the `fixed` mode the residual sample `k` is injected at the internal step where sample `k` was
  recorded. 0.1.10 counts the samples down from the last row, and `nt` and `nt_ext` are rounded down
  separately: for some recording lengths (`dt` = 4 ms and `t` = 0.2 or 0.4 s, for example) every
  sample was injected one recorded sample early, and the gradient changed by about 50 % (its
  directional derivatives then differed from finite differences by 88 % instead of 32 %). The same
  rounding the other way (`dt` = 5 ms, `t` = 0.25 s, for example) recorded the last sample one row
  past the seismograms; `fixed` does not record it.
- In the `fixed` mode the gradient loop is the adjoint of the forward step. A forward step
  updates the velocities, then the stresses; 0.1.10 updates the adjoint stresses first, injects the
  residual of a step before the adjoint updates and accumulates the gradient after them, pairing
  the forward field with the adjoint field of about two steps earlier. `fixed` accumulates the
  gradient first, then updates the adjoint velocities and stresses, then injects the residual. On
  the check against finite differences of `tools/math_baseline.py` the vp gradient goes from 6.9 %
  to 1.9 % (median error) and the density gradient from 135 % to 51 %; gradients change by up to
  30 % (mostly density; cosine 0.96 or more with `legacy`).
- In the `fixed` mode the adjoint kernels damp every adjoint field by `dxpml + dzpml` inside the
  absorbing layer, as the forward kernels do. 0.1.10 damps `Ataux` with `dxpml` alone, `Atauz` with
  `dzpml` alone, `Avx` with `2 dxpml + dzpml` and `Avz` with `dxpml + 2 dzpml`, so that part of the
  adjoint field comes back from the boundary. vp gradient against finite differences: 1.9 % → 0.6 %.
- In the `fixed` mode `Grad` takes the shear strain from the z derivative of `vx` and the x
  derivative of `vz`, as `update_tauxz` does; 0.1.10 used the z derivative of `vz` in place of that
  of `vx`, so the `mu` gradient, and with it the `vs` and density gradients, was wrong. Against
  finite differences, with the two corrections above: vs 60 % → 0.27 %, density 135 % → 1.4 %, vp
  6.9 % → 0.6 % (median error; `tools/math_baseline.py`). The raw gradients of `fixed` differ from
  those of `legacy` by up to 45 % (`mu`; cosine 0.89 or more) on the propagation cases.
- In the `fixed` mode order 8 uses the Taylor coefficients of the staggered first derivative
  (1225/1024, -245/3072, 49/5120, -5/7168). 0.1.10 uses them multiplied by 1434 and rounded to
  integers, so that `c1 + 3 c2 + 5 c3 + 7 c4` is 1436/1434 instead of 1: every order-8 wave travels
  0.14 % too fast (arrivals 0.16 ms early after 300 m at 2,500 m/s, 2.8 ms after a 2 s record).
  The truncated `dt` of 0.1.10 hid about half of it. The rounding also leaves a second-order error
  term: on grid refinement the order-8 scheme of 0.1.10 converges at order 2.2 to 2.7, against 7.5
  with the Taylor coefficients (order 4 converges at order 4.1 in both modes). With the correction
  the order-8 traces of a homogeneous model agree with the analytic solution to a correlation of
  0.9990 (0.9987 in 0.1.10).
- In the `fixed` mode the optimisers of `FWI` no longer depend on the amplitude of the data. 0.1.10
  gives L-BFGS an absolute `pgtol = 1e-8` and the conjugate gradient an absolute `gtol = 1e-8`; with
  a unit-amplitude source the gradients are 1e-11 to 1e-8, so L-BFGS stopped at iteration 0. The
  conjugate gradient also takes a first trial step proportional to the gradient, which changed the
  float32 model by about 1e-10 m/s, nothing: its line search gave up and returned the starting
  model. `fixed` makes both tolerances relative to the largest gradient component at the start of
  each stage, and the conjugate gradient minimises the misfit divided by that value (same minimum
  and directions; a first step of about 1 m/s). The examples keep their scaled source.
- In the `fixed` mode the crosswell kernels record `vz` in the `vz` seismogram; 0.1.10 (and the
  `legacy` mode) recorded `vx` there, so with `acq_type=0` the `vz` data, and the misfit and
  gradient of components 2 and 4, are a different quantity in `fixed`.
- In the `fixed` mode the kernels no longer access memory outside their buffers, except one row past
  the field edges: the order-4 stencils skip their zero-coefficient terms (which read up to 4 cells
  away) and the receiver loops stop at the last receiver (they read past the residual and wrote past
  the adjoint fields). No numerical change beyond rounding, since those reads returned zeros from the
  margins of step 1.7. The read of the row past the edge stays in both modes: that zero margin is the
  outer boundary, and not updating the last row instead worsened the absorbing layer.

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
