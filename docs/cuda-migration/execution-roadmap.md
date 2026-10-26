# PyFWI-CUDA: execution and verification roadmap

Oct 25, 2026 · Gabriel Carvalho · private repository [obb199/pyfwi-cuda](https://github.com/obb199/pyfwi-cuda)

This roadmap turns the [analysis and migration plan](analysis-and-plan.md) into a sequence of executable steps, from a ready environment to version 0.4 and maintenance. Each step says what goes into the code, which tests come with it and which criterion must be met before moving on. The technical decisions (CuPy as the base, a single kernel source for parity, `legacy` and `fixed` numeric modes, frozen API) come from the analysis and are not re-discussed here.

## Sequence at a glance

| # | Step | Version / gate |
| --- | --- | --- |
| P | Environments, repository and environment checker | **done** (`babbc0e`) |
| 0.1 | Fork hygiene: README, GPLv3 notice, CHANGELOG, lint configuration, CPU CI | Phase 0 · **done** |
| 0.2 | Known bugs of 0.1.10 become strict `xfail` tests | **done** |
| 0.3 | Example scripts that exercise the public API and must never break | **done** |
| 0.4 | Golden-master generator in the reference environment | **done** |
| 0.5 | Characterisation tests in dev and reference × dev cross-check | **done** |
| 0.6 | Mathematical baseline of 0.1.10 (gradient, analytic solution, convergence, adjoint) | **done** |
| 0.7 | Reference benchmark in the three scenarios | **done** |
| 0.8 | Profiling bench: Nsight Systems, Nsight Compute, compute-sanitizer | **done** (Nsight Compute waits for the owner) |
| 0.9 | Self-hosted runner on the RTX and GPU jobs | prepared (runner waits for the owner) · **[G0](gates/G0.md) approved with 2 exceptions** |
| 1.1 | Imports and packaging (current SciPy and NumPy, declared dependencies) | Phase 1 · **done** |
| 1.2 | Fixed OpenCL device selection | **done** |
| 1.3 | Cached kernels and fixed arguments | **done** |
| 1.4 | Residual and wavelet on the GPU; non-blocking copies | **done** (wavelet: Phase 3) |
| 1.5 | Contiguous checkpoints; `W.W` as a view | **done** |
| 1.6 | Element-wise low-pass filter | **done** |
| 1.7 | Padded buffers: no more reads outside the allocation | **done** (and one shared context per device) |
| 1.8 | `PyFWI.info()` and diagnostic log | **done** |
| 1.9 | Fork release 0.1.11 | **done** · **[G1](gates/G1.md) approved** |
| 2.1 | Engine and backend interface; OpenCL behind the interface | Phase 2 · **done** |
| 2.2 | Backend selector and fallback | **done** |
| 2.3 | Parity CUDA backend (single source, `legacy` mode) | **done** |
| 2.4 | `legacy` and `fixed` numeric modes | |
| 2.5 | API contract on the CUDA backend | |
| 2.6 | `[cuda13]` and `[cuda12]` extras | |
| 2.7 | Parity report and release 0.2.0 | **G2** |
| 3.1–3.10 | CUDA optimisations, one per commit, with a benchmark before and after | Phase 3 |
| 3.11 | `auto` default and release 0.3.0 | **G3** |
| 4.1–4.6 | Cost function on the GPU, boundary saving, FP16, multi-GPU, DLPack, geophysical validation | Phase 4 |
| 4.7 | Release 0.4.0 | **G4** |
| 5.x | Final documentation, upstream, publication and conditional optimisations | Maintenance |

Estimated duration in the analysis: about 20 weeks for one person (Phase 0 ≈ 3, Phase 1 ≈ 3, Phase 2 ≈ 4, Phase 3 ≈ 6, Phase 4 ≈ 4), or about 14 with two people, because Phase 4 can run in parallel with Phase 3 after G2.

## Working rules

**One step, one or more small commits.** Each commit brings one feature, the tests that prove it and a message that explains what changed and why. Before each commit:

1. `ruff check` on the changed files (inherited code is only cleaned when it is touched).
2. Fast suite in the dev environment: `../envs/dev/bin/python -m pytest -m "not slow"`.
3. If the commit touches kernels, backends or the time loop: also `-m opencl`, `-m gpu` and `compute-sanitizer` on the changed kernels.
4. If the commit is about performance: benchmark before and after, with the numbers in the commit message.
5. Commit and push to `origin/main`.

**Language and documentation.** Code, comments, docstrings, README, CHANGELOG, design documents and commit messages are written in English. Every function and class documents what each argument and return value means (units, shapes, dtypes), the important variables, and **where it is used** (callers and tests), in NumPy docstring style with a "Used by" section. The goal is that a newcomer can read any file without guessing.

**Examples are part of the contract.** The scripts in `example/scripts/` use only the public API, the way a user would. Every one of them runs in the test suite (in a reduced size) and its outputs are compared with those of PyFWI 0.1.10. A change that breaks an example is a regression, even if every unit test passes.

**Gates.** At the end of each phase, a gate gathers the checks that do not run at every commit (full golden-master matrix, benchmarks, profiling, complete FWI). The result is written to `docs/cuda-migration/gates/G<n>.md`, with numbers, versions and hashes. The next phase only starts once the gate is approved.

**Known bugs.** Each bug inherited from 0.1.10 becomes an `xfail(strict=True)` test. When the fix lands, the test starts passing, `strict` makes it fail as "XPASS", and the test must be updated in the same commit.

**Environments.** Created by `tools/setup_envs.sh` and checked by `tools/check_env.py`, both already in the repository.

| Environment | Content | Use |
| --- | --- | --- |
| `envs/ref-0.1.10` | Python 3.10, PyFWI 0.1.10 from PyPI, NumPy 1.21.6, SciPy 1.7.3, PyOpenCL 2022.2.4, POCL 3.0 | Generate golden masters and the baseline; never receives code from the fork |
| `envs/dev` | Python 3.12, editable fork, CuPy 14.2 (CUDA 13), cuda.core 1.2, PyOpenCL 2026.1, POCL 3.0, pytest-benchmark | Development and tests |
| `envs/cuda12` (Phase 2) | Same as dev with `cupy-cuda12x` | Test the `[cuda12]` extra |
| `envs/torch` (Phase 4) | Dev + PyTorch with CUDA | Test DLPack without mixing the `nvidia-*` wheels of CuPy and PyTorch |

## Current state

Target machine: GeForce RTX 3060 12 GB (Ampere, sm_86), driver 580.95.05 (CUDA 13.0), Ryzen 7 5700X, 23 GB of RAM, Debian 12. System CUDA Toolkit 12.4 with `nvcc`, Nsight Systems 2023.4, Nsight Compute 2024.1 and compute-sanitizer 2024.1.

What the preparation step delivered and verified (`babbc0e`, `ad049b8`, `f0ee2d2`):

- Local fork with `upstream` = AmirMardan/PyFWI (push disabled) and `origin` = the private repository. Base: `v0.1.10` = `1434a16`.
- Both environments, with versions pinned in `requirements/`. Re-created from scratch in a temporary folder, they matched the lock files exactly.
- `check_env.py` passes in both environments: NVRTC produces cubin, a CUDA Graph captures and replays, cuFFT has a relative error of 3×10⁻⁷ against NumPy, `cupyx` has a maximum error of 1.2×10⁻⁷ against SciPy, and 0.1.10 runs forward + gradient on the RTX bit-for-bit deterministically. With `--no-gpu` it also runs on machines without NVIDIA (CI).

**Findings that the analysis did not record**, all handled in this roadmap:

| Finding | Consequence | Handled in |
| --- | --- | --- |
| On POCL, 0.1.10 returns 99% NaN in the seismogram and different results on every run. With zeros after the end of the buffers the seismogram becomes finite and deterministic; the gradient still has NaN, consistent with the reads before the start (`i − 3` with `i = 2`). | POCL cannot be a reference for 0.1.10 as it is. Golden masters come from NVIDIA OpenCL, and the CPU CI only compares outputs after step 1.7. | 0.2, 0.4, 1.7 |
| `segyio` is imported at the top of `model_dataset.py` and `seismic_io.py`, but it is not in `install_requires` and is commented out in `requirements.txt`. | `pip install PyFWI` is not even enough to build a model. | 1.1 |
| `from numpy.lib.function_base import kaiser` breaks with NumPy 2. | `model_dataset` and `seismic_io` do not import in the dev environment. | 1.1 |
| `inpa['device']` fails when the platform has a single device: `create_some_context` reports a "left over" choice. It happens with PyOpenCL 2022.2.4 (reference environment), not with PyOpenCL 2026.1.4. | With old PyOpenCL the device must be chosen through `PYOPENCL_CTX`. Step 1.2 removed the dependency on `create_some_context` (the backend selector needs explicit device choice anyway). | 0.2, 1.2 (done) |
| The same forward takes 1.42 s with PyOpenCL 2022.2.4 and 2.07 s with PyOpenCL 2026.1.4 (probably `RepeatedKernelRetrieval`). | The reference benchmark must record both versions. Step 1.3 removed the difference (0.52 s and 0.49 s). | 0.7, 1.3 (done) |
| The NVRTC 13.4 installed by pip is newer than the driver (13.0): the PTX it generates cannot be loaded, only cubin can. | The CUDA backend always compiles to cubin for the real architecture; a test enforces it. Driver 580 is kept on purpose because it is the minimum supported. | 2.3, 2.6 |
| The PTX of the hot kernels has from 43 (`update_velx`) to 138 (`Grad`) FP64 instructions in order 8 (the analysis measured 42 to 131). | Confirms the precision guard; the test already exists and flips in the `fixed` mode. | 2.4 |
| The crosswell `vz` bug is on lines 27 and 74 of `elastic_crosswell.cl` (`injSrc` and `hessian_seismogram`). | Both lines are covered by the known-bug test. | 0.2, 2.4 |
| In one process, the 4th forward + gradient of a 50×50 case **crashes** the NVIDIA driver (`CL_OUT_OF_RESOURCES`), also with the published 0.1.10; with zero-filled guard bands around the buffers the same sequence runs. Attributed here to the out-of-bounds reads; step 1.7 found the real cause: releasing a context while another one exists on the same device (the guard bands kept every context alive). | A user running several propagations in one process (a notebook, TL-FWI) can hit it. Fixed in step 1.7 by one shared context per device; until then the test suite ran PyFWI cases inside zero guard bands. | 0.2, 0.4, 1.7 (done) |
| Five receiver loops have no bound on the receiver index (`elastic.cl` lines 89 and 352, right well; `elastic_crosswell.cl` lines 24, 48 and 71). Rows beyond the last receiver are recorded into the next time sample, and `Adj_injSrc` reads past the end of the residual and accumulates past the end of the five adjoint fields. | With `acq_type` 0 and 2 the results of 0.1.10 depend on what lies next to the buffers: with `acq_type` 2 they even differ between two processes with a different allocation history. Only the surface acquisition is reproducible as is. | 0.2, 0.4, 2.4 |
| Race conditions at two kinds of grid points: a source on a receiver cell (`injSrc` injects and records in the same launch) and two receivers on one cell (`Adj_injSrc` adds with a non-atomic `+=`). The kernels place receivers from the first one of each line every `dxr` cells, and the right well runs to the bottom of the grid, so the default geometry of `acq_parameters` hits both (sources on the surface line; wells crossing it in `acq_type` 2). | Results change between runs and between devices. `test/helpers.py` models the cells the kernels really use and builds race-free geometries; the golden masters and the examples use them. Fixed for good when injection and recording become separate 1D kernels. | 0.2, 0.3, 0.4, 3.4 |
| With zero guard bands and a race-free geometry, 0.1.10 is deterministic and **NVIDIA OpenCL and POCL agree bit for bit** (acquisitions 0, 1, 2; orders 4 and 8; seismograms and gradients). For the surface acquisition a plain run gives the same bits. | The CPU CI can check results against golden masters made on the GPU from step 0.5 on, instead of waiting for step 1.7. | 0.4, 0.5 |
| With the default `chpr=0` of `WavePropagator`, the gradient is wrong (cosine 0.54 with the checkpointed one, amplitude 60× smaller): the field is rebuilt backwards through the absorbing boundary over the whole recording. `chpr ≥ 5` gives exactly the gradient of `chpr = 100`. | The examples always pass `chpr`; strict known-bug test. | 0.2, 0.3, 4.2 |
| `FWI.lbfgs` gives L-BFGS-B an absolute `pgtol = 1e-8`. With a unit-amplitude source, realistic gradients are 1e-11 to 1e-8, so the optimiser "converges" at iteration 0 and returns the starting model (e.g. a 100×100, 7 m model). Scaling the source wavelet by 1000 or more (it scales observed and predicted data alike) gives a real inversion, the same for any factor. The conjugate-gradient path also left the model unchanged in a small test. The official `fwi_example_surface.py` does update the model, but its model error grows (elastic data inverted from an acoustic start). | The FWI example scales the source and explains why. The tolerance is recorded as a known bug; CG and the official example are characterised in the golden masters. | 0.2, 0.3, 0.4, 2.4 |
| Model library: `ModelGenerator(..., smoothing=1)` returns the homogeneous background (no anomalies) for `louboutin`, `yang`, `perturbation_dv`, `hu_circles` and `hu_laminar`, not a filtered model; density is in g/cm³ for `louboutin` and in kg/m³ for `yang`; `dupuy` fails with `FileNotFoundError` when PyFWI is installed from the PyPI wheel (no `data/` folder). | Documented in `example/scripts/models.py` and its test; the wheel content is fixed in step 1.1. | 0.3, 1.1 |
| The source wavelet is sampled at the user `dt` and added once every `dt_scale` internal steps, i.e. as a train of impulses. At grid spacings of 4 to 6 m (15 Hz wavelet, `dt` = 4 ms) this excites a grid-scale checkerboard mode that never leaves the model: 2 to 5 % of the peak energy remains after 1.2 s, against 4e-9 at 10 m, and seismograms show a strong spurious ringing. Injecting the source at every internal step (`dt` = 0.5 ms, `dt_scale` = 1) removes it (2e-9). The adjoint injection of residuals works the same way. | Known-bug test; the `fixed` mode interpolates the source (and the residual) to the internal step. | 0.2, 0.6, 2.4 |
| The gradient of 0.1.10 is approximate: its directional derivatives differ from finite differences of the misfit by 31 % in the median (one of six directions even has the wrong sign). The forward propagation itself agrees with the analytic solution to a correlation of 0.9987. | Recorded as the mathematical baseline; the `fixed` mode and the CUDA kernels must not be worse, and are expected to improve it. | 0.6, 2.4 |
| `RmProfilingAdminOnly: 1`: Nsight Compute cannot read GPU counters without root. | Depends on the project owner (see [Pending actions](#pending-actions-for-the-project-owner)). | 0.8 |
| The inherited workflows (`pr.yml`, `publish.yml`) test Python 3.8 to 3.10 on three operating systems with the old versions and would publish to PyPI as `PyFWI`. | They must be replaced before any PR or release. | 0.1 |
| PoCL 3.0 (LLVM 14), the OpenCL-on-CPU that comes with the pinned PyOpenCL wheels and the newest on PyPI, cannot compile any kernel on CPUs that LLVM 14 does not know, such as the AMD EPYC 9V45 (Zen 5) of some GitHub-hosted runners (`unknown target CPU 'generic'`); it has no option to name another CPU. The CI failed on such a runner (`fc7f969`) and passed on the others. | On those runners the OpenCL tests are reported as skipped with a warning (hook in `test/conftest.py`), so the CI checks less there; the local suites are not affected. A newer PoCL (7.1 on conda-forge) would remove the limitation but changes the CPU reference; to be decided when POCL comparisons become a CI requirement. | 0.1, 1.7 |

## Catalogue of tests and checks

Each step refers to the codes below. "Every commit" means running locally before the commit and in CI after the push.

| Code | What it checks | How | Where it runs | When | Criterion |
| --- | --- | --- | --- | --- | --- |
| LNT | Style and static errors | `ruff` on new code; `ruff --select E9,F63,F7,F82` on the whole repository, as the original CI did with flake8 | CPU CI | Every commit | Zero errors |
| UNI | Isolated functions: geometry, wavelet, filters, layout, selector, parsing | `pytest`, no device | CPU CI | Every commit | All pass |
| EXA | Example scripts keep working and give the same results | Each script in `example/scripts/` runs in reduced size; outputs compared with 0.1.10 | CPU CI (smoke, POCL after 1.7) and GPU runner (comparison) | Every commit | Runs without edits; outputs within the phase tolerance |
| CAR | Outputs equal to those of 0.1.10 (golden master) | `.npz` files generated in the reference environment (zero guard bands, race-free geometry); compared in dev | CPU CI (POCL agrees bit for bit with NVIDIA) and GPU runner | Subset at every commit; full matrix at the gate | Per phase (see [Tolerances](#tolerances)) |
| BIT | Bit-for-bit equality of the raw propagator outputs | `np.array_equal` on seismograms and gradients before host post-processing | Local and GPU runner | Every commit of Phase 1 and every refactoring | Identical |
| DET | Two runs give the same result | Same case twice on the same device | Local and GPU runner | Every commit that touches kernels | Identical; no floating-point `atomicAdd` |
| PAR | Parity CUDA backend × OpenCL on the same GPU | Golden case matrix, `legacy` mode | GPU runner | Every commit of Phase 2; full matrix at the gate | Relative L2 error ≤ 1e-6 |
| OTM | Optimised kernel (`fixed`) × parity CUDA (`fixed`) | Same matrix | GPU runner | Every commit of Phase 3 | Seismograms ≤ 1e-5 and gradients ≤ 1e-4 (relative L2) |
| MAT | Mathematical correctness | Gradient × centred finite differences; homogeneous medium × analytic solution; convergence order; adjoint dot product; PML reflection | Local and GPU runner | Baseline in 0.6; again at every gate | See step 0.6 |
| PRE | No unintended FP64 | NVRTC compiles the kernels and the test counts `.f64` in the PTX; no GPU needed | CPU CI | Every commit that touches kernels | `fixed` mode: zero FP64 in the hot kernels |
| SAN | GPU memory and concurrency | `compute-sanitizer` with `memcheck`, `racecheck`, `initcheck` and `synccheck` on a small model | GPU runner | Every commit that touches CUDA kernels | Zero errors |
| API | Frozen public contract | Signatures, `inpa` keys, types, `dtype`, shapes, axis order, `W.W`, `W.seismogram`, `show=True`; the original scripts in `example/` run without edits | CPU CI (POCL, after 1.7) and GPU runner | Every commit | Same behaviour as 0.1.10 |
| SEL | Backend choice and fallback | Precedence `inpa['backend']` → `PYFWI_BACKEND` → default; `auto` without CuPy, without GPU or with an old driver (simulated with monkeypatch); `cuda:strict`; PCI bus ID matching; log message | CPU CI and GPU runner | Every commit of the selector | Every branch covered |
| PKG | Packaging | Wheel and sdist build; install in a clean venv; `.cl`, `.cu`, `data/` and `torch/` present; extras; import of every module; `PyFWI.info()` | CPU CI | Every packaging commit; always before a release | Installs and imports on Python 3.10 to 3.13 |
| DES | Performance | `pytest-benchmark` in the three scenarios; median of 5 runs after warm-up; records forward, gradient, FWI iteration, host time per step, launches and copies per step, peak memory | GPU runner | Before and after every performance commit; nightly | No regression above 5%; phase targets |
| PRF | Where the time goes | Nsight Systems (GPU occupancy, idle gaps, number of launches and copies); Nsight Compute (achieved bandwidth per kernel); OpenCL profiling events | Manual, on the RTX | At every gate and whenever a benchmark surprises | Report attached to the gate |
| MEM | Memory | Peak on the GPU and on the host; checkpoint policy under a forced budget | GPU runner | Memory commits; gates | Within the estimate; same result with any budget |
| FWI | End-to-end inversion | Multi-scale FWI and TL-FWI on small models; misfit curve and model error; review by the geophysicist | GPU runner | Gates G1, G2 and G4 | See [Tolerances](#tolerances) |
| INT | Interoperability with PyTorch | `torchfwi.Fwi` with CPU and GPU tensors; DLPack without copies (same `data_ptr`); gradient × finite differences | GPU runner | `torchfwi` commits; G4 | Same as the NumPy path |
| EST | Stability and limits | Long runs, models close to the VRAM limit, many shots; clear errors when memory runs out | GPU runner | Gates G3 and G4 | No NaN, no hang, useful message |
| DOC | Documentation | Valid relative links and anchors; cited commits exist; notebooks executed in reduced size | CPU CI | Every documentation commit; release | Everything resolves and runs |

### Tolerances

| Comparison | Criterion | Origin |
| --- | --- | --- |
| Phase 1 × 0.1.10, raw propagator outputs on the same device and environment | Bit-for-bit identical | Analysis |
| Fork in dev × golden masters, before step 1.1 (PML profile differs under NumPy 2) | Seismograms ≤ 1e-6, gradients ≤ 1e-4 (relative L2) | Measured in step 0.5 (5.7e-7 and 2.8e-5) |
| Misfit across NumPy versions | Relative ≤ 1e-6 | Measured in step 0.5 (5e-8: summation order) |
| New × old low-pass filter (float64) | Relative difference ≤ 1e-12 | Analysis |
| CUDA `legacy` × OpenCL `legacy` | Relative L2 ≤ 1e-6 (initial target, confirmed at G2); measured in step 2.3: 1.1e-6 for seismograms, 1.9e-5 for gradients; tests use 3e-6 and 6e-5 until G2 | Analysis; step 2.3 |
| Optimised CUDA × parity CUDA (`fixed`) | Seismograms ≤ 1e-5; gradients ≤ 1e-4 | Analysis |
| `fixed` × `legacy` | Measured and published in the CHANGELOG, no fixed limit | Analysis |
| Adjoint dot product | Relative deviation ≤ 1e-5 for the new kernels; only measured for 0.1.10 | Analysis; step 0.6 |
| FWI (2 iterations) between backends or environments | Final model with relative L2 ≤ 1e-4; misfit per iteration ≤ 1e-4 | New |
| FP16 | Own limits validated by a complete FWI: final model error at most 10% above FP32 (target to confirm in 4.3) | Analysis, with a proposed target |
| Performance between commits | Regression ≤ 5% | Analysis |

## Phase 0: safety net and baseline

No change in the behaviour of the package. This phase builds everything that will prove the following phases broke nothing.

### 0.1 Fork hygiene

- **Goes in:** fork README (what it is, how to set it up, origin and citation of the original authors), the modification notice required by GPLv3, `CHANGELOG.md`, a `[tool.ruff]` configuration in `pyproject.toml` (full rules for new code, only critical errors for inherited code) and `.github/workflows/ci.yml` with a CPU job (LNT, UNI, PKG, DOC and, later, PRE and EXA). `pr.yml` and `publish.yml` are removed, because they test versions the fork does not support and would publish under the name of the original.
- **Tests:** LNT; DOC (relative links of every `.md` in the repository); repository hygiene (licence text identical to upstream, modification notice present, no workflow able to publish to PyPI); the CI must be green on the first push.
- **Criterion:** green CI; no workflow able to publish to PyPI.

### 0.2 Known bugs as tests

- **Goes in:** `test/test_known_bugs.py`, with one `xfail(strict=True)` per bug: out-of-buffer reads, `vz` storing `vx` in `elastic_crosswell.cl`, truncated `dt` in the `#define`, float literals without the `f` suffix, `inpa['device']` with a single device, broken imports with current SciPy and NumPy, undeclared `segyio` and `pyopencl`, and the wheel without `torch/` and `data/`.
- **Out-of-buffer reads, made deterministic:** reading outside a buffer is undefined behaviour, so a plain run may or may not show it. `test/oob_guard.py` allocates every PyFWI buffer inside a larger one whose bands before and after the data are filled with NaN, and hands PyFWI a sub-buffer covering only the data. Any out-of-bounds read then returns NaN on every device (NVIDIA or POCL), and "outputs are finite" becomes a reliable check. A control run with zero-filled bands gives finite outputs, which confirms that the NaN comes from the reads and not from the harness.
- **Tests:** the tests themselves; each must fail for the right reason (`raises=` or a precise assertion, not just "something failed"). Bugs that depend on the environment (imports) use a `condition`, so they are `xfail` only where they actually happen.
- **Criterion:** all `xfail`; no `xpass`.

### 0.3 Example scripts

- **Goes in:** `example/scripts/`, a set of small, commented scripts that use only the public API, one per use case: surface forward modelling, crosswell and combined acquisitions, gradient with checkpoints, multi-scale FWI (L-BFGS and CG), time-lapse FWI, gradient through PyTorch (`torchfwi.Fwi`) and the `lam`/`mu`/`rho` parameterisation. Each script has a `main(size="small")` function that returns its numerical results and can also be run from the command line.
- **Tests:** EXA: each script runs in the suite in its small size, with finite results and, where it applies, a physically meaningful check (the misfit decreases, the gradient points to the anomaly, the time-lapse difference is located where the model changed).
- **Criterion:** every script runs in both environments (reference and dev) where its imports allow; scripts that need `fwi.py` are `xfail` in dev until step 1.1.

### 0.4 Golden-master generator

- **Goes in:** `test/golden/generate.py`, run only in `envs/ref-0.1.10`, on NVIDIA OpenCL. It writes `.npz` files with:
  - forward (seismograms of the 5 components) and gradient, including the raw outputs captured before `grad_lmd_to_vd` and the `gaussian_filter`;
  - 2 FWI iterations (L-BFGS and CG), one round of every `TimeLapse` strategy and one forward/backward of `torchfwi.Fwi`;
  - the outputs of every example script of step 0.3;
  - a pairwise combinatorial matrix, not the full product: `acq_type` 0/1/2, `components` 0 to 4, order 4/8, `energy_balancing`, `chpr` 0/20/100, `dv` and λ/μ/ρ parameterisation, on a 60×60 model with about 300 steps;
  - metadata per file: versions, GPU, driver, commit, `inpa` and SHA-256 of the outputs.
- **Defined behaviour of 0.1.10:** because the out-of-bounds accesses of 0.1.10 make some results depend on the memory layout (and can crash the GPU), every case is run inside **zero-filled guard bands** (`test/oob_guard.py`): reads outside a buffer return 0 and writes land in the band. For the surface acquisition this is bit-for-bit identical to a plain run of 0.1.10; for `acq_type` 0 and 2 it is the only reproducible definition. The generator also records, per case, whether the plain run matches, so the gap is documented. Step 1.7 pads the buffers in the same way, so its results can be compared bit for bit.
- **Race-free geometry:** every case uses `helpers.race_free_geometry`, so no source sits on a receiver cell and no two receivers share one. Under these two rules the results are deterministic and identical on NVIDIA and POCL (`test/test_reproducibility.py`).
- **Checks:** the generator runs every case twice and aborts if the results are not bit-for-bit identical (DET). The files stay in the repository if they add up to less than 20 MB compressed; above that, Git LFS.
- **Tests:** UNI of the case matrix (every pair of factor values is covered) and of the loader (complete metadata, expected keys, hashes match).
- **Criterion:** full matrix generated, deterministic and with metadata.

### 0.5 Characterisation in dev and cross-check

- **Goes in:** `test/test_characterization.py`, which re-runs every golden case with the fork's code in the dev environment and compares. FWI cases only run in dev after step 1.1 (before that `fwi.py` does not import) and stay `xfail` until then.
- **Key check:** with the code still identical to 0.1.10, do the raw outputs in dev match those of the reference environment bit for bit? The only difference between the two environments is versions (PyOpenCL 2022 × 2026, NumPy 1.21 × 2.5). If they match, the BIT criterion of Phase 1 applies directly in dev. If not, the test localises the stage that diverges (wavelet generation, model assembly, kernel or post-processing), and the criterion becomes BIT on raw outputs plus the ulp tolerance on post-processing, recorded at G0.
- **Tests:** CAR, DET.
- **Criterion:** forward and gradient of every case approved under the chosen criterion; decision recorded.
- **Result (measured):** the dev environment does **not** reproduce the golden masters bit for bit: seismograms differ by up to 5.7e-7 and gradients by up to 2.8e-5 (relative L2), identically on NVIDIA and POCL. The only kernel input that differs is the PML damping profile: `pml_delta_calculation` mixes a `numpy.float32` grid spacing with Python numbers, and NumPy 2 (NEP 50) computes `n_pml * dh` and `3 / (2 * delta1)` in float32 where NumPy 1.21 used float64, changing the last bit of 3 of 10 values. With an explicit float64 formula every seismogram and gradient matched bit for bit; only the misfit (a float64 sum) still differs by up to 5e-8. **Decision:** `test/test_characterization.py` compares bit for bit wherever the installed PyFWI computes the PML of 0.1.10 (the reference environment, and dev after step 1.1), and with interim tolerances otherwise (1e-6 for seismograms, 1e-4 for gradients); the misfit is compared with a relative tolerance of 1e-6. A mutation test showed that the interim tolerances miss a 1e-7 change, so **the PML fix is the first change of step 1.1**.

### 0.6 Mathematical baseline of 0.1.10

- **Goes in:** `test/test_math.py` (marker `slow`), with:
  - **gradient × finite differences:** ⟨g, δm⟩ against [J(m + εδm) − J(m − εδm)] / 2ε, with ε swept over decades; `g_smooth` is already 0 by default. Records the best relative error;
  - **analytic solution:** acoustic limit (vs = 0, constant ρ) against the 2D Green's function convolved with the Ricker wavelet, away from the boundaries;
  - **convergence order:** the error against the analytic solution decreases with dh and dt at the expected rates (4 or 8 in space, 2 in time);
  - **PML:** energy reflected by the boundary below a threshold;
  - **stability:** a long run without energy growth;
  - **adjoint dot product:** ⟨A u, v⟩ × ⟨u, Aᵀ v⟩ for one time step, using the update kernels and their adjoints. For 0.1.10 it is only measured; the ≤ 1e-5 criterion applies to the new kernels.
- **Criterion:** numbers recorded at G0 as the baseline. From then on no phase may make them worse (except the dot product, which must improve up to the limit).
- **Result (measured on 0.1.10, stored in `test/golden/math_baseline.json` by `tools/math_baseline.py`):**
  - propagation against the analytic 2D solution (homogeneous acoustic medium, 10 m grid, 100-400 m offsets, orders 4 and 8): correlation ≥ 0.9987, no peak-time shift;
  - absorbing boundary: about 1.1 % reflection at a receiver 100 m from it; residual energy after 1.2 s: 4e-9 of the peak;
  - gradient against centred finite differences (6 smooth interior directions): the ratios are 1.82, -0.25, 1.30, 0.83, 0.85 and 1.32, a median error of 31 %; the gradient of 0.1.10 is only approximate;
  - **not measurable on 0.1.10:** the convergence order (at grid spacings of 4 to 6 m a known bug, the impulsive source injection, adds grid-scale noise that hides the discretisation error) and the adjoint dot-product test (the public API exposes no single time step and its adjoint). Both are measured in Phase 2, on the parity kernels and in the `fixed` mode.
  - `test/test_math.py` (marker `slow`) requires every later version to be at least as good on each metric.

### 0.7 Reference benchmark

- **Goes in:** `bench/` with `pytest-benchmark` and three scenarios: small (docs example, 100×100 + PML, 4 shots, 1,051 steps), medium (≈400×400, 8 shots, 3,000 steps) and large (≈450 thousand cells, 6,000 steps). Plus a counter of launches and copies per step (instrumentation by monkeypatching the PyOpenCL calls).
- **Care points:**
  - Memory: with `chpr = 20`, the medium scenario uses ≈15 GB of RAM in 0.1.10, and the machine has 23 GB. The large scenario with 50 shots does not fit; it is measured with 1 and 2 shots and extrapolated, and the extrapolation is recorded as such.
  - Variability: GeForce clocks vary with temperature. Warm-up, median of 5, temperature and clock recorded per run (`nvidia-smi --query-gpu`), and runs with throttling discarded.
  - Versions: 0.1.10 measured with PyOpenCL 2022 (ref) and 2026 (dev), to separate the fork's gain from the PyOpenCL regression.
- **Tests:** DES; UNI of the launch counter.
- **Criterion:** coefficient of variation below 3% in every scenario; results in `bench/baseline/` with machine metadata.

### 0.8 Profiling bench

- **Goes in:** `tools/profile.sh`, with recipes for `nsys`, `ncu` and `compute-sanitizer`, and a sanitizer self-test.
- **Checks:**
  - Sanitizer self-test: a CuPy kernel with a deliberate out-of-buffer access must be detected by `compute-sanitizer` 2024.1 with the CUDA 13 runtime. If it is not, install CUDA Toolkit 13.x system-wide (see Pending actions).
  - `ncu` reading counters (after access is granted); `nsys` producing the timeline of a CUDA forward.
  - NVIDIA OpenCL is measured with profiling events (`CommandQueue` with `PROFILING_ENABLE`), not with Nsight.
- **Criterion:** the three tools working, with a sample report attached.
- **Result:** `compute-sanitizer` 2024.1 (system CUDA 12.4) works with the CUDA 13 runtime of CuPy: it reports a deliberate out-of-bounds write and nothing for a correct kernel, so installing CUDA Toolkit 13 is **not** needed. Nsight Systems records CuPy kernels. Nsight Compute stays blocked until the GPU counters are opened (pending action 1). `tools/opencl_profile.py` measured that the GPU is busy 28 % of the time in the small scenario and 51 % in the medium one ([report](reports/benchmark-baseline-0.1.10.md)).

### 0.9 Self-hosted runner and GPU jobs

- **Goes in:** a GitHub Actions runner on this machine, restricted to the private repository; `opencl`, `gpu` and `sanitizer` jobs on every push, and a nightly DES job.
- **Checks:** a test push triggers the jobs; a deliberately failing test makes the job fail; the runner does not run on PRs from forks.
- **Criterion:** CPU and GPU CI green on `main`.
- **Status:** prepared, waiting for the owner. `.github/workflows/gpu.yml` (manual-only until a runner exists) re-creates the environments from the commit under test and runs the environment check, the fast and slow suites and the reference-environment suite; a nightly job runs the benchmark. [self-hosted-runner.md](self-hosted-runner.md) explains how to register the runner safely (pending action 3).

### Gate G0

- [ ] Complete, deterministic golden masters with metadata (0.4).
- [ ] Example scripts running and part of the golden master (0.3).
- [ ] Characterisation criterion decided and approved in dev (0.5).
- [ ] Mathematical baseline recorded (0.6).
- [ ] Reference benchmark with CV < 3% in the three scenarios (0.7).
- [ ] Profiling and sanitizer working (0.8).
- [ ] CPU and GPU CI green (0.1, 0.9).
- [ ] Every known bug in strict `xfail` (0.2).

## Phase 1: host fixes on OpenCL (fork version 0.1.11)

No computation of the propagator changes, so the criterion is BIT against the golden master at every step, and EXA at every commit. The analysis estimates that these first two phases already deliver most of the gain on small models.

### 1.1 Imports and packaging

- **Goes in:** first of all, `pml_delta_calculation` computed explicitly in float64, which restores the PML profile of 0.1.10 under NumPy 2 and with it the bit-for-bit characterisation in dev (step 0.5); then our own `MemoizeJac` (a small class, not depending on private SciPy names); `np.kaiser` instead of `numpy.lib.function_base`; `segyio` imported lazily, only in the functions that read SEG-Y; `pyopencl` in `install_requires`; `package_data` with `torch/` and `data/`; metadata in `pyproject.toml`; distribution name `pyfwi-cuda` with the `PyFWI` import; `python_requires >= 3.10`.
- **Tests:** PKG (wheel in a clean venv, every module imports with NumPy 2.5 and SciPy 1.18, files present); UNI of `MemoizeJac` (same behaviour as the original of SciPy 1.7.3: caches `fun` and `jac`); CAR, now including the FWI cases; EXA, now including the FWI examples in dev; the import and `segyio` `xfail`s start passing.
- **Criterion:** every module imports in dev; CAR and EXA approved.
- **Result:** done in five commits: the NumPy 2 imports (`aca0e30`, made during Phase 0 for the
  examples), the PML profile in float64 (`32422f9`), SciPy compatibility (`94859e0`), optional
  `segyio` (`fc7f969`) and the `pyfwi-cuda` packaging (`6c2c387`). Every module imports with
  NumPy 2.5 and SciPy 1.18, also from the wheel installed in a fresh environment
  (`tools/check_install.py`: kernels, `data/` and `PyFWI.torch` present, the `dupuy` model reads
  its data); the wheel and the source distribution
  contain every file of `src/PyFWI`. CAR: in dev, the 16 forward + gradient cases, the 10 inversions
  and the examples match the golden masters bit for bit (float64 optimiser outputs within 1e-12,
  because SciPy's L-BFGS-B changed implementation after 1.7.3); the slow suite passes (21 tests).
  EXA: every example script runs in dev, FWI ones included. Six known-bug tests now pass in dev
  (PML profile, optimiser imports, NumPy 2 imports, `segyio`, `pyopencl` dependency, wheel content);
  those whose bug shows in the reference environment (`segyio`, `pyopencl`) are still expected
  failures there.

### 1.2 Device selection

- **Goes in:** explicit platform and device choice without `create_some_context`, keeping the meaning of `inpa['platform']` and `inpa['device']`.
- **Tests:** UNI with simulated platforms (one, several and no device); the single-device `xfail` passes; API (the same call as in 0.1.10 picks the same device).
- **Criterion:** works with one or several devices; the old behaviour is kept where it already worked.
- **Result:** `PyFWI._opencl_device.choose_devices` replaces `create_some_context`. On simulated
  platforms (one device, two devices, none, a platform without devices) it gives the same choice as
  PyOpenCL 2026's own `choose_devices` for 17 values of `PYOPENCL_CTX`, and an error where PyOpenCL
  fails; it never prompts. On the real platforms (NVIDIA, POCL), the default and `PYOPENCL_CTX`
  select the same device in dev and with the published 0.1.10, and `inpa['device']` works on both
  single-device platforms. Behaviour changes, all where 0.1.10 failed or ignored the input:
  single-device platforms, the range check against the right platform, `inpa['platform']` alone, no
  prompt (the default answer is taken).

### 1.3 Cached kernels and fixed arguments

- **Goes in:** each kernel created once per `WavePropagator`; constant arguments set with `set_args`.
- **Tests:** BIT; a test that turns `RepeatedKernelRetrieval` into an error; DES of host time per launch (target measured in the analysis: from 168 µs down to 4–32 µs).
- **Criterion:** BIT; no warning; difference between PyOpenCL 2022 and 2026 removed.
- **Result:** BIT holds (characterisation in dev, bit for bit); `test/test_kernel_launches.py` checks
  that no kernel is retrieved and no `RepeatedKernelRetrieval` warning is emitted during forward and
  gradient, that the time loops only enqueue (all three fail against the code before the step), and
  that a propagator reused for several runs gives the results of a fresh one. DES
  ([report](reports/step-1.3-kernel-launches.md)): small scenario 3.0x faster forward and 2.4x faster
  gradient than 0.1.10; wall time per launch from 118 µs to 28 µs, of which 21 µs are GPU work (GPU
  busy 65 % against 28 %); PyOpenCL 2022 and 2026 within 6 %. Medium scenario: gradient 10 % faster,
  forward unchanged, as its kernels take about 200 µs each and the blocking copies (steps 1.4 and
  1.5) dominate the remaining host time.

### 1.4 Residual and wavelet on the GPU; non-blocking copies

- **Goes in:** the `[nt, nr, 5]` residual and the wavelet are uploaded once, and the kernel reads at position `t` (only the indexing changes, not the arithmetic); copies with `is_blocking=False` and explicit events.
- **Tests:** BIT; DES (blocking-copy counter: −750 per shot in the docs example); DET.
- **Criterion:** BIT; copy count as targeted.
- **Result:** the residual of each shot goes to the device once (5 copies per shot), and each time
  sample is copied on the device into the small buffers `Adj_injSrc` has always read. Reading the
  big buffer directly in the kernel, as first planned, would change results: the unbounded receiver
  loops of `Adj_injSrc` read past the end of the residual (known bug, step 2.4), and would then read
  the next time sample instead of what lies after the small buffer. The seismograms of a shot are
  read with non-blocking copies and one wait. The wavelet stays a kernel argument: since step 1.3
  its samples are set with `set_arg` and involve no copy, so moving it to the device only pays off
  with CUDA Graphs (Phase 3), where the time index must come from device memory. BIT holds; blocking
  copies per shot in the gradient: from `5 × restored checkpoints + 5 × injections` to
  `5 × restored checkpoints + 5` (`test_benchmark.py` checks both counts, for 0.1.10 and the fork).
  DES, small scenario (`bench/step-1.4/`): blocking copies per step from 2.71 to 2.00; gradient
  0.81 s to 0.79 s, forward unchanged (0.49 s). The gain is small because after step 1.3 the GPU
  work dominates and the residual was the smallest share of the copies; the checkpoint copies (one
  per step in each direction) are next, in step 1.5.

### 1.5 Contiguous checkpoints

- **Goes in:** `(ns, nchp, nz, nx)` layout; `W.W` becomes a property that returns a view with the same `[:, :, s, c]` axis order.
- **Tests:** API (shape, `dtype`, values and axis order of `W.W` equal to 0.1.10; the gradient example of the docs runs); BIT; DES (≈50× per copy at 400×400); MEM (peak RAM does not grow).
- **Criterion:** BIT; API intact.
- **Result:** checkpoints stored as `(ns, nchp, tnz, tnx)` and copied straight from and to the device
  without blocking; `W.W` is a property (with a setter, which `PyFWI.optimization` uses) that shows
  them as `(tnz, tnx, ns, nchp)` views. API (`test/test_checkpoints.py`, also run against 0.1.10):
  same shape, dtype, axis order and values, writing into `W.W` and assigning it change the gradient
  as in 0.1.10. BIT holds. MEM: peak traced memory of forward + gradient unchanged (1.03 MB in a case
  with 50 MB of checkpoints, before and after). DES (`bench/step-1.5/`): one copy of a 440 × 440
  field takes 110 µs instead of 368 µs (device to host) and 107 µs instead of 271 µs (host to
  device), i.e. 3× and not the ≈50× of the analysis, because here the PCIe transfer itself
  (≈7 GB/s from pageable memory) dominates once the strided scatter is gone. The larger gain is that
  the host no longer waits: blocking copies per step fall from 2.00 to 0.01, the GPU is busy 88 %
  of the time in the medium scenario (53 % before) and 73 % in the small one. Medium scenario:
  forward 47.9 s to 17.8 s (2.7× faster than 0.1.10), gradient 52.1 s to 38.6 s (1.5×); small
  scenario: 0.41 s and 0.76 s (3.6× and 2.5× faster than 0.1.10).

### 1.6 Element-wise low-pass filter

- **Goes in:** `lowpass` and `adj_lowpass` with `|h|²·X` instead of `np.diag(h) @ X`; observed data filtered once per band.
- **Tests:** UNI with relative difference ≤ 1e-12 against the old version (kept in the test as a reference), for several `nt`, numbers of traces, orders and axes; DES (65 to 75×); 2-iteration FWI within the tolerances; the adjoint identity ⟨lowpass x, y⟩ = ⟨x, adj_lowpass y⟩ still holds.
- **Criterion:** tolerances met; gain measured.
- **Result:** `|h|²·X` element by element, with the real FFT of `scipy.fft` on every core and the
  zero padding done by the FFT (`test/test_lowpass.py` keeps the 0.1.10 versions as reference).
  UNI: largest relative difference ≤ 1e-12 (measured 5e-16 to 7e-16) for float32 and float64 data,
  four sizes, orders 1 to 5; the adjoint identity holds to 1e-12; the filter now also works along
  other axes. FWI: the 10 inversions and the examples still match the golden masters (float32
  outputs bit for bit, float64 within 1e-12). DES, filtering per FWI iteration (observed and
  predicted data, and the adjoint of the residual): 0.48 s to 0.022 s (22×) with the data of the
  small scenario, 5.46 s to 0.48 s (11×) with those of the medium one. Element-wise products alone
  gave only 3 to 4× (the complex FFT of NumPy, on one core, then dominated): hence the real FFT on
  every core; the 65–75× of the analysis compared the matrix product with the product alone.
  Filtering the observed data once per band is not done: it would save 7 ms per iteration in the
  small scenario and 0.16 s of about 56 s in the medium one.

### 1.7 Padded buffers

- **Goes in:** allocations with zeroed margins before and after every buffer (fields, seismograms and residuals), aligned to `CL_DEVICE_MEM_BASE_ADDR_ALIGN` and used through sub-buffers, so that the out-of-bounds reads and writes of the stencils and of the unbounded receiver loops fall inside the allocation. This also removes the `CL_OUT_OF_RESOURCES` crashes, which raises the priority of this step.
- **Deciding check:** on NVIDIA, is the result with padding bit-for-bit equal to the golden master? If so, the change stays on the host, in the `legacy` mode. If not, it changes the arithmetic and moves to the `fixed` mode (step 2.4), recorded as such at G1.
- **Tests:** BIT on NVIDIA and on POCL against the golden masters, now **without** the guard bands (the padding replaces them); the out-of-bounds `xfail`s start passing; plain runs no longer crash after a few propagations in one process, and the example scripts run as programs without the guard (`gradient.py` crashes plain 0.1.10 today).
- **Criterion:** POCL finite and deterministic; decision about the mode recorded.
- **Result (padding):** every buffer is a sub-buffer of `[margin | data | margin]`; the margin is
  the largest of 64 KiB (the guard bands of the golden masters), `rec_top_right_var + 8` grid rows
  (the unbounded right-well loop reads up to `rec_top_right_var` rows past a field) and `tnz + 8`
  values (seismograms and residuals), rounded to the base-address alignment. **Deciding check:** plain
  runs (no guard bands) reproduce the 16 propagation golden masters bit for bit on NVIDIA and on POCL,
  so the change stays on the host, in the `legacy` mode. With guard bands placed outside the margins
  and no 64 KiB minimum, no read (NaN bands) and no write (1.0 bands) goes beyond them, for the three
  acquisitions and both orders; with margins one row too small the test fails. Found on the way: an
  allocation created with `COPY_HOST_PTR` is placed on the NVIDIA device only for the sub-buffer
  region the kernels use, so the margins held whatever the device memory held; each allocation is
  now created empty and written whole by an explicit copy. Created that way it is also not wrapped by
  the guard bands of the tests, which would nest overlapping sub-buffers in it: undefined in OpenCL,
  and a segmentation fault of the NVIDIA driver when they are released (reproduced in isolation). The
  out-of-bounds known-bug tests now fill PyFWI's own margins (NaN or 1.0) and
  stay expected failures: the kernels still access outside their logical buffers until the `fixed`
  mode of step 2.4.
- **Result (crash):** the padding did **not** remove the `CL_OUT_OF_RESOURCES` crash, and the
  experiments located its real cause, which Phase 0 had attributed to the out-of-bounds accesses: a
  loop that builds a new propagator while the previous one is still referenced fails at the fourth
  (with 0.1.10, and with the fork before and after the padding); building each one after the
  previous is released, or keeping only the old contexts alive, never fails. Releasing an OpenCL
  context while another exists on the same device breaks the NVIDIA driver. The guard bands avoided
  it only because they keep a command queue, hence the context, of every propagator. The fork now
  reuses one context per set of devices (`shared_context`); the loop runs (`test/test_contexts.py`,
  an expected failure for 0.1.10). With both changes the characterisation and the examples run
  **without** guard bands in dev, on NVIDIA and on POCL; only the reference environment still uses
  them.

### 1.8 Diagnostics

- **Goes in:** `PyFWI.info()`, with the package origin (fork and commit), backend, device, versions and numeric mode; the same summary goes to the log on the first propagation.
- **Tests:** UNI (fields present; works without a GPU); API (`import PyFWI` still has no side effects).
- **Result:** `PyFWI.info()` returns a dict shown as a readable summary (`PyFWI._info`): version and
  origin, source folder and git commit (only for the `src/PyFWI` of a checkout, `-dirty` with local
  changes), backend `opencl`, numeric mode `legacy`, the device chosen as by a propagator (optionally
  with `inpa`-like `platform` and `device`), every OpenCL platform and device, and the versions of
  Python, NumPy, SciPy, PyOpenCL, matplotlib, hdf5storage, segyio and torch. Without OpenCL it reports
  the reason instead of failing. The summary is logged once per process at INFO, at the first forward
  modelling while INFO is enabled. `test/test_info.py`: fields, commit (and none for a folder of
  another repository), device for each platform, no OpenCL, logging once, and an `import PyFWI` that
  loads no heavy module, adds no logging handler and changes no environment variable (also checked
  against 0.1.10).

### 1.9 Fork release 0.1.11

- **Goes in:** CHANGELOG, version and tag.
- **Tests:** full PKG (Python 3.10 to 3.13); the original scripts in `example/` and every script in `example/scripts/` run without edits; DES against the baseline; FWI.
- **Criterion:** gate G1.
- **Result:** version 0.1.11, tag `v0.1.11`. PKG: the wheel installs in fresh environments of Python
  3.10, 3.11, 3.12 and 3.13 from the package index and runs an example on POCL; the original
  `example/fwi_example_surface.py` and every script of `example/scripts/` run without edits
  (`test/test_release.py`, `test/test_examples.py`). DES ([report](reports/benchmark-0.1.11.md)): no
  scenario slower than the baseline; forward / gradient 3.5× / 2.5× (small), 2.6× / 1.5× (medium),
  1.5× / 1.14× (large). FWI: the inversions of the golden masters bit for bit. Gate report:
  [G1](gates/G1.md).

### Gate G1

- [ ] CAR and BIT on the whole matrix, on NVIDIA (and on POCL after 1.7).
- [ ] EXA: every example runs and matches 0.1.10.
- [ ] Gains measured and recorded; no scenario slower than the baseline.
- [ ] Installs and imports with NumPy 2 and current SciPy.
- [ ] MAT not worse.
- [ ] 2-iteration FWI within tolerance.
- [ ] Fixed bugs moved from `xfail` to normal tests.

## Phase 2: parity CUDA backend (version 0.2.0)

CUDA comes in as an option; the default stays `opencl`.

### 2.1 Engine and backend interface

- **Goes in:** `PyFWI/engine/` (time loop, checkpoint policy, shot batching) and `PyFWI/backends/opencl.py`, with the interface allocate / copy / compile with `#define` / launch / synchronise. `WavePropagator` becomes a facade over `self._engine`.
- **Tests:** BIT (pure refactoring); API; EXA; UNI of the interface with a fake NumPy backend, which lets the engine be tested without a device.
- **Criterion:** BIT; engine coverage above 90% with UNI alone.
- **Result:** `PyFWI/backends/` (interface documented in its `__init__`, `opencl.py`) and
  `PyFWI/engine/propagation.py` (`PropagationEngine`: buffers with margins, kernels, forward loop,
  checkpoints, gradient loop, residual; `checkpoint_steps` and `margin_bytes`). `WavePropagator`
  keeps the inputs, the plots and the post-processing, and shows the engine's attributes as its own.
  Shot batching is left to Phase 3, where the CUDA kernels can use it. BIT: the characterisation
  passes bit for bit on NVIDIA and POCL. UNI: `test/test_engine.py` runs the engine on a fake NumPy
  backend whose toy kernels have predictable effects (`test/fake_backend.py`) and checks the
  recording, the checkpoints, the backward reconstruction, the residual injection, the order of the
  launches and the copies; engine coverage with these tests alone: 100 %; mutations of the restore,
  of the order of the backward updates and of the residual sample are caught.

### 2.2 Backend selector and fallback

- **Goes in:** precedence `inpa['backend']` → `PYFWI_BACKEND` (`opencl`, `cuda`, `auto`, `cuda:strict`) → default `opencl`; fallback with the reason in the log; matching between OpenCL device and CUDA ordinal by PCI bus ID; a non-NVIDIA OpenCL device keeps `auto` on OpenCL.
- **Tests:** SEL with every branch: no CuPy (import blocked by monkeypatch), no GPU, simulated old driver, `cuda:strict` failing with a clear message, and `inpa` pointing to a CPU or to NVIDIA.
- **Criterion:** every branch covered and tested in the CPU CI.
- **Result:** `PyFWI/backends/selection.py`: `requested_backend` (precedence and validation),
  `probe_cuda` (CuPy, GPU count, driver against the CUDA version of CuPy, CUDA backend present;
  never raises), `opencl_bus_id` (`cl_nv_device_attribute_query`) and `select_backend`. `cuda`
  without an explicit device takes CUDA device 0; with `inpa` naming a device, or with `auto`, the
  CUDA device is the one with the PCI bus ID of the chosen OpenCL device, and a CPU or non-NVIDIA
  device keeps OpenCL. Fallbacks are logged (`cuda`: warning, `auto`: info). Until step 2.3 the probe
  reports that this PyFWI has no CUDA backend, so everything runs on OpenCL. `test/test_backend_selection.py`
  simulates CuPy (missing, no GPU, old driver, broken), GPUs and OpenCL devices; branch coverage of the
  selector without a GPU: 100 % (a test requires it); on the RTX the OpenCL device and CUDA device 0
  have the same PCI bus ID (`0000:04:00.0`).

### 2.3 Parity CUDA backend

- **Goes in:** `PyFWI/kernels/` with the portability header; `PyFWI/backends/cuda.py` with `RawModule` compiling the same `.cl` files in `legacy` mode, always to cubin for the device architecture; a bounds test (`if (i >= Nz || j >= Nx) return;`) in the kernels without one; the buffer padding of step 1.7.
- **Tests:**
  - PAR on the whole matrix;
  - DET;
  - SAN with zero errors (the padding of 1.7 keeps the inherited reads inside the allocation);
  - a test that the module loads with an NVRTC newer than the driver;
  - a grid that is not a multiple of the block (61×67, 100×100 and 127×129 grids);
  - EXA with `PYFWI_BACKEND=cuda`.
- **Criterion:** relative L2 ≤ 1e-6 against OpenCL on the same GPU; final tolerance set at G2.
- **Result:** `PyFWI/kernels/opencl_on_cuda.h` (also read by `tools/check_env.py`) and
  `PyFWI/backends/cuda.py` (`portable_source`, `CUDABackend` on CuPy). The `.cl` files are unchanged;
  the bounds guard is inserted at compile time and tests the *global ids*
  (`if (get_global_id(0) >= Nz || get_global_id(1) >= Nx) return;`), not `i` and `j`: several kernels
  compute `i = get_global_id(0) + sdo`, and the guard must keep exactly the work items of the OpenCL
  launch. Blocks of 32 columns × 8 rows, the column on `threadIdx.x` (coalesced). NVRTC 13.4 builds
  cubin for `sm_86`, which loads with the 13.0 driver. PAR (`test/test_parity.py`): not bit for bit;
  on the 16 propagation cases the seismograms differ by at most 1.1e-6 and the gradients by 1.9e-5
  (relative L2), the order of the effect of one bit of the PML profile (step 0.5); compiler options
  (`--fmad=false`, approximate division) do not bring them closer. The initial target of 1e-6 is met
  by the seismograms, not by the gradients: the tests use 3e-6 and 6e-5, and G2 sets the final
  tolerance. Grids of 61×67, 100×100 and 127×129 within the same tolerances; DET: CUDA runs identical
  bit for bit; SAN: `compute-sanitizer` memcheck reports 0 errors for the three acquisitions (47
  invalid reads without the margins); EXA: every example passes its checks with
  `PYFWI_BACKEND=cuda:strict`.

### 2.4 Numeric modes

- **Goes in:** `PYFWI_NUMERICS=legacy|fixed`. The `fixed` mode brings literals with the `f` suffix, `dt` with 9 digits, correct loop bounds (no reads outside the domain), correct `vz` in the well and the padding, if step 1.7 sent it here. CUDA starts in `fixed`; OpenCL stays in `legacy`.
- **Tests:**
  - PRE: zero FP64 in the hot kernels in `fixed`, in the CPU CI through NVRTC; the FP64 characterisation test of `legacy` stays;
  - the `dt`, `vz` and out-of-buffer `xfail`s pass in `fixed` and keep failing in `legacy`;
  - `fixed` × `legacy` differences measured on the whole matrix and published in the CHANGELOG;
  - MAT in `fixed` (the adjoint dot product must improve or stay the same);
  - SAN.
- **Criterion:** PRE zero; differences published; MAT not worse.

### 2.5 API contract on CUDA

- **Goes in:** `W.W` and `W.seismogram` copied from the GPU only when accessed; the `show=True` path; `torchfwi.Fwi` with CPU tensors.
- **Tests:** API with `PYFWI_BACKEND=cuda`, the same tests as OpenCL parametrised by backend; the original example scripts without edits; EXA; INT on the CPU.
- **Criterion:** the same API suite passes on both backends.

### 2.6 Installation extras

- **Goes in:** extras `[cuda13]` (`cupy-cuda13x`) and `[cuda12]` (`cupy-cuda12x`); `envs/cuda12` in `setup_envs.sh`.
- **Tests:** PKG for each extra in a clean venv; fast PAR in `envs/cuda12`; `check_env.py` passing in the three environments.
- **Criterion:** both extras install and pass PAR.

### 2.7 Parity report and release 0.2.0

- **Goes in:** `tools/parity_report.py`, which writes `docs/cuda-migration/reports/parity-0.2.0.md` with the error per case and per field; CHANGELOG; tag.
- **Tests:** full PAR; 2-iteration FWI CUDA × OpenCL; EXA on both backends; DES (parity CUDA cannot be slower than OpenCL with cached kernels); PKG.

### Gate G2

- [ ] PAR on the whole matrix, with the final tolerance recorded.
- [ ] SAN zero; PRE zero in `fixed`.
- [ ] Full SEL; API and EXA parametrised on both backends.
- [ ] FWI between backends within tolerance.
- [ ] `cuda13` and `cuda12` extras working.

## Phase 3: optimised CUDA kernels (version 0.3.0)

Each optimisation is a separate commit, with a benchmark before and after, so that a regression points to the guilty change. **Tests in every step:** OTM against the `fixed` parity, DET, SAN, PRE, API, EXA and DES before and after, with the numbers in the commit message.

| Step | Optimisation | Specific tests | Target |
| --- | --- | --- | --- |
| 3.1 | Coalesced mapping (`threadIdx.x` on `j`) and block size tunable per GPU (32×8, 64×4…) with an autotune table | OTM with several block shapes; grids that are not multiples of the block | To be measured (analysis) |
| 3.2 | Order as `template<int ORDER>`, no reads of `c3 = c4 = 0` and correct bounds | PRE; SAN; OTM in orders 4 and 8 | Fewer reads in order 4 |
| 3.3 | Precomputed λ + 2μ and 1/dx | OTM; UNI of the precomputation | Fewer reads and divisions |
| 3.4 | 1D injection and recording (`nr` threads, coordinate table) | OTM in `acq_type` 0/1/2; test with receivers at the border | −2 whole-grid kernels per step |
| 3.5 | Fusion: `update_v` and `update_tau` with injection and recording; adjoint fused with imaging | OTM; MAT (dot product) | 4 → 2 kernels per step in the forward |
| 3.6 | Time on the device: a counter incremented by a minimal kernel; source and residual read at `t` | OTM bit for bit against the previous step | Prerequisite of 3.7 |
| 3.7 | CUDA Graphs with K steps per graph; `show=True` outside the graph | OTM bit for bit with and without a graph; API with `show=True`; DES of host time per step; PRF (idle gaps) | A few µs per kernel |
| 3.8 | Checkpoints on the GPU when they fit; budget GPU → asynchronous pinned host → recomputation | MEM with a small forced budget: result bit-for-bit equal to the unlimited budget; EST close to the 12 GB limit | No VRAM overflow |
| 3.9 | Batched shots (3D grid x, z, shot); partial gradients summed in a fixed order at the end | DET (critical); OTM with B = 1, 2 and 4 | B times fewer launches |
| 3.10 | PML only at the borders: interior blocks skip the damping | OTM; MAT (PML reflection) | −2 reads per interior cell |

### 3.11 `auto` default and release 0.3.0

- **Goes in:** the default becomes `auto`. Whether OpenCL also switches to `fixed` by default in this version is decided and recorded at G3.
- **Tests:** SEL of the new default; PKG; EXA; FWI; full DES; PRF; EST.

### Gate G3

- [ ] Performance targets against the 0.1.10 baseline (estimates of the analysis): 20 to 100× in the small scenario, 10 to 40× in the medium one and 3 to 10× in the large one. Met, or the gap explained.
- [ ] No regression above 5% between the commits of the phase.
- [ ] OTM, DET, SAN, PRE and EXA on the whole matrix.
- [ ] PRF: Nsight Systems timeline without relevant idle gaps in the small scenario; bandwidth per kernel from Nsight Compute recorded, with the hot kernels close to the memory ceiling of the RTX 3060.
- [ ] EST close to the VRAM limit.

## Phase 4: opt-in features (version 0.4.0)

Everything here is optional and off by default. Each feature has its own tolerance, validated by a complete FWI, not only by a seismogram.

### 4.1 Cost function and smoothing on the GPU

- **Goes in:** `cupy.fft` and `cupyx.scipy.ndimage.gaussian_filter`; no more repeated `deepcopy`.
- **Tests:** UNI (cost and adjoint residual against the NumPy version, ≤ 1e-6); FWI; EXA; DES (the CPU leaves the critical path, checked with PRF).

### 4.2 Boundary saving

- **Goes in:** at every step, only the `sdo`-cell band next to the PML is stored, and the interior is reconstructed.
- **Tests:**
  - gradient against the checkpoint path (≤ 1e-4);
  - MEM: memory proportional to the perimeter, checked on three grid sizes;
  - a long run without reconstruction drift (EST);
  - complete FWI.

### 4.3 FP16 mode

- **Goes in:** half-precision fields with scaling, as in SeisCL.
- **Tests:**
  - overflow and underflow with extreme amplitudes;
  - seismogram SNR against FP32;
  - complete FWI with a final model error within the target (proposed: at most 10% above FP32);
  - DES (up to ~2×) and MEM (half).

### 4.4 Multi-GPU by shots

- **Goes in:** one process or stream per GPU; gradients summed on the host in a fixed order.
- **Tests:** with a single RTX, two processes on the same GPU (`CUDA_VISIBLE_DEVICES`) must give a result bit-for-bit equal to one process (DET); a failing process propagates a clear error.
- **Criterion:** almost linear scaling on a machine with more GPUs, if one is available; otherwise only correctness is checked, and that is recorded at the gate.

### 4.5 `torchfwi` with GPU tensors

- **Goes in:** DLPack without copies, in `envs/torch`.
- **Tests:** INT (same `data_ptr`; results equal to the NumPy path; gradient × finite differences; CPU tensors keep working); the PyTorch example of 0.3 with GPU tensors.

### 4.6 Geophysical validation

- **Goes in:** reference cases of multi-scale FWI and of every `TimeLapse` strategy, on both backends.
- **Tests:** FWI; review by the geophysicist with the figures and misfit curves in `docs/cuda-migration/reports/`.

### 4.7 Release 0.4.0

PKG, EXA, FWI, DES, PRF, EST and a CHANGELOG with the tolerances of every opt-in feature.

### Gate G4

- [ ] Every opt-in feature with a documented tolerance validated by a complete FWI.
- [ ] Defaults unchanged (API and EXA tests pass without any environment variable).
- [ ] Memory and time gains measured.
- [ ] Geophysical review recorded.

## Wrap-up and maintenance

| Step | What is done | Check |
| --- | --- | --- |
| 5.1 | Fork documentation: installation, backends, numeric modes, tolerances; notebooks of `example/` updated | DOC: notebooks executed in reduced size in CI |
| 5.2 | Periodic merge of `upstream` | Full suite after each merge; `check_env.py` passing |
| 5.3 | Bug fixes sent back to the original PyFWI by PR | Only with the owner's approval; each PR with the test that reproduces the bug |
| 5.4 | Decide publication: public repository, name on PyPI | Owner's decision; before that, full PKG and licence review (CuPy MIT and cuda.core Apache-2.0, compatible with GPLv3) |
| 5.5 | Conditional optimisations: tiling, temporal blocking, whole loop in a graph with conditional nodes | Only if Nsight Compute shows kernels bound by memory bandwidth; same protocol as Phase 3 |

## Pending actions for the project owner

1. **Allow Nsight Compute to read GPU counters** (before step 0.8). Needs sudo and a reboot:

   ```bash
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
   sudo update-initramfs -u && sudo reboot
   # afterwards: grep RmProfilingAdminOnly /proc/driver/nvidia/params   → must show 0
   ```

2. ~~**CUDA Toolkit 13.x system-wide**~~: not needed; the self-test of step 0.8 showed that `compute-sanitizer` 2024.1 works with the CUDA 13 runtime.
3. **Token for the self-hosted runner** (step 0.9), generated in the repository under *Settings → Actions → Runners*.
4. **An RTX of another generation**, if possible, so as not to tune the code to a single architecture (Phase 3).
5. **Support from a geophysicist** to validate FWI and TL-FWI (Phases 0 and 4).
6. **Future decisions:** final distribution name, when to make the repository public and whether to send fixes back to the original.

## Reference commands

```bash
cd ~/Desktop/apps/cuda_pyfwi/pyfwi-cuda
tools/setup_envs.sh all                                  # re-creates ../envs/ref-0.1.10 and ../envs/dev
../envs/dev/bin/python tools/check_env.py                # full environment check
../envs/dev/bin/python -m pytest -m "not slow"           # fast suite (before every commit)
../envs/dev/bin/python -m pytest -m "opencl or gpu"      # tests that need the RTX
../envs/ref-0.1.10/bin/python -m pytest                  # what runs in the reference environment
```
