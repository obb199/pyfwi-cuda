# PyFWI: GPU core and migration plan to CUDA

Oct 24, 2026 · Gabriel Carvalho · English translation of the original analysis (Portuguese), Oct 25, 2026

## Executive summary

Yes: moving the PyFWI core to CUDA-based tools makes it much faster on NVIDIA cards, but the gain comes from restructuring the orchestration, not from swapping the API. In version 0.1.10, most of the time of an FWI evaluation goes to Python, synchronisations and a CPU filter, not to the GPU.

- **Where the time goes today.** One forward + gradient of the documentation example (100×100, 4 shots, 1,051 steps) issues ≈44 thousand kernel launches and ≈11.4 thousand blocking copies. Almost every launch re-creates the kernel object in PyOpenCL: 168 µs of CPU per launch, against 4 µs when the kernel and its arguments are reused (measured on the test machine).
- **Slow kernels on GeForce RTX, the target of the fork.** Constants without the `f` suffix make the hot kernels compute in FP64 (42 to 131 FP64 instructions per kernel in the PTX), and on RTX cards FP64 runs at a tiny fraction of the FP32 rate (1/64 on Ampere). The index mapping also tends to produce non-coalesced accesses.
- **Bottleneck outside the GPU.** The low-pass filter of the cost function multiplies by a dense diagonal matrix (`np.diag(h) @ X`); the element-wise product gives the same result 65–75× faster.
- **Recommended tool.** CuPy 14 (C++ kernels compiled at run time by NVRTC, CUDA Graphs, cuFFT), with cuda.core as a complement to move the whole loop to the GPU. The three current `.cl` files compile as CUDA with a ~10-line header, from sm\_75 to sm\_120.
- **Expected gain (estimate, to be confirmed in Phase 0).** 20–100× on small 2D models, 10–40× on medium ones and 3–10× on large ones; the optional FP16 mode may add up to ~2×.
- **Controlled change, in a fork.** The public API (`WavePropagator`, `FWI`, `TimeLapse`, `torchfwi.Fwi`, `W.W`) does not change, and the fork keeps the `PyFWI` import package: scripts only change the install command. The backend is chosen by `PYFWI_BACKEND` or by an optional key in `inpa`, with OpenCL as the default until parity is proven, and automatic fallback.

Next step: Phase 0 (characterisation tests and a reference benchmark on an NVIDIA GPU) and Phase 1 (host fixes that already speed up OpenCL) before porting any kernel.

## PyFWI today

The latest version is **0.1.10**, published on PyPI on Nov 5, 2024; GitHub `main` points to the same commit (`1434a16`) and there has been no release since. The package does 2D elastic time-domain modelling and FWI, with the core in OpenCL through PyOpenCL ([PyPI](https://pypi.org/project/PyFWI/), [GitHub](https://github.com/AmirMardan/PyFWI), [SoftwareX 2023 paper](https://espace.inrs.ca/id/eprint/13791/), [documentation](https://pyfwi.readthedocs.io/)).

| Item | Status in 0.1.10 |
| --- | --- |
| Physics and grid | 2D elastic (vx, vz, τxx, τzz, τxz), staggered-grid finite differences, order 4 or 8 in space and 2 in time, damping PML |
| Internal time step | Limited to 0.6 ms: the user `dt` is split into `dt_scale` sub-steps |
| Gradient | Adjoint-state method; the forward field is reconstructed backwards and re-synchronised at checkpoints (`chpr`, % of the steps) |
| Acquisition | Crosswell (0), surface (1) or both (2); receiver positions fixed in the kernels through `#define` |
| Inversion | SciPy CG and L-BFGS, multi-scale in frequency, 7 TL-FWI strategies, TV/Tikhonov regularisation, `torchfwi.Fwi` for PyTorch |
| Code | 15 Python files (~6,500 lines) + 3 OpenCL files (675 lines); GPLv3+ licence |
| Tests | 1 test (Ricker wavelet); nothing covers the GPU core |

Packaging problems the migration must solve along the way:

- `pyopencl` is not in `install_requires`, only in `build-system.requires`; `pip install PyFWI` does not install the GPU engine.
- `fwi.py`, `tl_fwi.py` and `optimization.py` import `MemoizeJac` from `scipy.optimize.optimize`, a name that current SciPy no longer exports (tested: `ImportError` on SciPy 1.18.1). `requirements.txt` pins SciPy 1.7.3 and the package declares Python ≥ 3.7.
- The published wheel does not include the experimental PyTorch acoustic propagator (`PyFWI/torch`) nor the `data` folder.

## Anatomy of the GPU core (OpenCL)

The core is a Python time loop that enqueues, at every step and shot by shot, several small kernels on a single in-order queue. In the documentation example (100×100 + 20-point PML, 4 shots, 4 ms `dt` → 1,051 internal steps, `chpr` = 20), one forward + gradient adds up to ≈44 thousand launches and ≈11.4 thousand blocking copies.

| Stage (per shot) | Kernels per step | Host↔GPU transfers |
| --- | --- | --- |
| Forward | 4 updates (`update_velx`, `update_velz`, `update_tauz`, `update_tauxz`) + `injSrc` every `dt_scale` steps | 5 blocking GPU→host copies per checkpoint + 5 seismograms at the end |
| Gradient | 4 reverse reconstructions + `Adj_update_tau`, `Adj_update_v`, `Grad` + `Adj_injSrc` every `dt_scale` steps | 5 blocking host→GPU residual copies every `dt_scale` steps and 5 per checkpoint |

**Orchestration (host) bottlenecks**

1. **Kernel re-created at every launch.** `self.prg.update_velx(...)` creates a new `cl.Kernel` per call; PyOpenCL 2026.1.4 emits the `RepeatedKernelRetrieval` warning precisely for this ([issue #831](https://github.com/inducer/pyopencl/issues/831)). Measured: 168 µs of CPU per launch, 32 µs with the kernel reused and 4 µs with the arguments set once.
2. **Synchronisation every few steps.** `cl.enqueue_copy` is blocking by default, and the residual is uploaded slice by slice inside the adjoint loop. Each copy drains the GPU queue and removes any CPU/GPU overlap.
3. **Checkpoints on the host, in a scattered layout.** `W[field][:, :, s, c]` in a `(tnz, tnx, ns, nchp)` array copies with a large memory stride: at 400×400, 0.84 ms per field against 0.017 ms in a contiguous layout (≈50×). Memory grows as `5 × ns × nchp × cells`: 15 GB of RAM for 400×400, 8 shots, 3,000 steps and `chpr` = 20.
4. **Point kernels launched over the whole grid.** `injSrc` and `Adj_injSrc` use `tnz × tnx` work-items, with integer remainders, to touch one source and a few hundred receivers.

**Problems inside the kernels**

1. **Unintended FP64.** The constants are literals without a suffix (`#define c1 1.125000`, `#define dt 0.000571`), which in C and OpenCL C are `double` when the device supports FP64. Compiling the same files with NVRTC 12.9, the PTX has from 42 (`update_velx`) to 131 (`Grad`) FP64 instructions per kernel; with the `f` suffix, zero. On GeForce, FP64 runs at 1/64 of the FP32 rate ([GA102 whitepaper](https://web.cs.miami.edu/home/burt/learning/csc596.231/docs/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)).
2. **Zero-coefficient terms are still computed.** In order 4, `c3 = c4 = 0`, but the reads and multiplications of those terms remain in the generated code, because strict IEEE semantics prevent the compiler from removing them.
3. **Out-of-buffer reads.** The updates go up to `i = Nz − sdo`, but the stencil reads up to `i + sdo`, row `Nz`, which does not exist; in order 4 it also reads `i − 3` with `i = 2`. The effect stays in the PML, but it is undefined behaviour.
4. **Poorly coalesced access.** Dimension 0 of the NDRange, the fastest-varying one within a warp, walks rows `i`, while memory is contiguous in `j`. With `local_size=None`, the result depends on the work-group shape the driver picks.
5. **Truncated `dt`.** The `#define dt` uses `%f` (6 decimals): 0.000571 instead of 0.000571428…, a step 0.075% shorter than the one used to sample the wavelet on the host.
6. **Well recording bug.** In `elastic_crosswell.cl`, the `vz` seismogram stores values of `vx`.

**Outside the propagation loop**, the largest cost is in the cost function: `lowpass` and `adj_lowpass` filter with `np.diag(h) @ X`, a dense `nt × nt` matrix. That multiplication alone costs 218 ms (nt 662, 400 traces) and 2.7 s (nt 1,512, 1,000 traces); the element-wise version gives the same result (relative difference 4×10⁻¹⁶) in 3.3 ms and 36 ms, and FWI calls the filter 3 times per evaluation.

Method: measurements with our own scripts that reproduce the PyFWI patterns, without running the package code, on a 2.1 GHz Xeon CPU (2 cores) with PyOpenCL 2026.1.4 on POCL and NVRTC 12.9; no NVIDIA GPU was available.

## Would CUDA make it faster?

Yes, as long as the migration attacks the orchestration; swapping only the API yields little. With equivalent kernels, CUDA was at most ~30% ahead of OpenCL in most of 16 benchmarks, a difference that vanishes in a fair comparison ([Fang et al., ICPP 2011](https://hgpu.org/?p=5748)). SeisCL, by two co-authors of PyFWI, reports OpenCL slightly slower on NVIDIA GPUs for large models and faster for small ones ([SeisCL](https://github.com/gfabieno/SeisCL)).

A 1:1 translation is not enough because the dominant cost today (re-creating kernels, synchronising, filtering with a dense matrix) does not depend on the API. A port to CuPy with one launch per kernel per step would reduce the per-launch cost, but the same cut exists in PyOpenCL with cached kernels.

What only the CUDA path offers, and what explains the large gain:

- **CUDA Graphs.** One time step, or a block of steps, becomes a graph launched at once. In NVIDIA's test (V100), the effective cost per kernel fell from 9.6 µs with synchronised launches to 3.4 µs with a graph, for a 2.9 µs kernel ([NVIDIA, 2019](https://developer.nvidia.com/blog/cuda-graphs/)). IF/WHILE conditional nodes (CUDA 12.4+) keep whole loops on the GPU without returning to the CPU ([NVIDIA, 2024](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes)). PyOpenCL 2026.1.4 does not expose `cl_khr_command_buffer`, the OpenCL equivalent (verified in the package code).
- **Mature half precision.** Elastic modelling in FP16 with CUDA was almost 2× faster than FP32, with half the memory and without harming FWI and RTM ([Fabien-Ouellet, Geophysics 2020](https://publications.polymtl.ca/6303/)); the SeisCL README lists the FP16 modes as an advantage of the CUDA backend.
- **Libraries and tools.** cuFFT (`cupy.fft`) and `cupyx.scipy.ndimage` take the cost function and the smoothing off the CPU; streams and pinned memory make copies asynchronous; Nsight Systems/Compute and `compute-sanitizer` profile and flag out-of-buffer accesses; DLPack exchanges tensors with PyTorch without copies.

| Scenario | Bottleneck in 0.1.10 | What fixes it | Estimated gain |
| --- | --- | --- | --- |
| Small: docs example (140×140 with PML, 4 shots, 1,051 steps) | Python and PyOpenCL overhead (estimated at more than 90% of the time) and dense filter | Cached kernels, graphs, batched shots, element-wise filter | 20–100× |
| Medium: ≈400×400, 8 shots, 3,000 steps | Launches, scattered copies, dense filter | The above + checkpoints on the GPU | 10–40× |
| Large: ≈450 thousand cells, 50 shots, 6,000 steps | Launches and poorly coalesced FP64 kernels | Coalesced FP32 kernels, fusion, graphs | 3–10× |
| Large, with the optional FP16 mode | Memory bandwidth | Storage and compute in half precision | up to ~2× more |

How the estimate was made: launches and copies counted in the code, per-launch cost measured on this machine and GPU time estimated from the memory bandwidth of a ~1 TB/s card. Phase 0 replaces these estimates with measurements on the target GPU.

## Tools of the CUDA ecosystem

The recommended base is **CuPy**, with C++ kernels compiled at run time by NVRTC: it is the shortest path from the current design, which already generates C kernels with `#define` at run time. The three PyFWI `.cl` files compiled as CUDA with NVRTC 12.9 and a ~10-line header, from sm\_75 (Turing) to sm\_120 (Blackwell), in orders 4 and 8. Versions checked on PyPI in Oct 2026:

| Tool | Use in PyFWI | Strengths | Limitations | Verdict |
| --- | --- | --- | --- | --- |
| [CuPy 14.2](https://pypi.org/project/cupy-cuda13x/) (MIT) | `RawModule` through NVRTC, capture with `Stream.begin_capture`/`end_capture` and `Graph.launch`, `cupy.fft`, `cupyx.scipy.ndimage` | Reuses the current kernels; NumPy-style arrays for the rest of the pipeline; the `[ctk]` extra installs CUDA through pip | Python overhead outside graphs; large dependency | **Recommended base** |
| [cuda.core 1.2](https://pypi.org/project/cuda-core/) (Apache-2.0) | `Program` (NVRTC), `launch`, `GraphBuilder` with conditional nodes | Official, lean NVIDIA API, access to the new graph features | No array library; still under active development | Complement for the whole loop on the GPU |
| [Warp 1.18](https://pypi.org/project/warp-lang/) (Apache-2.0) | Kernels written in Python, graph capture, tiles | Readable, differentiable kernels, interop with PyTorch and JAX ([docs](https://warp.readthedocs.io/)) | Rewrite every kernel; wheels need driver R580+ and an sm\_75+ GPU | Alternative for new physics |
| PyTorch 2.14 with a C++/CUDA extension, or Triton 3.8 | Custom operator and `torch.cuda.CUDAGraph`, as in [Deepwave](https://github.com/ar4/deepwave) | Native integration with `torchfwi` | Heavy dependency; build per CUDA and torch version; Triton is awkward for stencils with halos | Integrate through DLPack, not as the core |
| [Numba-CUDA 0.30](https://pypi.org/project/numba-cuda/) (BSD) | `@cuda.jit` | Kernels in Python | In maintenance mode: critical fixes only; new work moves to Numba-CUDA-MLIR | Not recommended |
| Own C++/CUDA extension (nanobind + nvcc) | Whole time loop in C++ | Performance ceiling | Wheel matrix OS × CUDA × architecture | Only if profiling shows residual overhead |
| JAX 0.12 | `lax.scan` compiled by XLA | Compiled loop and automatic differentiation | Full rewrite, little control over the kernels, high memory in autodiff | Not recommended |

GeForce RTX focus: the four RTX generations (series 20 to 50; sm\_75, sm\_86, sm\_89 and sm\_120) run with CUDA 13, which requires a 580-series driver or newer ([CUDA 13.0 notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)). Maxwell to Volta cards were dropped from CUDA 13 and are not a target of the fork ([NVIDIA, 2025](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)). The kernels compiled for the four RTX architectures in the NVRTC test; the `cuda13` extra becomes the fork's default, and `cuda12` is kept only for machines stuck with drivers older than 580.

## Target architecture

*(The original document has a diagram here: proposed architecture, 6 layers, 2 backends.)*

The user keeps calling the same classes; the facade forwards to a single engine, and the selector picks OpenCL or CUDA, with automatic fallback to OpenCL.

**Package layout (proposal)**

- `PyFWI/wave_propagation.py` keeps `WavePreparation` and `WavePropagator` as a facade and delegates to `self._engine`.
- `PyFWI/engine/` holds the time loop, checkpoint policy, shot batching and graph capture, without calling GPU APIs directly.
- `PyFWI/backends/opencl.py` and `PyFWI/backends/cuda.py` expose the same interface: allocate, copy, compile with `#define`, launch, synchronise and, in CUDA, capture a graph.
- `PyFWI/kernels/` keeps the portability header, the shared parity sources and the optimised `.cu` files, exclusive to CUDA.

**Single source for parity.** The header below, prepended to the current `.cl` files, was enough to compile the three files with NVRTC, and it already puts `j`, the contiguous direction in memory, on `threadIdx.x`. Kernels without a bounds test (`MakeAllZero`, `MakeGradZero`, `injSrc`, `Adj_injSrc`) get `if (i >= Nz || j >= Nx) return;`, because the CUDA grid is rounded up to multiples of the block.

```c
#define __kernel extern "C" __global__
#define __global
#define __local __shared__
__device__ __forceinline__ int get_global_id(int d) {
    return d == 0 ? (int)(blockIdx.y * blockDim.y + threadIdx.y)   // i (z)
                  : (int)(blockIdx.x * blockDim.x + threadIdx.x);  // j (x), coalesced
}
```

**Loop in a graph.** What changes at every step (time index, source sample, residual slice) is read from GPU memory, from a counter that a minimal kernel increments. The same graph then serves every step (sketch):

```python
s = cp.cuda.Stream(non_blocking=True)
with s:
    s.begin_capture()
    for _ in range(K):          # K steps per graph, e.g. dt_scale
        step_forward(state)     # fused kernels; t read from the device
    graph = s.end_capture()
for _ in range(nt // K):
    graph.launch(stream=s)      # 1 Python call every K steps
```

## Phased migration plan

*(The original document has a diagram here: proposed roadmap, 5 phases, 5 gates, 4 versions.)*

The first two phases do not touch CUDA: they build the safety net and already deliver most of the gain on small models. CUDA enters in Phase 2 as an option and only becomes the default after gate G3 proves the gain without regressions.

Details the diagram does not show:

- **Versions.** The roadmap numbers are the fork's, published under its own distribution name; the import stays `PyFWI`.
- **Phase 0.** The characterisation tests are generated with 0.1.10 itself in a frozen environment (Python 3.10 with SciPy 1.7.3, the version pinned in `requirements.txt`, which still exports `MemoizeJac`), so that the reference does not depend on any fix.
- **Phase 1.** All changes stay on the host and do not change any computation of the propagator; therefore the criterion is bit-for-bit equality, not a tolerance.
- **Phase 2.** The parity kernels use the same source as OpenCL and the `legacy` mode; bug fixes go into the `fixed` mode, with the measured difference published.
- **Phase 3.** Each optimisation goes into a separate PR, with a benchmark before and after, so that a regression points to the guilty change.
- **Phase 4.** Boundary saving, FP16 and multi-GPU are opt-in and come with their own tolerances, validated by a complete FWI and not only by a seismogram.

## Optimisations for maximum speed

*(The original document has a chart here: qualitative prioritisation, 20 optimisations in 4 quadrants.)*

The five items of the highlighted quadrant attack the bottleneck that dominates 0.1.10 and do not depend on CUDA; those of the right-hand quadrant are the ones only the NVIDIA path makes possible. Detail of each optimisation:

| Optimisation | Mechanism | Expected gain | Phase |
| --- | --- | --- | --- |
| Cached kernels | Create each kernel once and set the constant arguments | 168 → 4–32 µs of CPU per launch (measured) | 1 |
| Residual and source on the GPU | Upload the `[nt, nr, 5]` residual and the wavelet once; the kernel reads at position `t` | Removes 750 blocking copies per shot in the docs example | 1 |
| Contiguous checkpoints | `(ns, nchp, nz, nx)` layout on the GPU when it fits; `W.W` becomes a transposed view | ≈50× per copy at 400×400 (measured) | 1 and 3 |
| Element-wise filter | `\|h\|² · X` in the frequency domain; observed data filtered once per band | 65–75×, same result (measured) | 1 |
| Float literals and exact `dt` | `#define` generated with the `f` suffix and 9 digits | Up to ~2× on GeForce for large-grid kernels (estimated) | 2, `fixed` mode |
| Coalesced mapping | `threadIdx.x` on `j`; 32×8 or 64×4 blocks tuned per GPU | To be measured: depends on the group shape the driver uses today | 2 and 3 |
| CUDA Graphs | Capture K steps; time and samples read from the device | Launch cost falls to a few µs per kernel | 3 |
| Kernel fusion | `update_v` (vx, vz) and `update_tau` (τxx, τzz, τxz) with injection and recording inside; adjoint fused with imaging | 4 → 2 kernels per step in the forward and less memory traffic | 3 |
| Batched shots | 3D grid `(x, z, shot)`; partial gradients summed at the end | More parallelism on small grids and B times fewer launches | 3 |
| Order as a template | `template<int ORDER>`, no reads for `c3 = c4 = 0`, correct bounds | Fewer reads in order 4 and no more out-of-buffer reads | 2 and 3 |
| 1D injection and recording | Kernel with `nr` threads and a table of receiver coordinates | Removes 2 whole-grid kernels per step | 3 |
| Precomputed parameters | Store λ + 2μ and multiply by 1/dx instead of dividing by dx | Fewer reads and divisions per cell | 2 and 3 |
| PML only at the borders | Interior blocks skip the damping and the reads of dxpml and dzpml | Two fewer reads per interior cell | 3 |
| Boundary saving | Store at every step only the `sdo`-cell band next to the PML and reconstruct the interior exactly | Memory per shot proportional to the perimeter, not the area; no need for `chpr` | 4 |
| Optional FP16 | Half-precision fields with scaling, as in SeisCL | Up to ~2× and half the memory ([Fabien-Ouellet, 2020](https://publications.polymtl.ca/6303/)) | 4 |
| Multi-GPU by shots | One process or stream per GPU; gradients summed on the host | Almost linear in the number of GPUs | 4 |
| Cost function and smoothing on the GPU | `cupy.fft`, `cupyx.scipy.ndimage.gaussian_filter`, no repeated `deepcopy` | Takes the CPU off the critical path of each iteration | 4 |
| Tiling, temporal blocking, loop in the graph | Only if Nsight Compute shows kernels bound by memory bandwidth | To be measured | After 0.4 |

On GeForce RTX two rows weigh more than the rest: the float literals, because the FP64 of these cards runs at a tiny fraction of the FP32 rate, and the memory budget of the checkpoints, because RTX cards have 6 to 32 GB of VRAM depending on the model. That is why boundary saving and the FP16 mode move up in priority within Phase 4.

## Compatibility: controlled change without touching user code

The contract is simple: every script that runs on 0.1.10 runs the same way on the new versions, with the same imports, classes and outputs. Only who executes the kernels changes, and that is decided outside the user's code.

1. **Frozen API.** `WavePreparation`, `WavePropagator`, `FWI`, `TimeLapse` and `torchfwi.Fwi` keep their signatures, `inpa` keys and return values (dicts of `np.float32`, `'2d'`/`'3d'` formats, Fortran-order reshape). `W.W[field][:, :, s, c]`, used in the gradient example of the documentation, and `W.seismogram` become properties that return NumPy views in the same axis order, copied from the GPU only when accessed.
2. **Selection outside the code**, with this precedence: optional key `inpa['backend']` → variable `PYFWI_BACKEND` (`opencl`, `cuda`, `auto`, `cuda:strict`) → default. The default stays `opencl` until 0.2.x and becomes `auto` in 0.3.0.
3. **Safe fallback.** In `auto`, without CuPy, without an NVIDIA GPU or with an old driver, PyFWI logs the reason and uses OpenCL; `cuda:strict` fails instead of falling back, for those who need to guarantee the GPU.
4. **Device respected.** If `inpa['platform']` and `inpa['device']` point to an OpenCL device that is not NVIDIA (CPU, AMD, Intel), `auto` stays on OpenCL. If they point to an NVIDIA one, the selector uses the CUDA ordinal of the same card, matched by PCI bus ID.
5. **Explicit numeric modes.** `PYFWI_NUMERICS=legacy` reproduces 0.1.10 (`double` literals, 6-decimal `dt`, known bugs) to redo published results; `fixed` fixes everything. OpenCL stays `legacy` by default until 0.3.0; CUDA starts in `fixed`, with the measured differences published in the CHANGELOG.
6. **Installing the fork.** The fork ships under its own distribution name (e.g. `pyfwi-cuda`) or straight from Git, but keeps the `PyFWI` import package and the same API; scripts do not change, only the install command. Since both packages use the same import name, the original PyFWI must be uninstalled first. Extras: `[cuda13]` for RTX with driver 580 or newer and `[cuda12]` for older drivers; `pyopencl` becomes a declared dependency.
7. **Visualisation and PyTorch.** `show=True` keeps working: in that mode the loop leaves the graph and copies the field every 20 steps, as today. `torchfwi.Fwi` keeps accepting and returning CPU tensors and starts accepting GPU tensors through DLPack, without copies.
8. **Diagnostics and deprecation.** `PyFWI.info()` shows the package origin (fork and commit), backend, device, versions and numeric mode, and the same summary goes to the log on the first propagation. Nothing in the public API is removed; future changes go through a `DeprecationWarning` for at least two minor versions.

In practice, an existing script moves to the RTX without edits; only the installation changes:

```bash
pip uninstall -y PyFWI                             # the fork uses the same import name
pip install "pyfwi-cuda[cuda13]"                   # example name; the import is still PyFWI
PYFWI_BACKEND=cuda python fwi_example_surface.py   # same script as with 0.1.10
PYFWI_BACKEND=opencl python fwi_example_surface.py # back to the current behaviour
```

## Tests, numerical validation and benchmarks

No default changes without passing six layers of tests, running in CI with OpenCL on the CPU (POCL) and on a runner with an NVIDIA GPU. Today the repository has a single test, so Phase 0 starts here.

| Layer | What it checks | Acceptance criterion |
| --- | --- | --- |
| Characterisation (golden master) | 0.1.10 outputs stored in `.npz` for small models (e.g. 60×60, 300 steps): forward, gradient and 2 FWI iterations, covering `acq_type` 0/1/2, `components` 0–4, order 4/8, `energy_balancing`, `chpr` 0/20/100 and `parameterization` | Phase 1: propagation identical bit for bit; filter with relative difference ≤ 1e-12 |
| Parity between backends | CUDA parity kernels (same source, `legacy` mode) × OpenCL on the same NVIDIA GPU | Relative L2 error ≤ 1e-6 (initial target, to be confirmed in Phase 2) |
| Optimised kernels | Optimised CUDA (`fixed`) × parity CUDA | Seismograms ≤ 1e-5 and gradients ≤ 1e-4 in relative L2 error; FP16 with its own limits, validated by a complete FWI |
| Mathematical correctness | Adjoint dot-product test (⟨Lm, d⟩ = ⟨m, Lᵀd⟩), gradient against finite differences, homogeneous medium against the analytic solution | Relative deviation ≤ 1e-5 in the dot product |
| Memory and concurrency | `compute-sanitizer` (memcheck, racecheck, initcheck) on all new kernels | Zero errors |
| Precision guard (no GPU) | NVRTC compiles the kernels in CI and the test reads the PTX of the hot kernels, as in this analysis | Zero FP64 instructions in fixed mode |
| Performance | `pytest-benchmark` or `asv` with the three scenarios of the estimate, on the team's RTX and, if possible, on an RTX of another generation, with the median of 5 runs after warm-up, because GeForce clocks vary with temperature | No regression above 5% between commits; targets of each phase met |

Each benchmark records forward, gradient and FWI-iteration times; host time per step; GPU occupancy in Nsight Systems; achieved bandwidth per kernel in Nsight Compute; and peak memory on the GPU and on the host. Summing the gradients across shots avoids floating-point `atomicAdd`, so that two identical runs give the same result.

## Risks and mitigation

The biggest risk is not technical: it is changing results without notice. That is why each mitigation below keeps a way back to the behaviour of 0.1.10.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Numerical differences between backends | Distrust in the results | `legacy` and `fixed` modes, published tolerances, parity report in every release |
| Fixing inherited bugs (out-of-buffer reads, `vz` in the well, truncated `dt`) changes outputs | Breaks reproducibility of published work | Fixes only in the `fixed` mode; `legacy` keeps the old behaviour in OpenCL |
| Checkpoints do not fit in the RTX VRAM (6 to 32 GB depending on the model) | Failure on large models | Memory budget: GPU → asynchronous pinned host → recomputation or boundary saving |
| Graphs incompatible with `show=True` and step-by-step debugging | Loss of functionality | Graph-free path kept and covered by tests |
| Heavy dependencies (CuPy + CUDA components) | Harder installation | Optional extras, `[ctk]` installing CUDA through pip, OpenCL as an alternative |
| Machine with a driver older than 580 | No access to CUDA 13 | `cuda12` extra (CUDA 12.9), which works with older drivers |
| Maintaining two backends | Ongoing cost | Single source for the parity kernels; advanced optimisations only in CUDA |
| Licences | GPLv3+ compliance | CuPy (MIT) and cuda.core/Warp (Apache-2.0) are compatible with GPLv3 |
| Project without a release since Nov 2024 | The fork diverges from the original PyFWI | `upstream` remote with periodic merges, bug fixes sent back to the original by PR, modification notice required by GPLv3 and citation of the original authors |

## Effort and resources

The plan adds up to about 20 weeks of one person experienced in CUDA and scientific Python. With two people it falls to ~14 weeks: after gate G2, Phase 4 (boundary saving, multi-GPU, cost function on the GPU) runs in parallel with Phase 3, and only the FP16 mode waits for the optimised kernels.

- **People:** one CUDA/CuPy developer at the core of the work and support from a geophysicist to validate FWI and TL-FWI in Phases 0 and 4.
- **GPUs:** one GeForce RTX, the team's, as the main target and, if possible, an RTX of another generation, so as not to tune the code to a single architecture. A data-centre GPU is not needed.
- **CI:** GitHub Actions with POCL for OpenCL on the CPU and a self-hosted runner on the RTX machine for parity, `compute-sanitizer` and nightly benchmarks.
- **Tools:** CuPy 14 (`cupy-cuda13x` and `cupy-cuda12x`), Nsight Systems, Nsight Compute, `compute-sanitizer` and `pytest-benchmark` or `asv`.

## Sources

- [PyFWI 0.1.10 on PyPI](https://pypi.org/project/PyFWI/) and [AmirMardan/PyFWI repository](https://github.com/AmirMardan/PyFWI) (commit `1434a16`, analysed source code)
- [Mardan, Giroux and Fabien-Ouellet (2023), PyFWI, SoftwareX 22](https://espace.inrs.ca/id/eprint/13791/) and [PyFWI documentation](https://pyfwi.readthedocs.io/)
- [PyOpenCL, issue #831: repeated kernel retrieval warning](https://github.com/inducer/pyopencl/issues/831)
- [clBuildProgram manual page (-cl-single-precision-constant option)](https://manpages.debian.org/jessie/opencl-1.2-man-doc/clBuildProgram.3clc.en.html)
- [NVIDIA Ampere GA102 GPU Architecture Whitepaper](https://web.cs.miami.edu/home/burt/learning/csc596.231/docs/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)
- [Fang, Varbanescu and Sips (2011), A Comprehensive Performance Comparison of CUDA and OpenCL](https://hgpu.org/?p=5748)
- [SeisCL (Fabien-Ouellet, Gloaguen and Giroux)](https://github.com/gfabieno/SeisCL)
- [Fabien-Ouellet (2020), Seismic modeling and inversion using half-precision floating-point numbers, Geophysics 85(3)](https://publications.polymtl.ca/6303/)
- [NVIDIA (2019), Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- [NVIDIA (2024), Dynamic Control Flow in CUDA Graphs with Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes)
- [NVIDIA (2025), Navigating GPU Architecture Support](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)
- [NVIDIA, CUDA Toolkit 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
- [CuPy 14.2 on PyPI](https://pypi.org/project/cupy-cuda13x/) and [CuPy graph capture code (stream.pyx)](https://github.com/cupy/cupy/blob/main/cupy/cuda/stream.pyx)
- [cuda.core on PyPI](https://pypi.org/project/cuda-core/) and [cuda.core documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [Warp on PyPI](https://pypi.org/project/warp-lang/) and [Warp documentation](https://warp.readthedocs.io/)
- [Numba-CUDA on PyPI](https://pypi.org/project/numba-cuda/)
- [Deepwave](https://github.com/ar4/deepwave)
