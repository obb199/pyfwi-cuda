# Benchmark of the fork release 0.4.0

Roadmap step 4.7 (gate [G4](../gates/G4.md)). Measured in the night of Oct 26 to 27, 2026 with
`tools/benchmark.py` (median of 5 runs after a warm-up) on the development machine: GeForce RTX 3060 12 GB,
driver 580.95.05, Ryzen 7 5700X, CuPy 14.2 with the CUDA 13 runtime. Raw results: `bench/release-0.4.0/`.

Runs, all measured in the same session but `fork-0.3.0-cuda-fixed`: `ref-0.1.10-pyopencl-2022`, PyFWI 0.1.10
from PyPI in the frozen reference environment (inside zero guard bands, as the baseline of step 0.7), against
which every speed-up is given; `fork-0.3.0-cuda-fixed`, the default of the previous release (measured on the
evening of Oct 26, see "GPU state" below); then this release on its configurations: `fork-0.4.0-opencl`
(`PYFWI_BACKEND=opencl`, the `legacy` mode: the results of 0.1.10), `fork-0.4.0-cuda-legacy` (the parity CUDA
backend: the kernel files of OpenCL), `fork-0.4.0-cuda-fixed`, the default of this release (`auto`: CUDA, the
`fixed` mode, the native kernels; the 4 shots of the small scenario in one batch, the 2 of the medium one, the
1 of the large one), and the default with the opt-in features of Phase 4 that change the propagation:
`PYFWI_BOUNDARY_SAVING=1` (step 4.2), `PYFWI_FP16=1` (step 4.3) and both. The GPU memory is new in this
release: the peak of one forward modelling and one gradient of the shots run (4, 2 and 1), CUDA only.

| Run | Device | PyFWI | Engine | PyOpenCL | NumPy | Date |
| --- | --- | --- | --- | --- | --- | --- |
| ref-0.1.10-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | opencl (PyFWI 0.1.10), legacy | 2022.2.4 | 1.21.6 | 2026-10-27T02:37:57+00:00 |
| fork-0.3.0-cuda-fixed | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.3.0 | cuda device 0 (auto, the default), fixed, native kernels, batches of 1/2/4 | 2026.1.4 | 2.5.4 | 2026-10-26T19:50:03+00:00 |
| fork-0.4.0-opencl | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.4.0 | opencl (opencl asked by PYFWI_BACKEND), legacy | 2026.1.4 | 2.5.4 | 2026-10-27T02:48:56+00:00 |
| fork-0.4.0-cuda-legacy | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.4.0 | cuda device 0 (cuda:strict asked by PYFWI_BACKEND), legacy, parity kernels | 2026.1.4 | 2.5.4 | 2026-10-27T02:56:07+00:00 |
| fork-0.4.0-cuda-fixed | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.4.0 | cuda device 0 (auto, the default), fixed, native kernels, batches of 1/2/4 | 2026.1.4 | 2.5.4 | 2026-10-27T03:01:40+00:00 |
| fork-0.4.0-cuda-boundary-saving | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.4.0 | cuda device 0 (auto, the default), fixed, native kernels, batches of 1/2/4, boundary saving | 2026.1.4 | 2.5.4 | 2026-10-27T03:02:21+00:00 |
| fork-0.4.0-cuda-fp16 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.4.0 | cuda device 0 (auto, the default), fixed, native kernels, batches of 1/2/4, FP16 | 2026.1.4 | 2.5.4 | 2026-10-27T03:03:02+00:00 |
| fork-0.4.0-cuda-boundary-saving-fp16 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.4.0 | cuda device 0 (auto, the default), fixed, native kernels, batches of 1/2/4, boundary saving, FP16 | 2026.1.4 | 2.5.4 | 2026-10-27T03:03:33+00:00 |

Times are medians of the measured runs, scaled to the number of shots of the scenario. Speed-up is relative to the first run. GPU memory: the peak of one forward modelling and one gradient of the shots run (not scaled), CUDA backend only (recorded since 0.4.0).

| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after | GPU memory (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | ref-0.1.10-pyopencl-2022 | 1.56 | 1.93 | 1.00 / 1.00 | 0.370 / 0.460 | 4.14 / 6.34 | 2.71 | 0.013 / 0.034 | 61.0 / 62.0 | - |
| small | fork-0.3.0-cuda-fixed | 0.02 | 0.07 | 72.29 / 29.44 | 0.005 / 0.016 | 0.00 / 0.00 | 0.00 | 0.025 / 0.011 | 59.0 / 65.0 | - |
| small | fork-0.4.0-opencl | 0.41 | 0.77 | 3.78 / 2.51 | 0.098 / 0.183 | 4.14 / 6.34 | 0.01 | 0.037 / 0.003 | 66.0 / 65.0 | - |
| small | fork-0.4.0-cuda-legacy | 0.29 | 0.65 | 5.46 / 2.96 | 0.068 / 0.155 | 0.00 / 0.00 | 0.00 | 0.001 / 0.001 | 66.0 / 65.0 | 320 |
| small | fork-0.4.0-cuda-fixed | 0.02 | 0.06 | 76.68 / 32.24 | 0.005 / 0.014 | 0.00 / 0.00 | 0.00 | 0.054 / 0.012 | 59.0 / 65.0 | 325 |
| small | fork-0.4.0-cuda-boundary-saving | 0.02 | 0.07 | 79.56 / 27.21 | 0.005 / 0.017 | 0.00 / 0.00 | 0.00 | 0.045 / 0.010 | 66.0 / 71.0 | 77 |
| small | fork-0.4.0-cuda-fp16 | 0.02 | 0.05 | 96.27 / 40.47 | 0.004 / 0.011 | 0.00 / 0.00 | 0.00 | 0.025 / 0.009 | 65.0 / 71.0 | 167 |
| small | fork-0.4.0-cuda-boundary-saving-fp16 | 0.02 | 0.06 | 92.64 / 34.18 | 0.004 / 0.013 | 0.00 / 0.00 | 0.00 | 0.015 / 0.007 | 65.0 / 72.0 | 43 |
| medium (scaled x4 to 8 shots) | ref-0.1.10-pyopencl-2022 | 46.61 | 57.18 | 1.00 / 1.00 | 1.943 / 2.384 | 4.14 / 6.34 | 2.71 | 0.103 / 0.021 | 65.0 / 65.0 | - |
| medium (scaled x4 to 8 shots) | fork-0.3.0-cuda-fixed | 1.66 | 4.17 | 28.06 / 13.71 | 0.069 / 0.174 | 0.00 / 0.00 | 0.00 | 0.007 / 0.007 | 70.0 / 72.0 | - |
| medium (scaled x4 to 8 shots) | fork-0.4.0-opencl | 17.93 | 39.42 | 2.60 / 1.45 | 0.748 / 1.644 | 4.14 / 6.34 | 0.00 | 0.025 / 0.005 | 72.0 / 73.0 | - |
| medium (scaled x4 to 8 shots) | fork-0.4.0-cuda-legacy | 13.24 | 34.26 | 3.52 / 1.67 | 0.552 / 1.428 | 0.00 / 0.00 | 0.00 | 0.009 / 0.009 | 65.0 / 64.0 | 4450 |
| medium (scaled x4 to 8 shots) | fork-0.4.0-cuda-fixed | 1.58 | 3.96 | 29.51 / 14.45 | 0.066 / 0.165 | 0.00 / 0.00 | 0.00 | 0.019 / 0.021 | 71.0 / 73.0 | 4468 |
| medium (scaled x4 to 8 shots) | fork-0.4.0-cuda-boundary-saving | 1.51 | 4.24 | 30.94 / 13.49 | 0.063 / 0.177 | 0.00 / 0.00 | 0.00 | 0.004 / 0.026 | 75.0 / 77.0 | 418 |
| medium (scaled x4 to 8 shots) | fork-0.4.0-cuda-fp16 | 1.16 | 3.04 | 40.08 / 18.82 | 0.048 / 0.127 | 0.00 / 0.00 | 0.00 | 0.006 / 0.002 | 76.0 / 77.0 | 2249 |
| medium (scaled x4 to 8 shots) | fork-0.4.0-cuda-boundary-saving-fp16 | 1.13 | 3.30 | 41.38 / 17.34 | 0.047 / 0.137 | 0.00 / 0.00 | 0.00 | 0.007 / 0.006 | 77.0 / 78.0 | 224 |
| large (scaled x50 to 50 shots) | ref-0.1.10-pyopencl-2022 | 792.55 | 1591.72 | 1.00 / 1.00 | 2.643 / 5.307 | 4.14 / 6.94 | 1.21 | 0.017 / 0.015 | 71.0 / 72.0 | - |
| large (scaled x50 to 50 shots) | fork-0.3.0-cuda-fixed | 48.42 | 139.75 | 16.37 / 11.39 | 0.161 / 0.466 | 0.00 / 0.00 | 0.00 | 0.007 / 0.004 | 73.0 / 75.0 | - |
| large (scaled x50 to 50 shots) | fork-0.4.0-opencl | 528.95 | 1449.18 | 1.50 / 1.10 | 1.764 / 4.832 | 4.14 / 6.94 | 0.00 | 0.006 / 0.007 | 76.0 / 76.0 | - |
| large (scaled x50 to 50 shots) | fork-0.4.0-cuda-legacy | 391.99 | 1142.74 | 2.02 / 1.39 | 1.307 / 3.810 | 0.00 / 0.00 | 0.00 | 0.013 / 0.014 | 64.0 / 64.0 | 2952 |
| large (scaled x50 to 50 shots) | fork-0.4.0-cuda-fixed | 46.50 | 133.44 | 17.04 / 11.93 | 0.155 / 0.445 | 0.00 / 0.00 | 0.00 | 0.006 / 0.005 | 74.0 / 76.0 | 2952 |
| large (scaled x50 to 50 shots) | fork-0.4.0-cuda-boundary-saving | 46.42 | 135.38 | 17.07 / 11.76 | 0.155 / 0.451 | 0.00 / 0.00 | 0.00 | 0.027 / 0.001 | 77.0 / 78.0 | 696 |
| large (scaled x50 to 50 shots) | fork-0.4.0-cuda-fp16 | 34.65 | 102.58 | 22.88 / 15.52 | 0.116 / 0.342 | 0.00 / 0.00 | 0.00 | 0.005 / 0.003 | 78.0 / 78.0 | 1501 |
| large (scaled x50 to 50 shots) | fork-0.4.0-cuda-boundary-saving-fp16 | 35.05 | 105.08 | 22.61 / 15.15 | 0.117 / 0.350 | 0.00 / 0.00 | 0.00 | 0.033 / 0.028 | 78.0 / 79.0 | 373 |

## Reading the numbers

- **The defaults did not change.** The default configuration is 77× / 32× faster (forward / gradient) than
  0.1.10 in the small scenario, 29.5× / 14.4× in the medium one and 17.0× / 11.9× in the large one; against
  the default of 0.3.0, 4 % to 10 % faster, which is the state of the GPU (below), not the code: Phase 4 left
  the default path as it was (its features are off by default), and 0.1.10 also ran up to 10 % faster in
  this session. OpenCL and the parity CUDA backend likewise.
- **Boundary saving trades a little gradient time for memory.** Instead of checkpoints of the whole field,
  a band of `sdo` cells around the undamped region is stored at every step: the peak GPU memory falls from
  325 MB to 77 MB (small, 4.2× less), from 4.5 GB to 418 MB (medium, 10.7×) and from 3.0 GB to 696 MB
  (large, 4.2×; its checkpoints are 5 % of the steps, `chpr=5`, against 20 % elsewhere). The forward
  modelling takes the same time (medium 5 % less: no checkpoint copies); the gradient 1.5 % (large) to 18 %
  (small) longer: three small launches more per step to rebuild the field. Where the checkpoints do not fit
  on the GPU it is faster on every count: the real medium scenario (8 shots) runs in 2.4 GB instead of
  9.8 GB, forward 1.8× faster (step 4.2).
- **FP16 halves the memory and gains 1.25× to 1.36×.** The fields, adjoint fields, checkpoints and stored
  bands take 2 bytes instead of 4: the peak memory is 50 % to 51 % of float32; forward 1.25× /
  1.36× / 1.34× faster and gradient 1.25× / 1.30× / 1.30× (small / medium / large). Less than 2×: the model,
  the gradients and their preconditioners stay float32. For modelling and gradients only: the complete
  inversion does not converge in FP16 (the tolerance table of the [CHANGELOG](../../../CHANGELOG.md)).
- **Both together**: 7.6× (small), 20× (medium, 4.5 GB → 224 MB) and 7.9× (large) less GPU memory, at the
  speed of FP16 for the forward modelling, the gradient 2 % (large) to 19 % (small) slower than FP16 alone,
  as boundary saving alone.
- **GPU state.** This session ran at night, the SM clock at 1.90 GHz, 59 °C to 76 °C: 0.1.10 ran 5 % to
  10 % faster than in the session of 0.3.0 in the medium and large scenarios (within 3 % in the small one).
  The comparison with 0.3.0 favours this release by about as much; every other speed-up compares runs of
  this session.
- The *launches* and *blocking copies per step* columns count OpenCL calls only (`tools/benchmark.py`):
  they read 0 for the CUDA runs, which replay CUDA graphs. The cost function on the GPU
  (`PYFWI_CUDA_COST=1`, step 4.1) is not in these scenarios, which time the propagation only: it cut the
  cost of an evaluation from 38.5 ms to 3.6 ms (the whole inversion of the small scenario −31 %). The shots in
  several processes (`PYFWI_CUDA_DEVICES`, step 4.4) need several GPUs to gain: not measurable here (one
  worker costs +2 %, step 4.4).
