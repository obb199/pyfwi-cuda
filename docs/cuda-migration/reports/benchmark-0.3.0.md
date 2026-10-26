# Benchmark of the fork release 0.3.0

Roadmap step 3.11 (gate [G3](../gates/G3.md)). Measured on Oct 26, 2026 with `tools/benchmark.py` (median
of 5 runs after a warm-up) on the development machine: GeForce RTX 3060 12 GB, driver 580.95.05, Ryzen 7
5700X, CuPy 14.2 with the CUDA 13 runtime. Raw results: `bench/release-0.3.0/`.

Runs, all measured in the same session: `ref-0.1.10-pyopencl-2022`, PyFWI 0.1.10 from PyPI in the frozen
reference environment (inside zero guard bands, as the baseline of step 0.7), against which every speed-up
is given; then this release on its three configurations: `fork-0.3.0-opencl` (`PYFWI_BACKEND=opencl`, the
`legacy` mode: the results of 0.1.10), `fork-0.3.0-cuda-legacy` (the parity CUDA backend: the kernel files
of OpenCL), and `fork-0.3.0-cuda-fixed`, the default of this release (`auto`: CUDA, `fixed` mode, the native
kernels of Phase 3; the 4 shots of the small scenario in one batch, the 2 of the medium one, the 1 of the
large one). `fork-0.2.0-cuda-fixed` is the default CUDA configuration of the previous release, measured on
the morning of the same day (see "GPU state" below).

| Run | Device | PyFWI | Engine | PyOpenCL | NumPy | Date |
| --- | --- | --- | --- | --- | --- | --- |
| ref-0.1.10-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | opencl (PyFWI 0.1.10), legacy | 2022.2.4 | 1.21.6 | 2026-10-26T19:24:45+00:00 |
| fork-0.2.0-cuda-fixed | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.2.0 | not recorded | 2026.1.4 | 2.5.4 | 2026-10-26T13:53:48+00:00 |
| fork-0.3.0-opencl | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.3.0 | opencl (opencl asked by PYFWI_BACKEND), legacy | 2026.1.4 | 2.5.4 | 2026-10-26T19:36:18+00:00 |
| fork-0.3.0-cuda-legacy | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.3.0 | cuda device 0 (cuda:strict asked by PYFWI_BACKEND), legacy, parity kernels | 2026.1.4 | 2.5.4 | 2026-10-26T19:44:03+00:00 |
| fork-0.3.0-cuda-fixed | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.3.0 | cuda device 0 (auto, the default), fixed, native kernels, batches of 1/2/4 | 2026.1.4 | 2.5.4 | 2026-10-26T19:50:03+00:00 |

Times are medians of the measured runs, scaled to the number of shots of the scenario. Speed-up is relative to the first run.

| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | ref-0.1.10-pyopencl-2022 | 1.53 | 1.99 | 1.00 / 1.00 | 0.364 / 0.473 | 4.14 / 6.34 | 2.71 | 0.008 / 0.012 | 51.0 / 52.0 |
| small | fork-0.2.0-cuda-fixed | 0.20 | 0.19 | 7.68 / 10.61 | 0.047 / 0.045 | 0.00 / 0.00 | 0.00 | 0.002 / 0.003 | 58.0 / 59.0 |
| small | fork-0.3.0-opencl | 0.48 | 0.83 | 3.15 / 2.40 | 0.115 / 0.198 | 4.14 / 6.34 | 0.01 | 0.019 / 0.004 | 65.0 / 64.0 |
| small | fork-0.3.0-cuda-legacy | 0.31 | 0.71 | 4.96 / 2.82 | 0.073 / 0.168 | 0.00 / 0.00 | 0.00 | 0.003 / 0.001 | 65.0 / 64.0 |
| small | fork-0.3.0-cuda-fixed | 0.02 | 0.07 | 71.00 / 30.29 | 0.005 / 0.016 | 0.00 / 0.00 | 0.00 | 0.025 / 0.011 | 59.0 / 65.0 |
| medium (scaled x4 to 8 shots) | ref-0.1.10-pyopencl-2022 | 51.08 | 59.82 | 1.00 / 1.00 | 2.130 / 2.494 | 4.14 / 6.34 | 2.71 | 0.016 / 0.007 | 62.0 / 62.0 |
| medium (scaled x4 to 8 shots) | fork-0.2.0-cuda-fixed | 2.99 | 5.88 | 17.11 / 10.17 | 0.125 / 0.245 | 0.00 / 0.00 | 0.00 | 0.003 / 0.002 | 67.0 / 69.0 |
| medium (scaled x4 to 8 shots) | fork-0.3.0-opencl | 19.54 | 43.14 | 2.61 / 1.39 | 0.815 / 1.799 | 4.14 / 6.34 | 0.00 | 0.004 / 0.004 | 71.0 / 72.0 |
| medium (scaled x4 to 8 shots) | fork-0.3.0-cuda-legacy | 14.41 | 37.30 | 3.54 / 1.60 | 0.601 / 1.555 | 0.00 / 0.00 | 0.00 | 0.002 / 0.001 | 64.0 / 62.0 |
| medium (scaled x4 to 8 shots) | fork-0.3.0-cuda-fixed | 1.66 | 4.17 | 30.75 / 14.34 | 0.069 / 0.174 | 0.00 / 0.00 | 0.00 | 0.007 / 0.007 | 70.0 / 72.0 |
| large (scaled x50 to 50 shots) | ref-0.1.10-pyopencl-2022 | 852.37 | 1699.14 | 1.00 / 1.00 | 2.842 / 5.666 | 4.14 / 6.94 | 1.21 | 0.020 / 0.008 | 70.0 / 72.0 |
| large (scaled x50 to 50 shots) | fork-0.2.0-cuda-fixed | 74.16 | 165.21 | 11.49 / 10.28 | 0.247 / 0.551 | 0.00 / 0.00 | 0.00 | 0.002 / 0.003 | 72.0 / 74.0 |
| large (scaled x50 to 50 shots) | fork-0.3.0-opencl | 574.87 | 1557.82 | 1.48 / 1.09 | 1.917 / 5.194 | 4.14 / 6.94 | 0.00 | 0.002 / 0.005 | 73.0 / 74.0 |
| large (scaled x50 to 50 shots) | fork-0.3.0-cuda-legacy | 426.54 | 1219.71 | 2.00 / 1.39 | 1.422 / 4.067 | 0.00 / 0.00 | 0.00 | 0.002 / 0.001 | 63.0 / 64.0 |
| large (scaled x50 to 50 shots) | fork-0.3.0-cuda-fixed | 48.42 | 139.75 | 17.60 / 12.16 | 0.161 / 0.466 | 0.00 / 0.00 | 0.00 | 0.007 / 0.004 | 73.0 / 75.0 |

## Reading the numbers

- **The targets of gate G3 are met.** Against 0.1.10, the default configuration of this release is
  71× / 30× faster (forward / gradient) in the small scenario (target 20 to 100×), 31× / 14× in the
  medium one (target 10 to 40×) and 17.6× / 12.2× in the large one (target 3 to 10×). Against the
  baseline stored at step 0.7 (Oct 25), measured with the GPU in a faster state: 69× / 29×, 28× / 14× and
  16× / 11×.
- **Phase 3 against Phase 2** (`fork-0.2.0-cuda-fixed`, the same arithmetic up to rounding): 9.3× / 2.9×
  faster in the small scenario, 1.8× / 1.4× in the medium one, 1.5× / 1.2× in the large one; at least these
  factors, since 0.2.0 was measured with the GPU in a faster state. The small scenario gained most: its
  time was mostly the host launching kernels and copies one by one (steps 3.5 to 3.9: fused kernels, the
  time step on the device, CUDA graphs, checkpoints on the GPU copied a batch at a time); its forward
  modelling now takes about 20 us per step for its 4 shots. The medium and large scenarios are limited by
  the memory bandwidth of the GPU (the kernels move at least 205 to 266 GB/s, of 360 GB/s: see the
  [timeline report](timeline-0.3.0.md)); they gained from fewer launches and fewer reads (fusion, the
  damping profiles read at the borders only) and from the checkpoints kept on the GPU.
- **The gradient of the small scenario is limited by the GPU**: with the 4 shots in one batch, its fields
  no longer fit in the 3 MB L2 cache (with 2 shots per batch it is 4 % faster, the forward modelling 17 %
  slower).
- **OpenCL and the parity CUDA backend are unchanged** since 0.2.0: Phase 3 changed the native path only.
  Their times here are 11 % to 16 % above those of the 0.2.0 report because of the GPU state; the CUDA
  parity backend stays faster than OpenCL in every scenario.
- **GPU state.** In this session the GPU ran in its P2 state (memory clock 7.3 GHz), as a CUDA program,
  and warmer than when 0.2.0 and the baseline were measured: the same engines took longer (0.1.10 2 % to
  11 %, the OpenCL backend of the fork 11 % to 16 %). Every speed-up above compares runs of this session, except those against 0.2.0 and against
  the stored baseline, which favour the older runs.
- The *launches* and *blocking copies per step* columns count OpenCL calls only (`tools/benchmark.py`):
  they read 0 for the CUDA runs, which replay CUDA graphs.

Real medium scenario (8 shots run instead of 2 scaled, the EST check of G3): 2.53 s forward, 4.59 s
gradient, in 2 batches of 4 shots; the checkpoints of 4 shots (9.3 GB) stay on the GPU, those of the other
4 in host memory; at most 9.6 GB of the 12.3 GB of the GPU used, desktop included.
