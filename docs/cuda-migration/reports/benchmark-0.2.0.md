# Benchmark of the fork release 0.2.0

Roadmap step 2.7 (gate [G2](../gates/G2.md)). Measured on Oct 26, 2026 with `tools/benchmark.py` (median
of 5 runs after a warm-up, plain buffers) on the development machine: GeForce RTX 3060 12 GB, driver
580.95.05, Ryzen 7 5700X, CuPy 14.2 with the CUDA 13 runtime. Raw results: `bench/release-0.2.0/`.

Runs: `ref-0.1.10-pyopencl-2022`, the baseline of step 0.7 (PyFWI 0.1.10 from PyPI in the frozen
reference environment), against which every speed-up is given; `fork-0.1.11-pyopencl-2026`, the first
release of the fork (OpenCL); then this release on its three configurations: `fork-0.2.0-opencl`, the
default (OpenCL, `legacy` numeric mode: the results of 0.1.10), `fork-0.2.0-cuda-legacy`, the parity
CUDA backend (`PYFWI_BACKEND=cuda`, `PYFWI_NUMERICS=legacy`: the same kernel files and arithmetic), and
`fork-0.2.0-cuda-fixed`, the CUDA backend in its default numeric mode (`fixed`, step 2.4).

| Run | Device | PyFWI | PyOpenCL | NumPy | Date |
| --- | --- | --- | --- | --- | --- |
| ref-0.1.10-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | 2022.2.4 | 1.21.6 | 2026-10-25T17:23:58+00:00 |
| fork-0.1.11-pyopencl-2026 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.11.dev0 | 2026.1.4 | 2.5.4 | 2026-10-25T21:38:06+00:00 |
| fork-0.2.0-opencl | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.2.0 | 2026.1.4 | 2.5.4 | 2026-10-26T13:41:41+00:00 |
| fork-0.2.0-cuda-legacy | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.2.0 | 2026.1.4 | 2.5.4 | 2026-10-26T13:48:28+00:00 |
| fork-0.2.0-cuda-fixed | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.2.0 | 2026.1.4 | 2.5.4 | 2026-10-26T13:53:48+00:00 |

Times are medians of the measured runs, scaled to the number of shots of the scenario. Speed-up is relative to the first run.

| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | ref-0.1.10-pyopencl-2022 | 1.50 | 1.91 | 1.00 / 1.00 | 0.356 / 0.454 | 4.14 / 6.34 | 2.71 | 0.011 / 0.011 | 59.0 / 55.0 |
| small | fork-0.1.11-pyopencl-2026 | 0.42 | 0.76 | 3.52 / 2.51 | 0.101 / 0.181 | 4.14 / 6.34 | 0.01 | 0.043 / 0.004 | 66.0 / 67.0 |
| small | fork-0.2.0-opencl | 0.43 | 0.75 | 3.45 / 2.53 | 0.103 / 0.179 | 4.14 / 6.34 | 0.01 | 0.064 / 0.008 | 54.0 / 56.0 |
| small | fork-0.2.0-cuda-legacy | 0.29 | 0.64 | 5.14 / 2.96 | 0.069 / 0.154 | 0.00 / 0.00 | 0.00 | 0.002 / 0.001 | 65.0 / 63.0 |
| small | fork-0.2.0-cuda-fixed | 0.20 | 0.19 | 7.52 / 10.19 | 0.047 / 0.045 | 0.00 / 0.00 | 0.00 | 0.002 / 0.003 | 58.0 / 59.0 |
| medium (scaled x4 to 8 shots) | ref-0.1.10-pyopencl-2022 | 47.33 | 57.32 | 1.00 / 1.00 | 1.973 / 2.390 | 4.14 / 6.34 | 2.71 | 0.003 / 0.027 | 63.0 / 64.0 |
| medium (scaled x4 to 8 shots) | fork-0.1.11-pyopencl-2026 | 17.94 | 38.74 | 2.64 / 1.48 | 0.748 / 1.615 | 4.14 / 6.34 | 0.00 | 0.003 / 0.001 | 71.0 / 68.0 |
| medium (scaled x4 to 8 shots) | fork-0.2.0-opencl | 17.57 | 38.25 | 2.69 / 1.50 | 0.733 / 1.595 | 4.14 / 6.34 | 0.00 | 0.004 / 0.001 | 66.0 / 71.0 |
| medium (scaled x4 to 8 shots) | fork-0.2.0-cuda-legacy | 13.49 | 33.43 | 3.51 / 1.71 | 0.562 / 1.394 | 0.00 / 0.00 | 0.00 | 0.002 / 0.003 | 63.0 / 61.0 |
| medium (scaled x4 to 8 shots) | fork-0.2.0-cuda-fixed | 2.99 | 5.88 | 15.85 / 9.75 | 0.125 / 0.245 | 0.00 / 0.00 | 0.00 | 0.003 / 0.002 | 67.0 / 69.0 |
| large (scaled x50 to 50 shots) | ref-0.1.10-pyopencl-2022 | 769.99 | 1541.96 | 1.00 / 1.00 | 2.567 / 5.142 | 4.14 / 6.94 | 1.21 | 0.023 / 0.022 | 69.0 / 69.0 |
| large (scaled x50 to 50 shots) | fork-0.1.11-pyopencl-2026 | 503.96 | 1346.70 | 1.53 / 1.14 | 1.680 / 4.491 | 4.14 / 6.94 | 0.00 | 0.002 / 0.003 | 71.0 / 71.0 |
| large (scaled x50 to 50 shots) | fork-0.2.0-opencl | 502.22 | 1348.28 | 1.53 / 1.14 | 1.675 / 4.496 | 4.14 / 6.94 | 0.00 | 0.001 / 0.003 | 73.0 / 75.0 |
| large (scaled x50 to 50 shots) | fork-0.2.0-cuda-legacy | 379.60 | 1067.12 | 2.03 / 1.44 | 1.266 / 3.558 | 0.00 / 0.00 | 0.00 | 0.000 / 0.000 | 62.0 / 63.0 |
| large (scaled x50 to 50 shots) | fork-0.2.0-cuda-fixed | 74.16 | 165.21 | 10.38 / 9.33 | 0.247 / 0.551 | 0.00 / 0.00 | 0.00 | 0.002 / 0.003 | 72.0 / 74.0 |

## Reading the numbers

- **The parity CUDA backend is faster than OpenCL in every scenario**, the DES criterion of step 2.7:
  forward / gradient 0.29 s / 0.64 s against 0.43 s / 0.75 s (small), 13.5 s / 33.4 s against 17.6 s /
  38.3 s (medium), 380 s / 1,067 s against 502 s / 1,348 s (large), with the same arithmetic (results
  within the tolerances of the [parity report](parity-0.2.0.md)). It got there in step 2.7: copies
  through pinned buffers (CuPy copies from pageable memory made the host wait for the GPU) and blocks
  sized to the grid.
- **The `fixed` mode is where CUDA pays**: 7.5× / 10.2× (small), 15.9× / 9.8× (medium) and 10.4× /
  9.3× (large) against 0.1.10, and 2.2× to 8.2× against the OpenCL default of this release. Its kernels
  compute in FP32 only (the `legacy` constants make part of the arithmetic FP64, 1/64 of the FP32 rate
  on GeForce cards); its other corrections (source injection at every step, adjoint gradient loop)
  cost little (one more launch per step in the forward). Its results differ from those of 0.1.10 (see
  the CHANGELOG) and are more accurate (mathematical baseline, `test/test_math.py`).
- **OpenCL is unchanged** since 0.1.11 (within 2 %): Phase 2 added the CUDA backend behind the same
  engine, not a faster OpenCL path.
- The *launches* and *blocking copies per step* columns count OpenCL calls only (`tools/benchmark.py`):
  they read 0 for the CUDA runs, which launch 4.1 (`legacy`) or 5 (`fixed`) kernels per step in the
  forward and about 7 in the gradient, and copy the checkpoints without blocking.
- Every coefficient of variation is below 0.7 % except the small OpenCL forward (6.4 %, 0.43 s runs);
  the conclusions do not depend on it.
