# Benchmark of the fork release 0.1.11

Roadmap step 1.9 (gate [G1](../gates/G1.md)). Measured on Oct 25, 2026 with `tools/benchmark.py` (median of
5 runs after a warm-up; the tool's zero guard bands, which the baseline needed, no longer wrap the
buffers of the fork since step 1.7) and
`tools/opencl_profile.py`, on the development machine: GeForce RTX 3060 12 GB, driver 580.95.05,
Ryzen 7 5700X. Raw results: `bench/release-0.1.11/`. The fork was measured at commit `d342596`,
whose code is that of the release (the result files still say `0.1.11.dev0`).

Runs: `ref-0.1.10-pyopencl-2022`, the baseline of step 0.7 (PyFWI 0.1.10 from PyPI in the frozen
reference environment), against which every gain is given; `dev-pyopencl-2026`, the same 0.1.10 code
in the development environment; `fork-0.1.11-pyopencl-2026`, this release in the development
environment.

| Run | Device | PyFWI | PyOpenCL | NumPy | Date |
| --- | --- | --- | --- | --- | --- |
| ref-0.1.10-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | 2022.2.4 | 1.21.6 | 2026-10-25T17:23:58+00:00 |
| dev-pyopencl-2026 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | 2026.1.4 | 2.5.4 | 2026-10-25T17:12:45+00:00 |
| fork-0.1.11-pyopencl-2026 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.11.dev0 | 2026.1.4 | 2.5.4 | 2026-10-25T21:38:06+00:00 |

Times are medians of the measured runs, scaled to the number of shots of the scenario. Speed-up is relative to the first run.

| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | ref-0.1.10-pyopencl-2022 | 1.50 | 1.91 | 1.00 / 1.00 | 0.356 / 0.454 | 4.14 / 6.34 | 2.71 | 0.011 / 0.011 | 59.0 / 55.0 |
| small | dev-pyopencl-2026 | 2.16 | 3.05 | 0.69 / 0.62 | 0.515 / 0.727 | 4.14 / 6.34 | 2.71 | 0.018 / 0.018 | 55.0 / 62.0 |
| small | fork-0.1.11-pyopencl-2026 | 0.42 | 0.76 | 3.52 / 2.51 | 0.101 / 0.181 | 4.14 / 6.34 | 0.01 | 0.043 / 0.004 | 66.0 / 67.0 |
| medium (scaled x4 to 8 shots) | ref-0.1.10-pyopencl-2022 | 47.33 | 57.32 | 1.00 / 1.00 | 1.973 / 2.390 | 4.14 / 6.34 | 2.71 | 0.003 / 0.027 | 63.0 / 64.0 |
| medium (scaled x4 to 8 shots) | dev-pyopencl-2026 | 48.80 | 58.27 | 0.97 / 0.98 | 2.035 / 2.430 | 4.14 / 6.34 | 2.71 | 0.005 / 0.004 | 65.0 / 63.0 |
| medium (scaled x4 to 8 shots) | fork-0.1.11-pyopencl-2026 | 17.94 | 38.74 | 2.64 / 1.48 | 0.748 / 1.615 | 4.14 / 6.34 | 0.00 | 0.003 / 0.001 | 71.0 / 68.0 |
| large (scaled x50 to 50 shots) | ref-0.1.10-pyopencl-2022 | 769.99 | 1541.96 | 1.00 / 1.00 | 2.567 / 5.142 | 4.14 / 6.94 | 1.21 | 0.023 / 0.022 | 69.0 / 69.0 |
| large (scaled x50 to 50 shots) | dev-pyopencl-2026 | 797.87 | 1617.23 | 0.97 / 0.95 | 2.660 / 5.393 | 4.14 / 6.94 | 1.21 | 0.014 / 0.024 | 69.0 / 70.0 |
| large (scaled x50 to 50 shots) | fork-0.1.11-pyopencl-2026 | 503.96 | 1346.70 | 1.53 / 1.14 | 1.680 / 4.491 | 4.14 / 6.94 | 0.00 | 0.002 / 0.003 | 71.0 / 71.0 |

## Reading the numbers

- **No scenario is slower than the baseline**, the criterion of G1. Forward / gradient speed-ups:
  small 3.5× / 2.5×, medium 2.6× / 1.5×, large 1.5× / 1.14×, with exactly the results of 0.1.10
  (characterisation bit for bit).
- **Where the gains come from** (one report per step): retrieving each kernel once and setting its
  arguments once (step 1.3: small scenario 3.0× / 2.4×); the residual sent once per shot (1.4); the
  checkpoints copied straight from and to contiguous storage without blocking (1.5: medium forward
  2.7×); the low-pass filter of the inversion, 11 to 22× faster, does not appear here because the
  benchmark measures one forward modelling and one gradient, not an FWI iteration.
- **The host no longer waits for the GPU**: blocking copies per step fell from 2.71 (1.21 in the large
  scenario) to below 0.01; the GPU is busy 72 % of the time in the small scenario and 88 % in the
  medium one (28 % and 51 % with 0.1.10).
- **What remains is GPU time.** The large gradient gains least (1.14×): its kernels dominate, and they
  still compute in FP64 because their float constants have no `f` suffix (known bug, `fixed` mode of
  step 2.4), and the CUDA kernels of Phases 2 and 3 address their speed.
- The forward run of the small scenario has a coefficient of variation of 4.3 %, above the 3 % asked of
  the baseline at G0 (0.42 s runs are sensitive to clock changes; 66 to 67 °C); every other is below
  0.5 %. The conclusion does not depend on it.
