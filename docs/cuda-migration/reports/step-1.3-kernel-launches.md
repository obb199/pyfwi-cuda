# Step 1.3: kernels retrieved once, arguments set once

Roadmap step 1.3. Measured on Oct 25, 2026 with `tools/benchmark.py` (median of 5 runs after a
warm-up, inside zero guard bands) and `tools/opencl_profile.py`, on the development machine
(GeForce RTX 3060, driver 580.95.05, Ryzen 7 5700X). Raw results: `bench/step-1.3/*.json`.

## What changed

PyFWI 0.1.10 launched each kernel as `self.prg.update_velx(queue, size, None, *arguments)`, about
five times per time step. Each launch retrieved the kernel from its program again (a new kernel
object, `clCreateKernel`; PyOpenCL 2026 also warns `RepeatedKernelRetrieval`) and set every
argument again. Since step 1.3 `WavePropagator` retrieves each kernel once (`W.knl`), sets the
arguments that stay the same during a time loop once per forward modelling or gradient
(`set_args`; only the source samples and the time index change, at each injection) and launches
the time-loop kernels with `pyopencl.enqueue_nd_range_kernel`. Same kernels, same order, same
arguments: the results are the same bit for bit (characterisation tests).

Runs:

- `ref-0.1.10-pyopencl-2022` and `dev-pyopencl-2026`: the baseline of step 0.7 (0.1.10 code);
- `fork-1.2-*`: the fork just before this step (commit `c397791`), whose launches are those of
  0.1.10, in the development environment (PyOpenCL 2026) and with the packages of the reference
  environment (PyOpenCL 2022, NumPy 1.21; the fork's sources put first on `PYTHONPATH`);
- `fork-1.3-*`: the same with this step.

## Small scenario (100 x 100, 4 shots, 1,050 steps)

| Run | Device | PyFWI | PyOpenCL | NumPy | Date |
| --- | --- | --- | --- | --- | --- |
| ref-0.1.10-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | 2022.2.4 | 1.21.6 | 2026-10-25T17:23:58+00:00 |
| dev-pyopencl-2026 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | 2026.1.4 | 2.5.4 | 2026-10-25T17:12:45+00:00 |
| fork-1.2-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.11.dev0 | 2022.2.4 | 1.21.6 | 2026-10-25T19:39:08+00:00 |
| fork-1.2-pyopencl-2026 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.11.dev0 | 2026.1.4 | 2.5.4 | 2026-10-25T19:38:11+00:00 |
| fork-1.3-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.11.dev0 | 2022.2.4 | 1.21.6 | 2026-10-25T19:39:38+00:00 |
| fork-1.3-pyopencl-2026 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.11.dev0 | 2026.1.4 | 2.5.4 | 2026-10-25T19:38:55+00:00 |

Times are medians of the measured runs, scaled to the number of shots of the scenario. Speed-up is relative to the first run.

| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | ref-0.1.10-pyopencl-2022 | 1.50 | 1.91 | 1.00 / 1.00 | 0.356 / 0.454 | 4.14 / 6.34 | 2.71 | 0.011 / 0.011 | 59.0 / 55.0 |
| small | dev-pyopencl-2026 | 2.16 | 3.05 | 0.69 / 0.62 | 0.515 / 0.727 | 4.14 / 6.34 | 2.71 | 0.018 / 0.018 | 55.0 / 62.0 |
| small | fork-1.2-pyopencl-2022 | 1.43 | 1.79 | 1.04 / 1.07 | 0.341 / 0.427 | 4.14 / 6.34 | 2.71 | 0.005 / 0.002 | 59.0 / 55.0 |
| small | fork-1.2-pyopencl-2026 | 2.06 | 2.92 | 0.73 / 0.65 | 0.490 / 0.695 | 4.14 / 6.34 | 2.71 | 0.012 / 0.007 | 48.0 / 55.0 |
| small | fork-1.3-pyopencl-2022 | 0.52 | 0.82 | 2.90 / 2.32 | 0.123 / 0.195 | 4.14 / 6.34 | 2.71 | 0.008 / 0.004 | 58.0 / 58.0 |
| small | fork-1.3-pyopencl-2026 | 0.49 | 0.81 | 3.04 / 2.36 | 0.117 / 0.193 | 4.14 / 6.34 | 2.71 | 0.005 / 0.003 | 61.0 / 64.0 |

## Medium scenario (400 x 400, 8 shots, 3,000 steps)

Speed-up relative to `ref-0.1.10-pyopencl-2022`.

| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medium (scaled x4 to 8 shots) | ref-0.1.10-pyopencl-2022 | 47.33 | 57.32 | 1.00 / 1.00 | 1.973 / 2.390 | 4.14 / 6.34 | 2.71 | 0.003 / 0.027 | 63.0 / 64.0 |
| medium (scaled x4 to 8 shots) | dev-pyopencl-2026 | 48.80 | 58.27 | 0.97 / 0.98 | 2.035 / 2.430 | 4.14 / 6.34 | 2.71 | 0.005 / 0.004 | 65.0 / 63.0 |
| medium (scaled x4 to 8 shots) | fork-1.3-pyopencl-2026 | 47.86 | 52.07 | 0.99 / 1.10 | 1.995 / 2.171 | 4.14 / 6.34 | 2.71 | 0.036 / 0.019 | 59.0 / 63.0 |

## Reading the numbers

- **Small scenario: 3.0x faster forward and 2.4x faster gradient than the baseline** (1.50 s to
  0.49 s, and 1.91 s to 0.81 s). Against the fork just before the step, with PyOpenCL 2026: 4.2x
  and 3.6x. The time per kernel launch (wall time per step divided by launches per step) fell from
  118 us to 28 us; the GPU itself needs 21 us per launch on average here, so the host is no longer
  the main limit: the GPU is busy 65 % of the time, against 28 % with 0.1.10 (`tools/opencl_profile.py`).
- **The difference between PyOpenCL 2022 and 2026 is gone**: 0.52 s and 0.49 s (forward), 0.82 s
  and 0.81 s (gradient), against 1.43 s and 2.06 s before. It came from re-creating a kernel object
  at every launch, which PyOpenCL 2026 does more slowly.
- **Medium scenario: gradient 10 % faster, forward unchanged.** Each launch there takes about
  200 us on the GPU, so launch overhead mattered less; the GPU is busy 53 % of the time (51 % before).
  The rest is mostly the blocking copies of the checkpoints and of the seismograms (2.7 per step),
  which steps 1.4 and 1.5 address. The coefficient of variation of the medium forward run is 3.6 %,
  above the 3 % of gate G0 (temperature 59 to 63 °C); the conclusion (unchanged) does not depend on it.
- Launches per step (4.14 forward, 6.34 gradient) and copies per step are unchanged: this step makes
  each launch cheaper, not fewer.
