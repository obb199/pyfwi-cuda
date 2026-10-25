# Reference benchmark of PyFWI 0.1.10

Roadmap step 0.7. Measured on Oct 25, 2026 with `tools/benchmark.py` (median of 5 runs after a
warm-up, inside zero guard bands; see the tool's docstring) on the development machine: GeForce RTX
3060 12 GB, driver 580.95.05, Ryzen 7 5700X. Raw results: `bench/baseline/*.json`.

Two runs of the **same code** (PyFWI 0.1.10) with different environments:

- `ref-0.1.10-pyopencl-2022`: the frozen reference environment (Python 3.10, NumPy 1.21, PyOpenCL
  2022.2.4) with PyFWI 0.1.10 from PyPI. **This is the baseline every later phase is compared with.**
- `dev-pyopencl-2026`: the development environment (Python 3.12, NumPy 2.5, PyOpenCL 2026.1.4) with
  the fork, whose code at that point computed exactly what 0.1.10 computes.

## Results

| Run | Device | PyFWI | PyOpenCL | NumPy | Date |
| --- | --- | --- | --- | --- | --- |
| ref-0.1.10-pyopencl-2022 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | 2022.2.4 | 1.21.6 | 2026-10-25T17:23:58+00:00 |
| dev-pyopencl-2026 | NVIDIA GeForce RTX 3060 (driver 580.95.05) | 0.1.10 | 2026.1.4 | 2.5.4 | 2026-10-25T17:12:45+00:00 |

Times are medians of the measured runs, scaled to the number of shots of the scenario. Speed-up is relative to the first run.

| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | ref-0.1.10-pyopencl-2022 | 1.50 | 1.91 | 1.00 / 1.00 | 0.356 / 0.454 | 4.14 / 6.34 | 2.71 | 0.011 / 0.011 | 59.0 / 55.0 |
| small | dev-pyopencl-2026 | 2.16 | 3.05 | 0.69 / 0.62 | 0.515 / 0.727 | 4.14 / 6.34 | 2.71 | 0.018 / 0.018 | 55.0 / 62.0 |
| medium (scaled x4 to 8 shots) | ref-0.1.10-pyopencl-2022 | 47.33 | 57.32 | 1.00 / 1.00 | 1.973 / 2.390 | 4.14 / 6.34 | 2.71 | 0.003 / 0.027 | 63.0 / 64.0 |
| medium (scaled x4 to 8 shots) | dev-pyopencl-2026 | 48.80 | 58.27 | 0.97 / 0.98 | 2.035 / 2.430 | 4.14 / 6.34 | 2.71 | 0.005 / 0.004 | 65.0 / 63.0 |
| large (scaled x50 to 50 shots) | ref-0.1.10-pyopencl-2022 | 769.99 | 1541.96 | 1.00 / 1.00 | 2.567 / 5.142 | 4.14 / 6.94 | 1.21 | 0.023 / 0.022 | 69.0 / 69.0 |
| large (scaled x50 to 50 shots) | dev-pyopencl-2026 | 797.87 | 1617.23 | 0.97 / 0.95 | 2.660 / 5.393 | 4.14 / 6.94 | 1.21 | 0.014 / 0.024 | 69.0 / 70.0 |

## Reading the numbers

- **All coefficients of variation are below 3 %**, the criterion of gate G0.
- **The small scenario is bound by the host.** About 0.36 ms of host time per internal step in the
  forward run and 0.45 ms in the gradient, for 4.14 and 6.34 kernel launches per step and 2.7 blocking
  copies per step. The GPU work per step is a small fraction of that (see `tools/opencl_profile.py`).
- **PyOpenCL 2026 is slower for the same code**: +44 % (forward) and +60 % (gradient) in the small
  scenario, where launch overhead dominates; within 5 % in the medium and large scenarios. This is the
  cost of re-creating a kernel object at every launch (`RepeatedKernelRetrieval`), which step 1.3
  removes. Gains of the fork are measured against the `ref-0.1.10-pyopencl-2022` row, so they include
  recovering this regression.
- **The medium and large scenarios are scaled** from 2 of 8 and 1 of 50 shots (times are linear in
  the number of shots). The large one uses `chpr = 5` because 20 % of its 6,000 steps would need
  12 GB of RAM per shot in 0.1.10. Its host time per gradient step (5.1-5.4 ms) is twice that of the
  forward, because with fewer checkpoints more of the forward field is rebuilt.
- With the 0.1.10 design, the full large scenario (50 shots) takes about 13 minutes per forward and
  26 minutes per gradient on this RTX 3060.
