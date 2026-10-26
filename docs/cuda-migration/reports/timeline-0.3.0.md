# GPU timeline of the fork release 0.3.0

Roadmap step 3.11 (gate G3, test type PRF). Measured on Oct 26, 2026 with `tools/timeline.py` on the
development machine: GeForce RTX 3060 12 GB (which also drives the desktop: Xorg, the editor and a
browser), driver 580.95.05, Nsight Systems 2023.4.4, CuPy 14.2 with the CUDA 13 runtime. PyFWI ran with
its default backend (`auto`: CUDA, `fixed` mode, native kernels, the 4 shots of the small scenario and the
2 of the medium one in one batch).

For each scenario of `tools/benchmark.py` the tool builds the propagator, warms it up (two forward
modellings and a gradient), then runs one forward modelling and one gradient inside the NVTX ranges
`forward` and `gradient`, traced with `--cuda-graph-trace=node` (every kernel replayed by the CUDA graphs
of the native loops appears). It reports the time the GPU was busy (the union of the kernels and copies)
and the idle gaps between them.

## Summary

Small scenario, before and after the changes of this step (several runs each: the GPU also serves the
desktop, which adds isolated pauses of about 1 ms at random places, inside graph replays too):

| Range | Runs | GPU busy | Gaps of 10 us or more | Their total (ms) | Smaller gaps total (ms) |
| --- | --- | --- | --- | --- | --- |
| forward, before (`48eacbd`) | 4 | 80.0 to 85.8 % (median 82.8 %) | 23 to 26 | 1.93 to 3.48 | 1.03 to 1.11 |
| forward, after | 5 | 83.1 to 89.2 % (median 85.6 %) | 7 to 9 | 0.88 to 2.24 | 1.03 to 1.11 |
| gradient, before (`48eacbd`) | 4 | 92.3 to 94.8 % (median 93.8 %) | 22 to 27 | 1.64 to 3.27 | 1.40 to 1.51 |
| gradient, after | 5 | 92.8 to 96.0 % (median 95.4 %) | 19 to 24 | 0.75 to 2.91 | 1.42 to 1.55 |

Medium scenario, after: forward 415.8 ms with the GPU busy 97.0 % of the time (11 gaps of 10 us or more,
5.05 ms; 2.48 ms in smaller ones), gradient 1,040.8 ms busy 98.8 % (25 gaps, 5.21 ms; 3.77 ms).

## Where the GPU waits

- **Inside the time loops, nothing relevant.** Between the kernels of a graph replay the GPU waits a
  fraction of a microsecond (median 0.06 us; about 1.1 ms over the 4,400 gaps of the small forward
  modelling, 4 % of it). The steps between two checkpoints are graphs (step 3.7), and the checkpoints of
  a batch are one launch (step 3.9).
- **Before the loops**, the host prepares the run: the model and the absorbing layer copied into their
  buffers (5 copies of the small grid with their margins, about 0.4 ms), the samples of the source read,
  and in the gradient the 11 buffers of the gradient and of the adjoint field filled (about 0.7 ms).
- **After the loops**, the results are read: the seismograms (5 reads) or the gradients.
- **Isolated pauses of about 1 ms**, one or two per run at random places, also inside graph replays
  where the host does nothing: the GPU drawing the desktop, or the driver; they do not repeat at the
  same place.

Step 3.11 removed the gaps that the first timeline showed in the loops, all on the host:

- **The CUDA graphs were recorded again at every run** (0.2 to 1.1 ms each, a few per run), because the
  model, boundary and gradient buffers were new allocations at every run. A buffer of the same size now
  keeps its allocation, filled exactly as a new one, margins included (`PropagationEngine._set_buffer`),
  and the graphs stay from run to run; the cache is emptied when a buffer is replaced or the undamped
  rectangle of the absorbing layer changes. A reused propagator gives the bits of a new one (test).
- **The seismograms were read shot by shot**, five reads and a wait per shot (20 reads and about 0.6 ms
  at the end of the small forward modelling); the slots of a batch follow each other in the buffers, so
  one read per component now takes the whole batch.
- **The source was resampled at every forward modelling** (0.4 to 0.7 ms of host time): its samples are
  still read at every run, and resampled only when they changed.

Measured by `tools/benchmark.py` (15 runs, same session), the small forward modelling went from 22.3 ms
to 21.9 ms; its gradient, which in the benchmark is the first of a new propagator (it records the
graphs of the gradient), did not change. In an inversion, which calls the gradient of the same
propagator at every iteration, the graphs of the gradient are recorded once.

## Bandwidth of the kernels (estimate)

Gate G3 asks for the bandwidth of every kernel from Nsight Compute, which needs the GPU performance
counters; the driver restricts them to root on this machine (`RmProfilingAdminOnly: 1`), an action of
the project owner (roadmap, pending action 1). Until then, `tools/timeline.py` estimates it from the
durations of the timeline: the bytes a kernel must move at least (every array it reads or writes once
per cell: `ARRAYS_MOVED`), divided by its duration. It ignores the extra traffic of the stencils'
neighbours that miss the caches, so the real bandwidth is higher. Medium scenario (2 shots in one batch,
394,240 threads per launch), the peak of the RTX 3060 being 360 GB/s:

| Kernel | Mean duration (us) | Arrays moved per cell | Estimated GB/s | Share of the peak |
| --- | --- | --- | --- | --- |
| update_v (forward) | 49.8 | 8 | 253 | 70 % |
| update_tau (forward) | 68.2 | 10 | 231 | 64 % |
| grad_adjoint_v | 166.5 | 27 | 256 | 71 % |
| Adj_update_tau | 55.1 | 9 | 258 | 72 % |
| copy_checkpoint | 61.4 | 10 | 257 | 71 % |

The hot kernels move at least 64 to 72 % of the peak bandwidth; `copy_checkpoint`, a plain copy without
arithmetic, reaches the same rate: the stencil kernels move their data about as fast as a copy does here.

## Detail: small scenario, after

### forward

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 337.4 | 0.44 | copy 209472 B | MakeAllZero |
| 210.7 | 0.92 | copy_checkpoint | inject_and_record |
| 148.8 | 0.21 | copy 209472 B | copy 209472 B |
| 101.8 | 0.79 | copy 4 B | inject_and_record |
| 28.4 | 0.18 | copy 209472 B | copy 209472 B |
| 25.5 | 0.37 | copy 209472 B | copy 209472 B |
| 24.7 | 0.41 | copy 209472 B | copy 209472 B |

| Kernel or copy | Count | Total (ms) | Mean (us) |
| --- | --- | --- | --- |
| update_tau | 1050 | 8.45 | 8.04 |
| update_v | 1050 | 7.74 | 7.37 |
| copy_checkpoint | 210 | 2.24 | 10.65 |
| inject_and_record | 1050 | 2.11 | 2.01 |
| advance | 1050 | 1.06 | 1.01 |
| copy 230400 B | 5 | 0.08 | 16.96 |
| copy 209472 B | 5 | 0.05 | 9.90 |
| MakeAllZero | 1 | 0.00 | 4.83 |
| copy 16 B | 1 | 0.00 | 0.35 |
| copy 4 B | 1 | 0.00 | 0.35 |

### gradient

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 240.9 | 0.82 | copy 446912 B | MakeGradZero |
| 213.5 | 5.73 | advance | copy_checkpoint |
| 70.8 | 65.12 | copy 78400 B | copy 78400 B |
| 49.6 | 0.62 | copy 446912 B | copy 446912 B |
| 49.2 | 0.13 | copy 446912 B | copy 446912 B |
| 48.4 | 1.23 | copy 4 B | copy_checkpoint |
| 46.9 | 0.55 | copy 446912 B | copy 446912 B |
| 46.7 | 0.20 | copy 446912 B | copy 446912 B |
| 46.2 | 0.41 | copy 446912 B | copy 446912 B |
| 44.5 | 0.34 | copy 446912 B | copy 446912 B |
| 44.2 | 0.48 | copy 446912 B | copy 446912 B |
| 44.0 | 0.27 | copy 446912 B | copy 446912 B |
| 41.2 | 0.69 | copy 446912 B | copy 446912 B |
| 40.4 | 0.75 | copy 446912 B | copy 446912 B |
| 29.6 | 1.16 | copy 230400 B | copy 230400 B |
| 29.5 | 65.20 | copy 78400 B | copy 78400 B |
| 24.6 | 1.13 | copy 230400 B | copy 230400 B |
| 13.3 | 1.20 | copy 230400 B | copy 16 B |
| 10.4 | 1.29 | copy_checkpoint | grad_adjoint_v |

| Kernel or copy | Count | Total (ms) | Mean (us) |
| --- | --- | --- | --- |
| grad_adjoint_v | 1049 | 26.33 | 25.10 |
| update_tau | 840 | 13.13 | 15.63 |
| Adj_update_tau | 1049 | 10.11 | 9.63 |
| update_v | 840 | 5.83 | 6.94 |
| copy_checkpoint | 209 | 3.60 | 17.22 |
| inject_residual | 1049 | 2.20 | 2.09 |
| advance | 1049 | 1.02 | 0.97 |
| copy 446912 B | 11 | 0.25 | 23.18 |
| copy 230400 B | 5 | 0.06 | 11.85 |
| copy 78400 B | 3 | 0.01 | 4.04 |
| sum_slots | 1 | 0.01 | 11.97 |
| MakeAllZero | 2 | 0.01 | 5.17 |
| MakeGradZero | 1 | 0.01 | 6.21 |
| copy 16 B | 1 | 0.00 | 0.35 |
| copy 4 B | 1 | 0.00 | 0.35 |

## Detail: medium scenario, after

### forward

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 1152.3 | 141.36 | advance | copy_checkpoint |
| 1094.9 | 88.71 | advance | copy_checkpoint |
| 1088.1 | 172.33 | advance | inject_and_record |
| 672.3 | 1.84 | copy 905472 B | MakeAllZero |
| 477.9 | 1.12 | copy 905472 B | copy 905472 B |
| 206.0 | 194.36 | advance | copy_checkpoint |
| 104.8 | 2.54 | copy 4 B | inject_and_record |
| 74.1 | 2.80 | copy_checkpoint | inject_and_record |
| 62.3 | 1.01 | copy 905472 B | copy 905472 B |
| 59.1 | 1.64 | copy 905472 B | copy 905472 B |
| 55.6 | 1.74 | copy 905472 B | copy 905472 B |

| Kernel or copy | Count | Total (ms) | Mean (us) | Threads | Estimated GB/s |
| --- | --- | --- | --- | --- | --- |
| update_tau | 2998 | 204.35 | 68.16 | 394240 | 231 |
| update_v | 2998 | 149.24 | 49.78 | 394240 | 253 |
| copy_checkpoint | 599 | 36.78 | 61.40 | 394240 | 257 |
| inject_and_record | 2998 | 8.39 | 2.80 | 1664 |  |
| advance | 2998 | 3.79 | 1.26 | 64 |  |
| copy 1359072 B | 5 | 0.42 | 84.53 |  |  |
| copy 905472 B | 5 | 0.22 | 44.50 |  |  |
| MakeAllZero | 1 | 0.02 | 23.33 | 394240 |  |
| copy 4 B | 1 | 0.00 | 0.35 |  |  |
| copy 8 B | 1 | 0.00 | 0.32 |  |  |

### gradient

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 990.2 | 875.29 | advance | update_tau |
| 984.4 | 908.37 | update_v | grad_adjoint_v |
| 569.6 | 2.49 | copy 1683392 B | MakeGradZero |
| 229.5 | 1037.42 | copy 774400 B | copy 774400 B |
| 210.8 | 178.25 | advance | update_tau |
| 209.2 | 913.73 | advance | copy_checkpoint |
| 204.1 | 826.87 | grad_adjoint_v | Adj_update_tau |
| 155.6 | 3.31 | copy 1359072 B | copy 1359072 B |
| 125.1 | 3.89 | copy 1359072 B | copy 1359072 B |
| 115.9 | 0.37 | copy 1683392 B | copy 1683392 B |
| 114.8 | 1037.69 | copy 774400 B | copy 774400 B |
| 114.3 | 2.05 | copy 1683392 B | copy 1683392 B |
| 113.3 | 3.70 | copy 1359072 B | copy 1359072 B |
| 106.9 | 2.27 | copy 1683392 B | copy 1683392 B |
| 106.7 | 0.59 | copy 1683392 B | copy 1683392 B |
| 105.5 | 0.80 | copy 1683392 B | copy 1683392 B |
| 105.4 | 1.01 | copy 1683392 B | copy 1683392 B |
| 105.2 | 1.63 | copy 1683392 B | copy 1683392 B |
| 105.0 | 1.84 | copy 1683392 B | copy 1683392 B |
| 104.0 | 1.22 | copy 1683392 B | copy 1683392 B |
| 103.8 | 1.43 | copy 1683392 B | copy 1683392 B |
| 87.3 | 4.11 | copy 4 B | copy_checkpoint |
| 83.2 | 3.54 | copy 1359072 B | copy 1359072 B |
| 54.8 | 3.17 | MakeAllZero | copy 1359072 B |
| 10.4 | 4.10 | copy 8 B | copy 4 B |

| Kernel or copy | Count | Total (ms) | Mean (us) | Threads | Estimated GB/s |
| --- | --- | --- | --- | --- | --- |
| grad_adjoint_v | 2997 | 498.97 | 166.49 | 394240 | 256 |
| update_tau | 2399 | 184.47 | 76.90 | 394240 | 205 |
| Adj_update_tau | 2997 | 165.00 | 55.05 | 394240 | 258 |
| update_v | 2399 | 133.92 | 55.83 | 394240 | 226 |
| copy_checkpoint | 598 | 35.49 | 59.34 | 394240 | 266 |
| inject_residual | 2997 | 5.85 | 1.95 | 1664 |  |
| advance | 2997 | 3.08 | 1.03 | 64 |  |
| copy 1683392 B | 11 | 1.15 | 104.36 |  |  |
| copy 1359072 B | 5 | 0.40 | 79.33 |  |  |
| copy 774400 B | 3 | 0.11 | 38.07 |  |  |
| MakeAllZero | 2 | 0.07 | 34.69 | 394240 |  |
| sum_slots | 1 | 0.05 | 48.06 | 197120 |  |
| MakeGradZero | 1 | 0.04 | 42.08 | 394240 |  |
| copy 8 B | 1 | 0.00 | 0.35 |  |  |
| copy 4 B | 1 | 0.00 | 0.32 |  |  |
