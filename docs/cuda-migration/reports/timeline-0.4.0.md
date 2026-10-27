# GPU timeline of the fork release 0.4.0

Roadmap step 4.7 (gate G4, test type PRF). Measured on Oct 27, 2026 with `tools/timeline.py` on the
development machine: GeForce RTX 3060 12 GB (which also drives the desktop), driver 580.95.05, Nsight Systems
2023.4.4, CuPy 14.2 with the CUDA 13 runtime. PyFWI ran with its default backend (`auto`: CUDA, `fixed` mode,
native kernels, the 4 shots of the small scenario and the 2 of the medium one in one batch), then with the
opt-in features of Phase 4 that change the propagation. How the tool measures: see the
[timeline report of 0.3.0](timeline-0.3.0.md) (one forward modelling and one gradient after a warm-up, every
node of the CUDA graphs traced; the time the GPU is busy and the idle gaps between its kernels and copies).

## Summary

The default path did not change in Phase 4, and the timeline neither: the GPU is as busy as in 0.3.0, inside
the time loops only the sub-microsecond gaps between graph nodes remain. With boundary saving and with FP16 the
GPU stays as busy (the shorter FP16 runs make the fixed gaps before and after the loops weigh a little more).

| Configuration | Range | Runs | GPU busy | Gaps of 10 us or more | Their total (ms) |
| --- | --- | --- | --- | --- | --- |
| default, small | forward | 5 | 88.3 to 89.1 % (median 88.8 %; 0.3.0: median 85.6 %) | 7 to 10 | 0.69 to 0.91 |
| default, small | gradient | 5 | 94.9 to 95.8 % (median 95.5 %; 0.3.0: median 95.4 %) | 21 to 23 | 0.62 to 1.21 |
| default, medium | forward | 2 | 97.7 % (0.3.0: 97 %) | 8 to 9 | 1.42 to 1.54 |
| default, medium | gradient | 2 | 99.0 % (0.3.0: 99 %) | 20 to 22 | 2.26 to 2.41 |
| boundary saving, small | forward / gradient | 1 | 84.1 % / 96.6 % | 8 / 21 | 2.31 / 1.09 |
| boundary saving, medium | forward / gradient | 1 | 97.5 % / 99.2 % | 6 / 21 | 3.55 / 2.58 |
| FP16, small | forward / gradient | 1 | 84.9 % / 94.7 % | 9 / 21 | 0.92 / 0.70 |
| FP16, medium | forward / gradient | 1 | 96.3 % / 98.7 % | 8 / 21 | 2.78 / 2.51 |

Boundary saving launches three small kernels more per step (`save_band` in the forward modelling,
`restore_band` twice and `remove_source` in the gradient: 5,264 and 9,462 intervals against 4,422 and 6,109),
inside the CUDA graphs: they add no gap of 10 us or more. Its forward modelling of the small scenario has one
longer gap, before the time loop, between the last small copy and the first injection (host work: 2.3 ms in a
detailed run, against 0.09 ms by default; not investigated, the benchmark measures the same forward time with
and without boundary saving). The bandwidth of the kernels is
still an estimate from the timeline: Nsight Compute cannot read the GPU counters (owner action, open since G3).

## The default configuration, one run of each scenario

## Timeline of the small scenario

| Range | Length (ms) | GPU busy | Gaps | Median gap (us) | Gaps >= 10 us | Their total (ms) | Smaller gaps total (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward | 20.83 | 88.6 % | 4422 | 0.06 | 9 | 0.71 | 1.09 |
| gradient | 59.83 | 95.4 % | 6109 | 0.06 | 22 | 0.89 | 1.47 |

### forward

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 221.6 | 0.35 | copy 209472 B | MakeAllZero |
| 177.7 | 0.72 | copy_checkpoint | inject_and_record |
| 120.9 | 0.16 | copy 209472 B | copy 209472 B |
| 92.1 | 0.60 | copy 4 B | inject_and_record |
| 24.2 | 0.13 | copy 209472 B | copy 209472 B |
| 22.8 | 0.29 | copy 209472 B | copy 209472 B |
| 20.5 | 0.32 | copy 209472 B | copy 209472 B |
| 15.9 | 0.58 | MakeAllZero | copy 16 B |
| 10.1 | 0.59 | copy 16 B | copy 4 B |

| Kernel or copy | Count | Total (ms) | Mean (us) | Threads | Estimated GB/s |
| --- | --- | --- | --- | --- | --- |
| update_tau | 1050 | 7.16 | 6.82 | 92160 | 540 |
| update_v | 1050 | 5.90 | 5.62 | 92160 | 525 |
| inject_and_record | 1050 | 2.18 | 2.08 | 768 |  |
| copy_checkpoint | 210 | 2.02 | 9.64 | 92160 | 382 |
| advance | 1050 | 1.08 | 1.03 | 64 |  |
| copy 230400 B | 5 | 0.06 | 12.24 |  |  |
| copy 209472 B | 5 | 0.05 | 9.34 |  |  |
| MakeAllZero | 1 | 0.00 | 4.45 | 92160 |  |
| copy 16 B | 1 | 0.00 | 0.35 |  |  |
| copy 4 B | 1 | 0.00 | 0.35 |  |  |

### gradient

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 235.7 | 21.96 | copy_checkpoint | grad_adjoint_v |
| 50.9 | 59.45 | copy 78400 B | copy 78400 B |
| 44.0 | 0.62 | copy 446912 B | MakeGradZero |
| 41.5 | 0.91 | copy 4 B | copy_checkpoint |
| 36.8 | 0.71 | MakeAllZero | copy 230400 B |
| 36.7 | 0.12 | copy 446912 B | copy 446912 B |
| 34.3 | 0.17 | copy 446912 B | copy 446912 B |
| 32.6 | 0.22 | copy 446912 B | copy 446912 B |
| 31.6 | 0.42 | copy 446912 B | copy 446912 B |
| 31.5 | 0.47 | copy 446912 B | copy 446912 B |
| 31.4 | 0.75 | copy 230400 B | copy 230400 B |
| 31.1 | 0.27 | copy 446912 B | copy 446912 B |
| 31.0 | 0.37 | copy 446912 B | copy 446912 B |
| 30.8 | 0.32 | copy 446912 B | copy 446912 B |
| 30.8 | 0.52 | copy 446912 B | copy 446912 B |
| 30.7 | 0.57 | copy 446912 B | copy 446912 B |
| 29.0 | 59.51 | copy 78400 B | copy 78400 B |
| 22.5 | 0.83 | copy 230400 B | copy 230400 B |
| 22.0 | 0.79 | copy 230400 B | copy 230400 B |
| 21.3 | 0.86 | copy 230400 B | copy 230400 B |
| 20.5 | 0.67 | MakeGradZero | MakeAllZero |
| 10.6 | 0.89 | copy 230400 B | copy 16 B |

| Kernel or copy | Count | Total (ms) | Mean (us) | Threads | Estimated GB/s |
| --- | --- | --- | --- | --- | --- |
| grad_adjoint_v | 1049 | 25.12 | 23.95 | 92160 | 416 |
| update_tau | 840 | 11.97 | 14.25 | 92160 | 259 |
| Adj_update_tau | 1049 | 9.38 | 8.95 | 92160 | 371 |
| update_v | 840 | 4.84 | 5.77 | 92160 | 511 |
| copy_checkpoint | 209 | 2.54 | 12.15 | 92160 | 303 |
| inject_residual | 1049 | 1.91 | 1.82 | 768 |  |
| advance | 1049 | 1.03 | 0.98 | 64 |  |
| copy 446912 B | 11 | 0.20 | 18.40 |  |  |
| copy 230400 B | 5 | 0.05 | 10.08 |  |  |
| copy 78400 B | 3 | 0.01 | 4.05 |  |  |
| sum_slots | 1 | 0.01 | 12.06 | 22400 |  |
| MakeAllZero | 2 | 0.01 | 5.22 | 92160 |  |
| MakeGradZero | 1 | 0.01 | 6.08 | 92160 |  |
| copy 4 B | 1 | 0.00 | 0.38 |  |  |
| copy 16 B | 1 | 0.00 | 0.35 |  |  |

## Timeline of the medium scenario

| Range | Length (ms) | GPU busy | Gaps | Median gap (us) | Gaps >= 10 us | Their total (ms) | Smaller gaps total (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward | 367.49 | 97.7 % | 12603 | 0.00 | 8 | 1.42 | 2.47 |
| gradient | 908.81 | 99.0 % | 17408 | 0.00 | 22 | 2.41 | 3.70 |

### forward

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 566.3 | 1.85 | copy 905472 B | MakeAllZero |
| 436.6 | 1.19 | copy 905472 B | copy 905472 B |
| 160.3 | 2.45 | copy 4 B | inject_and_record |
| 64.4 | 1.09 | copy 905472 B | copy 905472 B |
| 61.2 | 1.66 | copy 905472 B | copy 905472 B |
| 61.1 | 2.77 | copy_checkpoint | inject_and_record |
| 58.6 | 1.76 | copy 905472 B | copy 905472 B |
| 10.3 | 2.44 | copy 8 B | copy 4 B |

| Kernel or copy | Count | Total (ms) | Mean (us) | Threads | Estimated GB/s |
| --- | --- | --- | --- | --- | --- |
| update_tau | 2998 | 181.90 | 60.67 | 394240 | 260 |
| update_v | 2998 | 135.61 | 45.23 | 394240 | 279 |
| copy_checkpoint | 599 | 29.59 | 49.41 | 394240 | 319 |
| inject_and_record | 2998 | 7.86 | 2.62 | 1664 |  |
| advance | 2998 | 3.52 | 1.17 | 64 |  |
| copy 1359072 B | 5 | 0.35 | 69.71 |  |  |
| copy 905472 B | 5 | 0.18 | 35.90 |  |  |
| MakeAllZero | 1 | 0.02 | 23.14 | 394240 |  |
| copy 8 B | 1 | 0.00 | 0.35 |  |  |
| copy 4 B | 1 | 0.00 | 0.32 |  |  |

### gradient

Gaps of 10 us or more, longest first:

| Gap (us) | At (ms) | Before the gap | After the gap |
| --- | --- | --- | --- |
| 350.5 | 2.10 | copy 1683392 B | MakeGradZero |
| 217.3 | 668.01 | copy_checkpoint | grad_adjoint_v |
| 170.8 | 905.55 | copy 774400 B | copy 774400 B |
| 148.6 | 2.70 | copy 1359072 B | copy 1359072 B |
| 144.3 | 335.60 | Adj_update_tau | inject_residual |
| 103.4 | 0.40 | copy 1683392 B | copy 1683392 B |
| 103.0 | 2.54 | MakeAllZero | copy 1359072 B |
| 99.8 | 0.58 | copy 1683392 B | copy 1683392 B |
| 95.0 | 0.76 | copy 1683392 B | copy 1683392 B |
| 93.4 | 1.27 | copy 1683392 B | copy 1683392 B |
| 93.3 | 0.93 | copy 1683392 B | copy 1683392 B |
| 92.9 | 1.10 | copy 1683392 B | copy 1683392 B |
| 91.8 | 1.44 | copy 1683392 B | copy 1683392 B |
| 90.4 | 1.61 | copy 1683392 B | copy 1683392 B |
| 83.1 | 1.78 | copy 1683392 B | copy 1683392 B |
| 82.2 | 1.94 | copy 1683392 B | copy 1683392 B |
| 79.4 | 905.75 | copy 774400 B | copy 774400 B |
| 72.7 | 3.28 | copy 4 B | copy_checkpoint |
| 63.0 | 2.91 | copy 1359072 B | copy 1359072 B |
| 62.3 | 3.03 | copy 1359072 B | copy 1359072 B |
| 61.0 | 3.15 | copy 1359072 B | copy 1359072 B |
| 14.3 | 2.48 | MakeGradZero | MakeAllZero |

| Kernel or copy | Count | Total (ms) | Mean (us) | Threads | Estimated GB/s |
| --- | --- | --- | --- | --- | --- |
| grad_adjoint_v | 2997 | 442.44 | 147.63 | 394240 | 288 |
| update_tau | 2399 | 153.28 | 63.89 | 394240 | 247 |
| Adj_update_tau | 2997 | 143.95 | 48.03 | 394240 | 295 |
| update_v | 2399 | 119.50 | 49.81 | 394240 | 253 |
| copy_checkpoint | 598 | 29.95 | 50.08 | 394240 | 315 |
| inject_residual | 2997 | 6.11 | 2.04 | 1664 |  |
| advance | 2997 | 2.84 | 0.95 | 64 |  |
| copy 1683392 B | 11 | 0.85 | 77.69 |  |  |
| copy 1359072 B | 5 | 0.30 | 59.47 |  |  |
| copy 774400 B | 3 | 0.09 | 30.48 |  |  |
| sum_slots | 1 | 0.05 | 48.22 | 197120 |  |
| MakeAllZero | 2 | 0.05 | 23.33 | 394240 |  |
| MakeGradZero | 1 | 0.03 | 27.97 | 394240 |  |
| copy 8 B | 1 | 0.00 | 0.35 |  |  |
| copy 4 B | 1 | 0.00 | 0.32 |  |  |
