# PyFWI-CUDA user guide

For people who use PyFWI and want to run it with this fork: what changes for you, what does not, which
engine runs your code, why its results can differ from those of PyFWI 0.1.10 and by how much. The
[README](../../README.md) has the installation commands and the list of settings; the
[overview](overview.md) tells the story of the project with figures.

## In one minute

- **Your code does not change.** The fork is imported as `PyFWI`, with the same modules, classes and
  arguments as PyFWI 0.1.10. The examples of the original repository (the notebooks of `example/`) run
  unchanged; the test suite runs them.
- **The engine is chosen outside your code.** With a CUDA extra installed (`pip install ".[cuda13]"`) and
  an NVIDIA GPU, PyFWI runs on CUDA, 17 to 77 times faster than 0.1.10 for a forward modelling and 12 to 32
  times for a gradient ([benchmark](reports/benchmark-0.4.0.md)). Otherwise it runs the OpenCL engine of
  0.1.10, with the host-side fixes of the fork: 1.1 to 3.8 times faster than 0.1.10 itself.
- **The two engines start in different numeric modes.** OpenCL computes exactly what PyFWI 0.1.10 computes
  (the `legacy` mode). CUDA corrects the numerical bugs of 0.1.10 (the `fixed` mode). The same mode gives
  the same results on both engines, up to rounding; the two modes give different results (below).
- **`print(PyFWI.info())`** says which engine and which mode run. Attach it to any question or bug report.

## Which engine runs

Every propagator (`WavePropagator`, `FWI`, `TimeLapse`, `torchfwi.Fwi`) chooses its engine when it is
created, from, in this order:

1. `inpa["backend"]`, if your parameters set it;
2. the environment variable `PYFWI_BACKEND`;
3. `auto`, the default.

| Value | What runs |
| --- | --- |
| `auto` | CUDA when the OpenCL device PyFWI would use is an NVIDIA GPU (or there is no OpenCL device) and CUDA can run (CuPy installed, a working driver); otherwise OpenCL, and `PyFWI.info()` says why |
| `opencl` | The OpenCL engine of PyFWI 0.1.10 |
| `cuda` | CUDA, falling back to OpenCL with a warning when CUDA cannot run |
| `cuda:strict` | CUDA, or an error that says why it cannot run (for scripts and clusters, where a silent fallback would be slow) |

```bash
PYFWI_BACKEND=opencl python my_inversion.py      # the results of PyFWI 0.1.10, bit for bit
PYFWI_BACKEND=cuda:strict python my_inversion.py # CUDA or a clear error
```

OpenCL is the right choice when there is no NVIDIA GPU (an AMD or Intel GPU, or POCL on the CPU), when CuPy
cannot be installed, or when you need the results of 0.1.10 exactly, for example to compare with a study made
with it.

## The numeric modes

PyFWI 0.1.10 has numerical bugs: some make results slightly wrong (the time step truncated to 6 decimals,
the coefficients of order 8 rounded), some make a part of the results wrong (the gradients of `vs` and of the
density, the `vz` data of crosswell surveys), and one makes the inversion stop before it starts (absolute
tolerances in the optimisers, with a source of unit amplitude). Fixing them changes the results, so the fork
keeps both behaviours:

- **`legacy`**: what PyFWI 0.1.10 computes, bugs included. The default of OpenCL, so that a plain
  installation of the fork gives the results of 0.1.10 (the test suite checks it against results stored
  from 0.1.10 itself).
- **`fixed`**: the bugs corrected. The default of CUDA, which a user gets only by installing a CUDA extra.

Choose the mode with `inpa["numerics"]` or `PYFWI_NUMERICS` (`legacy` or `fixed`); either engine runs
either mode. What `fixed` corrects (details and measurements in the [CHANGELOG](../../CHANGELOG.md),
release 0.2.0):

| Bug of 0.1.10 | Effect in `legacy` | In `fixed` |
| --- | --- | --- |
| The time step truncated to 6 decimals, double-precision constants | Waves 0.075 % slower than the source they come from (with `dt` = 4 ms); slow double-precision arithmetic on GeForce cards | The full time step; single precision throughout |
| The source added once every `dt_scale` internal steps | Grid-scale noise in the seismograms, 3 to 6 % at 10 m cells, dominant at 5 m | The wavelet resampled to the internal step, injected at every step |
| Residual samples injected one record early for some recording lengths | Gradients changed by about 50 % for those lengths (`dt` = 4 ms with `t` = 0.2 or 0.4 s) | Each residual sample injected where it was recorded |
| The gradient loop not the adjoint of the forward step, adjoint fields damped differently, a wrong derivative in the `mu` gradient | Gradients against finite differences (median error): vp 6.9 %, vs 60 %, density 135 % | vp 0.6 %, vs 0.27 %, density 1.4 % |
| Order-8 coefficients rounded to integers | Waves 0.14 % too fast with `sdo=8` | The exact coefficients |
| `vx` recorded in the `vz` data of crosswell surveys (`acq_type=0`) | The `vz` data, misfit and gradient are another quantity | `vz` recorded |
| Absolute tolerances in `FWI`'s optimisers | With a unit-amplitude source, L-BFGS and the conjugate gradient can return the starting model | Tolerances relative to the first gradient of each frequency band |

The kernels of 0.1.10 also read and write outside their buffers, which gave NaN or random results on some
OpenCL drivers (POCL, on the CPU). In the fork this is harmless in both modes: every buffer has a margin of
zeros around it; `fixed` also removes the accesses.

How much the results differ between the modes: up to 6 % on seismograms and up to 45 % on raw gradients
(the `mu` gradient; the propagation cases of the golden masters); on complete inversions, up to 15 % of the
recovered change ([geophysical validation](reports/geophysical-validation-0.4.0.dev0.md)). The `fixed` mode is
the one whose gradients match finite differences; whether OpenCL should also default to it is waiting for
the review of a geophysicist (gate [G4](gates/G4.md)).

## How close the two engines are

In the same numeric mode, CUDA and OpenCL compute the same thing with different compilers, and the native
CUDA kernels order a few operations differently, so the results differ by rounding
([parity report](reports/parity-0.4.0.md), relative L2 differences on the whole matrix of test cases):

| Output | `legacy` mode | `fixed` mode |
| --- | --- | --- |
| Seismograms | 1.1e-6 at most | 1.1e-6 at most |
| Gradients | 1.9e-5 at most | 3.6e-4 at most (in the case most sensitive to rounding) |
| Short inversions (2 iterations) | models 3.3e-5, misfits 8.2e-5 | models 3.4e-5, misfits 4.7e-5 |

A long inversion amplifies these differences, as any change of rounding would (another computer, another
driver): after 45 L-BFGS iterations the estimates of the two engines differ by up to 1.7 % of what the
inversion recovers, and the geophysical measures of the result (model error, correlation with the true
change) by at most 0.008 ([geophysical validation](reports/geophysical-validation-0.4.0.dev0.md)).

On the same machine, two runs give the same bits, with the shots one at a time, in batches or shared between
processes (checked by the tests, with the sources off the receiver cells). With the default geometry of
`acq_parameters` a source sits on a receiver cell, a race inherited from 0.1.10 that can change the last bits
from run to run on OpenCL; the example scripts move the sources one row up.

## Optional features

Off by default; each is turned on by an environment variable and documented with its tolerance in the
[CHANGELOG](../../CHANGELOG.md) (release 0.4.0) and in the [README](../../README.md#tuning-the-cuda-backend):

| Feature | Variable | What you gain | What it costs |
| --- | --- | --- | --- |
| Cost function and filters on the GPU | `PYFWI_CUDA_COST=1` | An inversion about 30 % faster on small models | Results within 1e-6 |
| Boundary saving | `PYFWI_BOUNDARY_SAVING=1` | 4 to 11 times less GPU memory, faster when the checkpoints did not fit | A gradient up to 18 % slower; CUDA, `fixed` mode only |
| FP16 fields | `PYFWI_FP16=1` | Half the memory, up to 1.36 times faster | Seismograms at least 44 dB from float32; **not for inversions**, which do not converge; CUDA, `fixed` mode only |
| Shots in several processes | `PYFWI_CUDA_DEVICES=0,1` | One process per GPU | Nothing measurable with one GPU; same results |
| CUDA tensors in `torchfwi` | tensors on a CUDA device | Gradients for PyTorch models on the GPU | The model and results still cross the host memory |

## When something goes wrong

- **`cuda:strict` refuses to run**: the message says why (no CuPy, no NVIDIA GPU, a driver older than the
  CUDA extra needs: 580 for `cuda13`, 525 for `cuda12`). `python tools/check_env.py` checks the whole chain.
- **The GPU memory runs out**: the checkpoints of the forward field take most of it. PyFWI keeps on the GPU
  what fits in 80 % of the free memory and the rest in host memory (`PYFWI_CUDA_CHECKPOINT_BUDGET` changes
  the share); boundary saving needs far less.
- **`PYFWI_BOUNDARY_SAVING` or `PYFWI_FP16` raise an error**: they need the CUDA engine in the `fixed` mode.
- **An FP16 run raises `FloatingPointError`**: a field exceeded half precision; run it without FP16.
- **Results differ from a colleague's**: compare `PyFWI.info()` first: another engine is a difference of
  rounding, another numeric mode a real difference (above).
- **The original PyFWI and the fork in one environment**: both install the package `PyFWI`; uninstall one
  before installing the other.
