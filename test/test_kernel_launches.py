"""How the propagator launches its OpenCL kernels (roadmap step 1.3).

PyFWI 0.1.10 launched every kernel as ``self.prg.<name>(queue, size, None, *arguments)``. Each
``self.prg.<name>`` created a new kernel object (``clCreateKernel``), and each call set every
argument again, about five times per time step: the host spent more time preparing launches than
the RTX spent computing in small models (analysis, section 2). Since step 1.3 the fork:

* retrieves each kernel once, when the propagator is built (``WavePropagator.knl``);
* sets the arguments that do not change during a time loop once per forward modelling or
  gradient (``set_args``), and only the source samples and the time index at each injection;
* launches the time-loop kernels with ``pyopencl.enqueue_nd_range_kernel``.

The results do not change (same kernels, same order, same arguments): the characterisation tests
check that bit for bit. These tests check the launches themselves, and that a propagator reused
for several runs (new model buffers, coefficient +1 then -1) gives the results of a fresh one.

The first tests are expected to fail with the published 0.1.10 (reference environment).
"""
import collections
import warnings

import numpy as np
import pytest

from helpers import forward_and_gradient, running_reference, tiny_case
from oob_guard import zero_guard_bands

#: Marker of the tests that 0.1.10 fails: it retrieves kernels and sets arguments at every launch.
fixed_in_step_1_3 = pytest.mark.xfail(condition=running_reference(), strict=True, raises=AssertionError,
                                      reason="0.1.10 retrieves each kernel and sets all its arguments at every launch")


@fixed_in_step_1_3
def test_no_kernel_is_retrieved_while_propagating(opencl_platform, monkeypatch):
    import pyopencl as cl
    case = tiny_case()  # builds the propagator: the kernels are retrieved here
    retrieved = []
    original = cl.Program.__getattr__

    def recording(program, name):
        retrieved.append(name)
        return original(program, name)

    monkeypatch.setattr(cl.Program, "__getattr__", recording)
    forward_and_gradient(case)
    assert not retrieved, f"kernels retrieved during forward modelling and gradient: {sorted(set(retrieved))}"


def test_no_repeated_kernel_retrieval_warning(opencl_platform):
    # PyOpenCL 2024 and newer warn when a program gives the same kernel twice
    # (https://github.com/inducer/pyopencl/issues/831); 0.1.10 does it at every launch.
    import pyopencl as cl
    if not hasattr(cl, "RepeatedKernelRetrieval"):
        pytest.skip("this PyOpenCL has no RepeatedKernelRetrieval warning")
    case = tiny_case()
    with warnings.catch_warnings():
        warnings.simplefilter("error", cl.RepeatedKernelRetrieval)
        forward_and_gradient(case)


@fixed_in_step_1_3
def test_time_loops_do_not_set_every_argument_again(opencl_platform, monkeypatch):
    # Calling a kernel object sets all its arguments, then enqueues it. In the fork only the
    # zeroing kernels are called that way, a few times per run; the time loops enqueue kernels
    # whose arguments were set before.
    import pyopencl as cl
    case = tiny_case(ns=2)
    counts = collections.Counter()
    original_call, original_enqueue = cl.Kernel.__call__, cl.enqueue_nd_range_kernel

    def counting_call(kernel, *args, **kwargs):
        counts[kernel.function_name] += 1
        return original_call(kernel, *args, **kwargs)

    def counting_enqueue(queue, kernel, *args, **kwargs):
        counts["enqueued"] += 1
        return original_enqueue(queue, kernel, *args, **kwargs)

    monkeypatch.setattr(cl.Kernel, "__call__", counting_call)
    monkeypatch.setattr(cl, "enqueue_nd_range_kernel", counting_enqueue)
    forward_and_gradient(case)  # two forward modellings and one gradient
    enqueued = counts.pop("enqueued", 0)
    ns = int(case.W.ns)
    # MakeAllZero: once per source in each forward modelling, twice per source in the gradient.
    assert dict(counts) == {"MakeAllZero": 2 * ns + 2 * ns, "MakeGradZero": 1}
    assert enqueued > 4 * int(case.W.nt)  # every time step went through enqueue_nd_range_kernel


def test_a_reused_propagator_gives_the_results_of_a_fresh_one(opencl_platform):
    # Each forward modelling creates new model and PML buffers, and the update kernels run with
    # coefficient +1 in the forward modelling and -1 in the gradient: the arguments set once per
    # run must follow both. Inside zero guard bands the runs are deterministic (race-free geometry).
    with zero_guard_bands():
        case = tiny_case()
        first = forward_and_gradient(case)
        second = forward_and_gradient(case)  # same propagator, after a gradient
        fresh = forward_and_gradient(tiny_case())
    for name, run in (("second run", second), ("fresh propagator", fresh)):
        for expected, got in zip(first, run):
            for key in expected:
                np.testing.assert_array_equal(got[key], expected[key], err_msg=f"{name}: {key}")
