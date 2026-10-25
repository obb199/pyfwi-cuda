"""One OpenCL context per device for the whole process (roadmap step 1.7).

PyFWI 0.1.10 created a new OpenCL context for every ``WavePropagator``. On the NVIDIA driver,
releasing a context while another one exists on the same device breaks the following ones: a loop
that builds a new propagator while the previous one is still referenced fails at the fourth
propagator with ``CL_OUT_OF_RESOURCES``. Phase 0 blamed the out-of-bounds accesses of the kernels,
because runs inside guard bands never failed; step 1.7 found that the guard bands avoided it only
by keeping the contexts alive (their command queues are kept), and that padding the buffers does
not change it. The fork therefore reuses one context per set of devices
(``PyFWI._opencl_device.shared_context``).
"""
import os
import subprocess
import sys

import pytest

from helpers import running_reference, tiny_case

#: True in the reference environment (published 0.1.10).
REFERENCE = running_reference()

#: The loop that crashes 0.1.10 on NVIDIA, run in a fresh process (a crash may leave the driver of
#: the test process in a bad state). argv: <test folder> <OpenCL platform index>.
LOOP = """
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = [sys.argv[1], os.path.dirname(sys.argv[1])]  # the test helpers and the repository root
os.environ["PYOPENCL_CTX"] = sys.argv[2]
from helpers import tiny_case, forward_and_gradient
case = None
for i in range(6):
    case = tiny_case(acq_type=1, ns=2, chpr=20, nz=50, nx=50, t=0.4)  # the previous one is released here
    forward_and_gradient(case)
    print(i, "ok", flush=True)
"""


@pytest.mark.skipif(REFERENCE, reason="the published 0.1.10 creates a context per propagator")
def test_propagators_on_the_same_device_share_one_context(opencl_platform):
    first, second = tiny_case().W, tiny_case().W
    assert first.ctx is second.ctx
    assert first.queue is not second.queue  # each propagator keeps its own queue


@pytest.mark.xfail(condition=REFERENCE, strict=True,
                   reason="0.1.10 creates and releases a context per propagator: the 4th fails on NVIDIA")
def test_rebuilding_propagators_in_a_loop_does_not_crash(nvidia_opencl_platform):
    run = subprocess.run([sys.executable, "-c", LOOP, os.path.dirname(__file__), str(nvidia_opencl_platform)],
                         capture_output=True, text=True, timeout=600, check=False)
    assert run.returncode == 0 and "5 ok" in run.stdout, run.stdout[-500:] + run.stderr[-2000:]
