"""Margins around every device buffer (roadmap step 1.7).

The kernels of PyFWI 0.1.10 read and write outside their buffers (``test_known_bugs.py``): the
stencils a few grid rows before and after a field, the unbounded loop of the right well many rows
past its end, and the receiver loops past the end of seismograms and residuals. In 0.1.10 those
accesses hit whatever lies next to the buffer, so results depend on the memory layout and on the
device (POCL returns NaN). Since step 1.7 every buffer is a sub-buffer of an allocation
``[margin | data | margin]`` whose margins hold ``wave_propagation.MARGIN_FILL`` (0): the same
accesses now read zeros inside the allocation, as the guard bands of the golden masters did, so
plain runs reproduce the golden masters bit for bit (``test_characterization.py``).

These tests check the layout of the allocations, that the kernels really see the margins on the
device, and, with bands placed beyond the margins, that no access goes beyond them, with the
smallest margins the formula of ``WavePreparation`` allows.
"""
import contextlib

import numpy as np
import pytest

from helpers import forward_and_gradient, running_reference, tiny_case
from oob_guard import guard_bands

#: True in the reference environment (published 0.1.10, which has no margins).
REFERENCE = running_reference()
fork_only = pytest.mark.skipif(REFERENCE, reason="the published 0.1.10 has no margins")

#: Every device buffer of a propagator after a forward modelling and a gradient.
BUFFERS = {"vx_b", "vz_b", "taux_b", "tauz_b", "tauxz_b",
           "avx_b", "avz_b", "ataux_b", "atauz_b", "atauxz_b",
           "res_vx_b", "res_vz_b", "res_taux_b", "res_tauz_b", "res_tauxz_b",
           "seismogramid_vx_b", "seismogramid_vz_b", "seismogramid_taux_b", "seismogramid_tauz_b",
           "seismogramid_tauxz_b", "rho_b", "mu_b", "lam_b", "vdx_pml_b", "vdz_pml_b",
           "Gmu_b", "Glam_b", "Grho_b", "g_mu_precond_b", "g_lam_precond_b", "g_rho_precond_b"}


@fork_only
def test_every_buffer_is_an_aligned_sub_buffer_with_margins(opencl_platform):
    import pyopencl as cl

    import PyFWI.wave_propagation as wave
    case = tiny_case(acq_type=2)
    forward_and_gradient(case)
    W = case.W
    align = max(d.mem_base_addr_align for d in W.ctx.devices) // 8
    assert W.margin_bytes % align == 0 and W.margin_bytes >= wave.MIN_MARGIN_BYTES
    assert W.margin_bytes >= (int(W.rec_top_right_var) + 8) * int(W.tnx) * 4  # the right-well loop
    assert set(W._allocations) == BUFFERS
    for name, allocation in W._allocations.items():
        buffer = getattr(W, name)
        assert buffer.get_info(cl.mem_info.OFFSET) == W.margin_bytes, name
        assert allocation.size == buffer.size + 2 * W.margin_bytes, name
    assert W.vx_b.size == W.tnz * W.tnx * 4 and W.seismogramid_vx_b.size == W.nt_ext * W.nr * 4


@fork_only
def test_the_kernels_see_the_margins_on_the_device(opencl_platform, monkeypatch):
    # The kernels use the sub-buffers only. Created with COPY_HOST_PTR, an allocation is placed on
    # the NVIDIA device only for that region, and the margins hold whatever the device memory held
    # (zeros, by luck, in a fresh process); PyFWI writes it whole explicitly. NaN margins must show.
    import PyFWI.wave_propagation as wave
    monkeypatch.setattr(wave, "MARGIN_FILL", np.nan)
    _, grad = forward_and_gradient(tiny_case())
    assert not all(np.isfinite(v).all() for v in grad.values())


@contextlib.contextmanager
def _bands_beyond_the_margins(monkeypatch, fill, band=1 << 16):
    """Put bands of ``fill`` beyond the margins of every buffer; yield a function listing those written.

    The fork: ``WavePreparation._set_buffer`` is replaced, for the test, by an allocator of
    ``[band | margin | data | margin | band]`` (margins as PyFWI fills them, bands of ``fill``).
    The guard bands of ``test/oob_guard.py`` cannot do it, because PyFWI allocates its buffers
    without a host array, which the guard does not wrap. 0.1.10 (reference environment): the guard
    bands, around its buffers, which have no margins.
    """
    if REFERENCE:
        with guard_bands(fill=fill) as report:
            yield report.touched
        return

    import pyopencl as cl

    import PyFWI.wave_propagation as wave
    allocations = []

    def set_buffer(self, name, host):
        host = np.ascontiguousarray(host)
        start = band + self.margin_bytes  # offset of the data; both terms are aligned
        whole = np.full((2 * start + host.nbytes) // 4, fill, np.float32)
        whole[band // 4:(start + host.nbytes + self.margin_bytes) // 4] = wave.MARGIN_FILL
        whole.view(np.uint8)[start:start + host.nbytes] = host.reshape(-1).view(np.uint8)
        allocation = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, whole.nbytes)
        cl.enqueue_copy(self.queue, allocation, whole)
        allocations.append((name, self.queue, allocation, start + host.nbytes + self.margin_bytes))
        setattr(self, name, allocation.get_sub_region(start, host.nbytes))

    def written():
        names = []
        for name, queue, allocation, end in allocations:
            whole = np.empty(allocation.size // 4, np.float32)
            cl.enqueue_copy(queue, whole, allocation)
            bands = np.concatenate([whole[:band // 4], whole[end // 4:]])
            if not (np.isnan(bands).all() if np.isnan(fill) else (bands == np.float32(fill)).all()):
                names.append(name)
        return names

    monkeypatch.setattr(wave.WavePreparation, "_set_buffer", set_buffer)
    yield written


@pytest.mark.xfail(condition=REFERENCE, strict=True, raises=AssertionError,
                   reason="0.1.10 reads and writes next to its buffers")
@pytest.mark.parametrize("sdo", [4, 8])
@pytest.mark.parametrize("acq_type", [0, 1, 2])
def test_every_out_of_bounds_access_stays_inside_the_margins(opencl_platform, monkeypatch, acq_type, sdo):
    # Bands beyond the margins: NaN shows a read beyond them, 1.0 a write. No minimum margin, so
    # that only the formula of WavePreparation sizes the margins.
    import PyFWI.wave_propagation as wave
    monkeypatch.setattr(wave, "MIN_MARGIN_BYTES", 0, raising=False)
    with _bands_beyond_the_margins(monkeypatch, np.nan):
        d_obs, grad = forward_and_gradient(tiny_case(acq_type=acq_type, sdo=sdo, components=4))
    assert all(np.isfinite(v).all() for v in d_obs.values()), "read beyond the margins (seismogram)"
    assert all(np.isfinite(v).all() for v in grad.values()), "read beyond the margins (gradient)"
    with _bands_beyond_the_margins(monkeypatch, 1.0) as written:
        forward_and_gradient(tiny_case(acq_type=acq_type, sdo=sdo, components=4))
        assert not written(), "written beyond the margins"
