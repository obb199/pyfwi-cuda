"""Gradient through PyTorch with ``PyFWI.torchfwi.Fwi``.

What this example shows
-----------------------
* ``torchfwi.Fwi`` is a ``torch.autograd.Function``: ``Fwi.apply(W, vp, vs, rho)`` runs
  PyFWI's forward modelling with the propagator ``W`` on the model given as tensors and
  returns the pressure seismograms ``(taux, tauz)`` as tensors. Its backward pass calls
  ``W.gradient`` with the derivative of the loss as adjoint source, so any PyTorch loss
  and optimiser can drive an elastic inversion.
* The model tensors of the starting model are created with ``requires_grad=True``;
  after ``loss.backward()`` their ``.grad`` holds the gradient (adjoint-state method).
* ``W`` needs checkpoints (``chpr``) for the gradient, exactly as in ``gradient.py``.
* For an L2 loss, the PyTorch gradient equals the one of PyFWI's own path
  (``fwi_tools.cost_seismic`` + ``WavePropagator.gradient``): the example computes both.

How to run it
-------------
::

    python example/scripts/torch_gradient.py                       # full size, with a figure
    python example/scripts/torch_gradient.py --size small --no-show

The test suite (``test/test_examples.py``) calls ``main(size="small")`` and checks that
both gradients agree and point to the inclusion. It needs PyTorch (CPU is enough).
"""
import argparse

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import PyFWI.acquisition as acq
import PyFWI.fwi_tools as tools
import PyFWI.processing as process
import PyFWI.wave_propagation as wave
from PyFWI.torchfwi import Fwi

#: Problem sizes. nz, nx: model size in grid points; dh: grid spacing (m);
#: ns: number of sources; t: recording length (s).
SIZES = {
    "small": {"nz": 40, "nx": 60, "dh": 10.0, "ns": 2, "t": 0.4},
    "full": {"nz": 100, "nx": 100, "dh": 7.0, "ns": 5, "t": 0.8},
}


def make_models(nz, nx):
    """True model (faster circular inclusion) and smooth starting model, as NumPy arrays.

    Returns ``(true, start)``, each ``{"vp", "vs", "rho"}`` in m/s, m/s, kg/m³,
    ``float32`` arrays ``(nz, nx)``. Only vp differs between the two models.
    """
    z, x = np.mgrid[:nz, :nx]
    inside = (z - nz // 2) ** 2 + (x - nx // 2) ** 2 <= (nz // 10) ** 2
    vp = np.where(inside, 3000.0, 2500.0).astype(np.float32)
    vs = np.full((nz, nx), 2500.0 / np.sqrt(3.0), np.float32)
    rho = np.full((nz, nx), 310.0 * 2500.0 ** 0.25, np.float32)
    true = {"vp": vp, "vs": vs, "rho": rho}
    start = {"vp": gaussian_filter(vp, nz / 4).astype(np.float32), "vs": vs.copy(), "rho": rho.copy()}
    return true, start


def main(size="small", show=False):
    """Compute the gradient of an L2 loss at the starting model with PyTorch, and with PyFWI directly.

    Parameters
    ----------
    size : {"small", "full"}
        Problem size, see ``SIZES``.
    show : bool
        If True, plot minus the vp gradient from both paths.

    Returns
    -------
    dict
        * ``"loss"``: value of the L2 loss at the starting model;
        * ``"torch_gradient"``: ``{"vp", "vs", "rho"}`` NumPy arrays from ``loss.backward()``;
        * ``"pyfwi_gradient"``: the same from ``cost_seismic`` + ``W.gradient``;
        * ``"true"``, ``"start"``: the models; ``"inpa"``.

    Used by
    -------
    The command line (below) and ``test/test_examples.py::test_torch_gradient``.
    """
    p = SIZES[size]
    nz, nx, dh = p["nz"], p["nx"], p["dh"]
    true, start = make_models(nz, nx)
    inpa = {"ns": p["ns"], "sdo": 4, "fdom": 15, "dh": dh, "dt": 0.004, "acq_type": 1, "t": p["t"],
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": dh}  # see forward_surface.py for each key
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(inpa["ns"], dh, dh * nx, dh * nz, dh, inpa["sdo"],
                                                         acq_type=1)
    src_loc[:, 1] -= dh  # one row above the receivers (see forward_surface.py)
    src = acq.Source(src_loc, dh, inpa["dt"])
    src.Ricker(inpa["fdom"])

    # Observed data, through the same autograd function (no gradient needed for them).
    W_obs = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), components=0, n_well_rec=n_well_rec, chpr=0)
    with torch.no_grad():
        taux_obs, tauz_obs = Fwi.apply(W_obs, *(torch.tensor(true[k]) for k in ("vp", "vs", "rho")))

    # Starting model as tensors that track gradients; a propagator with checkpoints.
    params = {k: torch.tensor(start[k], requires_grad=True) for k in ("vp", "vs", "rho")}
    W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), components=0, n_well_rec=n_well_rec, chpr=20)
    taux, tauz = Fwi.apply(W, params["vp"], params["vs"], params["rho"])
    loss = 0.5 * torch.sum((taux - taux_obs) ** 2) + 0.5 * torch.sum((tauz - tauz_obs) ** 2)
    loss.backward()  # runs W.gradient with the derivative of the loss as adjoint source
    torch_gradient = {k: v.grad.numpy() for k, v in params.items()}

    # The same gradient with PyFWI's own tools (the forward run of W above is reused).
    d_est = process.prepare_residual({"taux": taux.detach().numpy(), "tauz": tauz.detach().numpy()}, 1)
    d_obs = process.prepare_residual({"taux": taux_obs.numpy(), "tauz": tauz_obs.numpy()}, 1)
    _, adj_src = tools.cost_seismic(d_est, d_obs, fun=tools.CostFunction("l2"))
    pyfwi_gradient = W.gradient(adj_src, show=False)

    if show:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, (title, g) in zip(ax, (("PyTorch: minus the vp gradient", torch_gradient),
                                      ("PyFWI: minus the vp gradient", pyfwi_gradient))):
            clip = np.abs(g["vp"]).max()
            a.imshow(-g["vp"], cmap="seismic", vmin=-clip, vmax=clip, extent=[0, dh * nx, dh * nz, 0])
            a.set(title=title, xlabel="x (m)", ylabel="z (m)")
        plt.tight_layout()
        plt.show()

    return {"loss": float(loss), "torch_gradient": torch_gradient, "pyfwi_gradient": pyfwi_gradient,
            "true": true, "start": start, "inpa": inpa}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient through PyTorch with PyFWI.torchfwi.Fwi.")
    parser.add_argument("--size", choices=sorted(SIZES), default="full", help="problem size")
    parser.add_argument("--no-show", action="store_true", help="do not plot")
    args = parser.parse_args()
    main(size=args.size, show=not args.no_show)
