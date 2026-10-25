"""Forward modelling with a surface acquisition.

What this example shows
-----------------------
* How to build an elastic model (P- and S-wave velocities and density) as NumPy arrays.
* How to describe a surface acquisition: sources and a line of receivers near the top.
* The ``inpa`` dictionary of PyFWI input parameters, key by key.
* How to run ``WavePropagator.forward_modeling`` and read the seismograms it returns.

How to run it
-------------
::

    python example/scripts/forward_surface.py                       # full size, with a figure
    python example/scripts/forward_surface.py --size small --no-show

The test suite (``test/test_examples.py``) calls ``main(size="small")`` and checks
that the first arrivals come later at receivers farther from the source.
"""
import argparse

import numpy as np

import PyFWI.acquisition as acq
import PyFWI.wave_propagation as wave

#: Problem sizes. "small" runs in about a second and is used by the tests; "full" is
#: the setup of the PyFWI documentation (100 x 100 model, 7 m grid, 4 shots, 0.6 s).
#: nz, nx: model size in grid points; dh: grid spacing (m); ns: number of sources;
#: t: recording length (s).
SIZES = {
    "small": {"nz": 40, "nx": 60, "dh": 10.0, "ns": 2, "t": 0.3},
    "full": {"nz": 100, "nx": 100, "dh": 7.0, "ns": 4, "t": 0.6},
}


def make_model(nz, nx):
    """Build an elastic model with a circular high-velocity inclusion.

    Parameters
    ----------
    nz, nx : int
        Number of grid points in depth (z, rows) and in x (columns).

    Returns
    -------
    dict[str, numpy.ndarray]
        ``"vp"`` and ``"vs"`` in m/s and ``"rho"`` in kg/m³, each ``float32`` with
        shape ``(nz, nx)``. vp is 2500 m/s in the background and 3000 m/s in a
        circle of radius ``nz // 10`` at the centre; vs = vp / √3 (Poisson's ratio
        0.25) and rho follows Gardner's relation 310 · vp^0.25. PyFWI expects
        exactly these three keys.
    """
    z, x = np.mgrid[:nz, :nx]  # row and column index of every grid point
    inside = (z - nz // 2) ** 2 + (x - nx // 2) ** 2 <= (nz // 10) ** 2
    vp = np.where(inside, 3000.0, 2500.0)
    model = {"vp": vp, "vs": vp / np.sqrt(3.0), "rho": 310.0 * vp ** 0.25}
    return {name: values.astype(np.float32) for name, values in model.items()}


def main(size="small", show=False):
    """Model the seismograms of a surface survey over the inclusion model.

    Parameters
    ----------
    size : {"small", "full"}
        Problem size, see ``SIZES``.
    show : bool
        If True, plot the model and the shot gather of the first source.

    Returns
    -------
    dict
        * ``"data"``: the seismograms, one array per component (``"vx"``, ``"vz"``,
          ``"taux"``, ``"tauz"``, ``"tauxz"``), each ``(nt, nr * ns)``: one row per
          time sample, and the traces of shot 0, then shot 1, ...;
        * ``"src_loc"``, ``"rec_loc"``: source and receiver positions in metres,
          ``(n, 2)`` arrays of ``(x, z)``;
        * ``"inpa"``: the input parameters; ``"model"``: the model.

    Used by
    -------
    The command line (below) and ``test/test_examples.py::test_forward_surface``.
    """
    p = SIZES[size]
    nz, nx, dh = p["nz"], p["nx"], p["dh"]
    model = make_model(nz, nx)

    # inpa: the dictionary of input parameters that every PyFWI class receives.
    inpa = {
        "ns": p["ns"],      # number of sources (shots)
        "sdo": 4,           # spatial order of the finite differences: 4 or 8
        "fdom": 15,         # dominant frequency of the Ricker wavelet, Hz
        "dh": dh,           # grid spacing, m (the same in x and z)
        "dt": 0.004,        # sampling interval of the recorded data, s (PyFWI splits it into
                            # internal steps of at most 0.6 ms for stability)
        "acq_type": 1,      # acquisition: 0 crosswell, 1 surface, 2 both
        "t": p["t"],        # recording length, s
        "npml": 20,         # thickness of the absorbing boundary (PML), grid points
        "pmlR": 1e-5,       # reflection coefficient the PML aims at
        "pml_dir": 2,       # absorbing boundary on every side
        "rec_dis": dh,      # receiver spacing, m (one receiver per grid point)
    }

    # Sources spread along x and one receiver every rec_dis, all at the same depth.
    # acq_parameters also returns the number of surface receivers (not needed here) and
    # the number of receivers per well (0 for a surface survey).
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(
        inpa["ns"], inpa["rec_dis"], dh * nx, dh * nz, dh, inpa["sdo"], acq_type=inpa["acq_type"])
    # Move the sources one grid row up. In PyFWI 0.1.10 a source placed exactly on a
    # receiver cell makes what that receiver records depend on the order in which the
    # GPU threads run (the injection and the recording happen in the same kernel).
    src_loc[:, 1] -= dh

    # The source wavelet: a Ricker wavelet sampled at dt.
    src = acq.Source(src_loc, dh, inpa["dt"])
    src.Ricker(inpa["fdom"])

    # components=4 records all five fields. chpr is the percentage of time steps kept
    # as checkpoints for a later gradient; a pure forward run does not need them.
    W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), components=4, n_well_rec=n_well_rec, chpr=0)
    data = W.forward_modeling(model, show=False)

    if show:
        import matplotlib.pyplot as plt
        nr = rec_loc.shape[0]
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(model["vp"], cmap="coolwarm", extent=[0, dh * nx, dh * nz, 0])
        ax[0].plot(src_loc[:, 0], src_loc[:, 1], "r*", label="sources")
        ax[0].plot(rec_loc[:, 0], rec_loc[:, 1], "kv", markersize=2, label="receivers")
        ax[0].set(title="vp (m/s)", xlabel="x (m)", ylabel="z (m)")
        ax[0].legend(loc="lower right")
        gather = data["taux"][:, :nr]  # traces of the first shot
        clip = np.percentile(np.abs(gather), 99)  # clip the strong direct wave to see the reflections
        ax[1].imshow(gather, cmap="gray", aspect="auto", vmin=-clip, vmax=clip,
                     extent=[rec_loc[0, 0], rec_loc[-1, 0], inpa["t"], 0])
        ax[1].set(title="Shot 1, taux", xlabel="receiver x (m)", ylabel="time (s)")
        plt.tight_layout()
        plt.show()

    return {"data": data, "src_loc": src_loc, "rec_loc": rec_loc, "inpa": inpa, "model": model}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward modelling with a surface acquisition.")
    parser.add_argument("--size", choices=sorted(SIZES), default="full", help="problem size")
    parser.add_argument("--no-show", action="store_true", help="do not plot")
    args = parser.parse_args()
    main(size=args.size, show=not args.no_show)
