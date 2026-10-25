"""The three acquisition geometries of PyFWI: crosswell, surface, and both.

What this example shows
-----------------------
* ``acq_type`` 0 (crosswell): sources in a well on the left, receivers in a well on
  the right. 1 (surface): sources and receivers along a line near the top. 2 (both):
  a left well, the surface line and a right well, all recorded together.
* How ``PyFWI.acquisition.acq_parameters`` lays the sources and receivers out, and
  what ``n_well_rec`` (receivers per well) means.
* How the columns of the returned seismograms map to receivers. With ``acq_type=2``
  PyFWI returns the traces of the **left well in reverse order** (bottom to top), so
  that the traces run left well → surface → right well as one continuous line.
  ``trace_positions()`` gives the receiver position of every column.
* How to keep the geometry free of the race conditions of PyFWI 0.1.10: no source on
  a receiver grid point, and no two receivers on the same grid point.

How to run it
-------------
::

    python example/scripts/forward_acquisitions.py                       # full size, with figures
    python example/scripts/forward_acquisitions.py --size small --no-show

The test suite (``test/test_examples.py``) calls ``main(size="small")`` and checks,
for each geometry, that the first arrivals get later with the source-receiver
distance when the traces are matched with ``trace_positions()``.
"""
import argparse

import numpy as np

import PyFWI.acquisition as acq
import PyFWI.wave_propagation as wave

#: Problem sizes. nz, nx: model size in grid points; dh: grid spacing (m);
#: ns: number of sources; t: recording length (s).
SIZES = {
    "small": {"nz": 40, "nx": 40, "dh": 10.0, "ns": 2, "t": 0.4},
    "full": {"nz": 100, "nx": 100, "dh": 7.0, "ns": 4, "t": 0.6},
}

#: Names of the acquisition types, indexed by PyFWI's ``acq_type``.
ACQ_NAMES = {0: "crosswell", 1: "surface", 2: "both"}


def make_model(nz, nx):
    """Elastic model with a circular high-velocity inclusion (see ``forward_surface.py``).

    Returns ``{"vp", "vs", "rho"}``: m/s, m/s, kg/m³, ``float32`` arrays of shape ``(nz, nx)``.
    """
    z, x = np.mgrid[:nz, :nx]
    inside = (z - nz // 2) ** 2 + (x - nx // 2) ** 2 <= (nz // 10) ** 2
    vp = np.where(inside, 3000.0, 2500.0)
    model = {"vp": vp, "vs": vp / np.sqrt(3.0), "rho": 310.0 * vp ** 0.25}
    return {name: values.astype(np.float32) for name, values in model.items()}


def make_geometry(acq_type, ns, dh, nz, nx, sdo):
    """Sources and receivers for one acquisition type, free of race conditions.

    Starts from ``acq_parameters`` and adjusts it:

    * surface and both: the sources go one grid row above the surface receiver line
      (by default they are on it);
    * both: the sources go between the two wells, and the surface line keeps only
      the receivers strictly between the well columns. The kernels place the surface
      receivers from the first one every ``rec_dis``, so the line must stay contiguous,
      and it must not cross the wells: two receivers on one grid point make the
      gradient depend on GPU thread order in 0.1.10.

    Parameters
    ----------
    acq_type : int
        0 crosswell, 1 surface, 2 both.
    ns : int
        Number of sources.
    dh : float
        Grid spacing in metres (also used as receiver spacing).
    nz, nx : int
        Model size in grid points.
    sdo : int
        Spatial order of the finite differences (it sets how far from the edges
        ``acq_parameters`` puts sources and receivers).

    Returns
    -------
    src_loc, rec_loc : numpy.ndarray
        Positions in metres, shape ``(n, 2)``, columns ``(x, z)``. For ``acq_type=2``
        the receivers are ordered left well (top to bottom), surface line (left to
        right), right well (top to bottom), as PyFWI expects.
    n_well_rec : int
        Receivers per well (0 for surface).
    """
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(ns, dh, dh * nx, dh * nz, dh, sdo, acq_type=acq_type)
    src_loc, rec_loc = np.array(src_loc, np.float32), np.array(rec_loc, np.float32)
    if acq_type == 1:
        src_loc[:, 1] -= dh
    elif acq_type == 2:
        left, surface, right = rec_loc[:n_well_rec], rec_loc[n_well_rec:-n_well_rec], rec_loc[-n_well_rec:]
        x_left, x_right = left[0, 0], right[0, 0]
        surface = surface[(surface[:, 0] > x_left) & (surface[:, 0] < x_right)]
        rec_loc = np.vstack([left, surface, right])
        src_loc[:, 0] = np.linspace(x_left + 2 * dh, x_right - 2 * dh, ns)
        src_loc[:, 1] = surface[0, 1] - dh
    return src_loc, rec_loc, n_well_rec


def trace_positions(rec_loc, n_well_rec, acq_type):
    """Receiver position of every trace (column) of one shot in the seismograms PyFWI returns.

    For ``acq_type`` 0 and 1 the traces follow ``rec_loc``. For ``acq_type=2`` the
    left-well block comes in reverse order (``forward_propagator`` flips it).

    Parameters
    ----------
    rec_loc : numpy.ndarray
        Receiver positions in metres, shape ``(nr, 2)``.
    n_well_rec : int
        Receivers per well.
    acq_type : int
        0 crosswell, 1 surface, 2 both.

    Returns
    -------
    numpy.ndarray
        Positions ``(x, z)`` in metres, shape ``(nr, 2)``, in trace order.
    """
    positions = np.array(rec_loc, np.float32)
    if acq_type == 2:
        positions[:n_well_rec] = positions[:n_well_rec][::-1]
    return positions


def model_acquisition(acq_type, size="small"):
    """Run the forward modelling for one acquisition type.

    Parameters
    ----------
    acq_type : int
        0 crosswell, 1 surface, 2 both.
    size : {"small", "full"}
        Problem size, see ``SIZES``.

    Returns
    -------
    dict
        ``"data"`` (seismograms, one ``(nt, nr * ns)`` array per component; with
        ``components=0`` PyFWI returns the pressure as ``"taux"`` and ``"tauz"``),
        ``"src_loc"``, ``"rec_loc"``, ``"n_well_rec"``, ``"traces"`` (receiver
        position of every column of one shot) and ``"inpa"``.
    """
    p = SIZES[size]
    nz, nx, dh = p["nz"], p["nx"], p["dh"]
    inpa = {"ns": p["ns"], "sdo": 4, "fdom": 15, "dh": dh, "dt": 0.004, "acq_type": acq_type, "t": p["t"],
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": dh}  # see forward_surface.py for each key
    src_loc, rec_loc, n_well_rec = make_geometry(acq_type, p["ns"], dh, nz, nx, inpa["sdo"])
    src = acq.Source(src_loc, dh, inpa["dt"])
    src.Ricker(inpa["fdom"])
    # n_well_rec must be given for crosswell and both: it tells PyFWI how rec_loc is split.
    W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), components=0, n_well_rec=n_well_rec, chpr=0)
    data = W.forward_modeling(make_model(nz, nx), show=False)
    return {"data": data, "src_loc": src_loc, "rec_loc": rec_loc, "n_well_rec": n_well_rec,
            "traces": trace_positions(rec_loc, n_well_rec, acq_type), "inpa": inpa}


def main(size="small", show=False):
    """Model the three acquisition types and optionally plot the first shot of each.

    Parameters
    ----------
    size : {"small", "full"}
        Problem size, see ``SIZES``.
    show : bool
        If True, plot the geometry and the pressure gather of the first shot for each type.

    Returns
    -------
    dict
        ``{"crosswell": ..., "surface": ..., "both": ...}``, each as returned by
        ``model_acquisition()``.

    Used by
    -------
    The command line (below) and ``test/test_examples.py::test_forward_acquisitions``.
    """
    results = {ACQ_NAMES[a]: model_acquisition(a, size) for a in (0, 1, 2)}
    if show:
        import matplotlib.pyplot as plt
        p = SIZES[size]
        _, axes = plt.subplots(2, 3, figsize=(12, 7))
        for col, (name, r) in enumerate(results.items()):
            nr = r["rec_loc"].shape[0]
            ax = axes[0, col]
            ax.plot(r["rec_loc"][:, 0], r["rec_loc"][:, 1], "kv", markersize=2, label="receivers")
            ax.plot(r["src_loc"][:, 0], r["src_loc"][:, 1], "r*", label="sources")
            ax.set(xlim=(0, p["dh"] * p["nx"]), ylim=(p["dh"] * p["nz"], 0), title=name, xlabel="x (m)", ylabel="z (m)")
            ax.legend(loc="lower right", fontsize=7)
            gather = r["data"]["taux"][:, :nr]
            clip = np.percentile(np.abs(gather), 99)
            axes[1, col].imshow(gather, cmap="gray", aspect="auto", vmin=-clip, vmax=clip,
                                extent=[0, nr, p["t"], 0])
            axes[1, col].set(title=f"{name}: shot 1, pressure", xlabel="trace", ylabel="time (s)")
        plt.tight_layout()
        plt.show()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The three acquisition geometries of PyFWI.")
    parser.add_argument("--size", choices=sorted(SIZES), default="full", help="problem size")
    parser.add_argument("--no-show", action="store_true", help="do not plot")
    args = parser.parse_args()
    main(size=args.size, show=not args.no_show)
