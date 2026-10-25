"""The synthetic model library of PyFWI: ``model_dataset.ModelGenerator``.

What this example shows
-----------------------
* ``ModelGenerator(name)`` creates a generator; calling it returns the model as a
  dictionary of ``(nz, nx)`` arrays. The keyword arguments are the same for every model:
  ``vintage=1`` (baseline) or ``vintage=2`` (monitor, for time-lapse studies), and
  ``smoothing``. Mind what ``smoothing`` means: for the models shown here a non-zero
  value returns the **homogeneous background without the anomalies** (the starting
  model used in the PyFWI notebooks), not a filtered version; only ``"marmousi"`` and
  ``"dupuy"`` apply a smoothing filter (``model_dataset.model_smoother``).
* The models available without downloads, 100 x 100 grid points each:

  - ``"louboutin"`` and ``"yang"``: elastic models (``vp``, ``vs`` in m/s, ``rho``) with
    circular anomalies; their monitors change vp in a small zone;
  - ``"perturbation_dv"``: elastic anomalies in vp, vs and rho at different places
    (baseline only: ``vintage=2`` raises an exception);
  - ``"hu_circles"`` and ``"hu_laminar"``: rock-physics models with porosity ``phi``,
    clay content ``cc`` and water saturation ``sw`` (fractions between 0 and 1); their
    monitors change the saturation.

* Units are not uniform: ``rho`` of ``"louboutin"`` is in g/cm³ (Gardner's relation),
  ``rho`` of ``"yang"`` in kg/m³. Check before mixing models.
* Not covered: ``"marmousi"`` downloads the Marmousi-2 model, and ``"dupuy"`` reads a data
  file that the PyPI wheel of PyFWI 0.1.10 does not include (fixed in step 1.1).

How to run it
-------------
::

    python example/scripts/models.py                       # with a figure
    python example/scripts/models.py --size small --no-show

The models have a fixed size, so ``size`` does not change them. The test suite
(``test/test_examples.py``) checks their keys, shapes, monitors and smoothing.

Requirements: ``PyFWI.model_dataset`` does not import with NumPy 2 in PyFWI 0.1.10
(fixed in step 1.1 of the fork's roadmap).
"""
import argparse

import PyFWI.model_dataset as md

#: Accepted for uniformity with the other examples; the library models have a fixed size.
SIZES = {"small": {}, "full": {}}

#: Models of the library shown here, with the property plotted for each.
MODELS = {"louboutin": "vp", "yang": "vp", "perturbation_dv": "vp", "hu_circles": "sw", "hu_laminar": "sw"}


def generate(name):
    """Generate the baseline, monitor and smoothed versions of one library model.

    Parameters
    ----------
    name : str
        A key of ``MODELS``.

    Returns
    -------
    dict
        ``{"baseline": model, "monitor": model or None, "smooth": model}`` (``"smooth"`` is
        the result of ``smoothing=1``: the background), each model a
        ``dict[str, numpy.ndarray]`` of ``float32`` ``(nz, nx)`` arrays. ``"monitor"`` is
        None for models that have no monitor version.
    """
    generator = md.ModelGenerator(name)
    baseline = generator()                      # vintage=1, smoothing=0
    try:
        monitor = generator(vintage=2)
    except Exception:  # noqa: BLE001 - PyFWI raises a bare Exception for models without a monitor
        monitor = None
    smooth = generator(smoothing=1)
    return {"baseline": baseline, "monitor": monitor, "smooth": smooth}


def main(size="small", show=False):
    """Generate every model of ``MODELS`` and optionally plot them.

    Parameters
    ----------
    size : {"small", "full"}
        Ignored (the library models have a fixed size); kept so that every example has
        the same interface.
    show : bool
        If True, plot the baseline, the monitor change and the ``smoothing=1`` version of each model.

    Returns
    -------
    dict
        ``{name: generate(name)}`` for every name of ``MODELS``.

    Used by
    -------
    The command line (below) and ``test/test_examples.py::test_models``.
    """
    results = {name: generate(name) for name in MODELS}
    if show:
        import matplotlib.pyplot as plt
        _, axes = plt.subplots(len(MODELS), 3, figsize=(9, 3 * len(MODELS)))
        for row, (name, versions) in zip(axes, results.items()):
            prop = MODELS[name]
            row[0].imshow(versions["baseline"][prop], cmap="jet")
            row[0].set_title(f"{name}: {prop}")
            if versions["monitor"] is not None:
                change = versions["monitor"][prop] - versions["baseline"][prop]
                clip = max(abs(change).max(), 1e-12)  # symmetric scale: white means unchanged
                row[1].imshow(change, cmap="seismic", vmin=-clip, vmax=clip)
                row[1].set_title("monitor - baseline")
            else:
                row[1].set_title("(no monitor)")
            row[2].imshow(versions["smooth"][prop], cmap="jet",
                          vmin=versions["baseline"][prop].min(), vmax=versions["baseline"][prop].max())
            row[2].set_title("smoothing=1 (background)")
            for a in row:
                a.axis("off")
        plt.tight_layout()
        plt.show()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The synthetic model library of PyFWI.")
    parser.add_argument("--size", choices=sorted(SIZES), default="full", help="ignored (fixed-size models)")
    parser.add_argument("--no-show", action="store_true", help="do not plot")
    args = parser.parse_args()
    main(size=args.size, show=not args.no_show)
