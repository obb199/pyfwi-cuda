"""The case matrix of the golden masters (roadmap step 0.4).

A golden master is a set of outputs of PyFWI 0.1.10, stored once and compared with
the fork at every later step. The cases must exercise every option that changes
what the GPU kernels compute, without running the full Cartesian product (which
would be 3 x 5 x 2 x 2 x 3 x 2 = 360 runs). The matrix below is **pairwise**:
every pair of values of any two factors appears together in at least one case,
which catches most interaction bugs at a fraction of the cost (about 15 cases).

Factors of the propagation cases
--------------------------------
* ``acq_type`` -- 0 crosswell, 1 surface, 2 both (different kernels record and inject);
* ``components`` -- 0 pressure, 1 taux, 2 vx/vz, 3 stresses, 4 all (what is recorded,
  and therefore which residuals drive the gradient);
* ``order`` -- spatial order of the finite differences, 4 or 8 (different stencil
  coefficients and stencil reach);
* ``energy_balancing`` -- whether the gradient is preconditioned by the forward energy;
* ``chpr`` -- percentage of time steps kept as checkpoints (0 is the default and gives
  a wrong gradient in 0.1.10, which the golden master must still reproduce);
* ``parameterization`` -- gradient with respect to vp/vs/rho (``"dv"``) or lam/mu/rho.

Every case runs the forward modelling of the true model and the gradient at the
starting model, on a 60 x 60 model, with the reproducibility rules of
``test_reproducibility.py`` (zero guard bands and a race-free geometry).

Used by
-------
``test/golden_runner.py`` (runs a case), ``tools/make_golden.py`` (stores the
outputs of 0.1.10) and the characterisation tests (compare the fork with them).
"""
from dataclasses import asdict, dataclass

#: Values of each factor, in a fixed order (the order makes the matrix reproducible).
FACTORS = {
    "acq_type": (0, 1, 2),
    "components": (0, 1, 2, 3, 4),
    "order": (4, 8),
    "energy_balancing": (False, True),
    "chpr": (0, 20, 100),
    "parameterization": ("dv", "lmd"),
}


@dataclass(frozen=True)
class PropagationCase:
    """One golden-master case: a forward modelling and a gradient with fixed options.

    Attributes
    ----------
    acq_type, components, order, energy_balancing, chpr, parameterization
        The factor values (see the module docstring).
    nz, nx : int
        Model size in grid points (without the absorbing boundary).
    dh : float
        Grid spacing in metres; receivers every ``dh``.
    ns : int
        Number of sources.
    t : float
        Recording length in seconds (with ``dt = 4 ms``: 63 samples, 438 internal steps).
    npml : int
        Thickness of the absorbing boundary in grid points.
    """
    acq_type: int
    components: int
    order: int
    energy_balancing: bool
    chpr: int
    parameterization: str
    nz: int = 60
    nx: int = 60
    dh: float = 10.0
    ns: int = 2
    t: float = 0.25
    npml: int = 10

    @property
    def name(self):
        """A short, unique, file-name-friendly identifier, e.g. ``a1_c4_o8_eb1_chpr20_dv``."""
        return (f"a{self.acq_type}_c{self.components}_o{self.order}_eb{int(self.energy_balancing)}"
                f"_chpr{self.chpr}_{self.parameterization}")

    def as_dict(self):
        """All attributes as a plain dict (stored in the golden-master manifest)."""
        return asdict(self)


def pairwise_rows(factors):
    """Build a small set of rows in which every pair of factor values appears at least once.

    Greedy construction, deterministic for a given ``factors`` (dicts keep their order):

    1. start with the full product of the two factors with the most values (every
       pair of those two is then covered);
    2. fill the other factors of each row, one factor at a time in the order of
       ``factors``, with the value that covers the most pairs not yet covered
       (ties: the first value in order);
    3. while some pair is still missing, add a row seeded with it and fill it the same way.

    Parameters
    ----------
    factors : dict[str, tuple]
        Factor name -> possible values.

    Returns
    -------
    list of dict
        One ``{factor: value}`` per row.

    Used by
    -------
    ``propagation_cases()`` and ``test_golden_cases.py``.
    """
    names = list(factors)
    uncovered = {((a, va), (b, vb)) for i, a in enumerate(names) for b in names[i + 1:]
                 for va in factors[a] for vb in factors[b]}

    def pair(a, va, b, vb):
        """The pair in canonical order (the order of ``names``)."""
        return ((a, va), (b, vb)) if names.index(a) < names.index(b) else ((b, vb), (a, va))

    def fill(row):
        for name in names:
            if name in row:
                continue
            gains = [sum(pair(name, v, other, row[other]) in uncovered for other in row) for v in factors[name]]
            row[name] = factors[name][gains.index(max(gains))]
        return row

    def add(row):
        rows.append(row)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                uncovered.discard(pair(a, row[a], b, row[b]))

    rows = []
    first, second = sorted(names, key=lambda n: -len(factors[n]))[:2]
    for v1 in factors[first]:
        for v2 in factors[second]:
            add(fill({first: v1, second: v2}))
    while uncovered:
        (a, va), (b, vb) = min(uncovered, key=repr)  # the first missing pair, in a fixed order
        add(fill({a: va, b: vb}))
    return rows


def propagation_cases():
    """The golden-master propagation cases: one ``PropagationCase`` per pairwise row of ``FACTORS``."""
    return [PropagationCase(**row) for row in pairwise_rows(FACTORS)]


#: FWI optimisers of ``FWI.__call__`` and time-lapse strategies of ``TimeLapse.__call__``.
FWI_METHODS = ("lbfgs", "cg")
TL_METHODS = ("cc", "pa", "dd", "sim", "wa", "cj", "cd", "cu")


@dataclass(frozen=True)
class InversionCase:
    """One inversion golden-master case: a short FWI or time-lapse FWI on a small model.

    The models are those of ``helpers.tiny_case`` (30 x 40 grid, 10 m spacing, a faster
    circular inclusion, smooth starting model); the monitor model of time-lapse cases has
    vp lowered by 10 % in rows 8-11, columns 15-24. The source wavelet is scaled by 1000,
    so that the misfit is of order 1 and the absolute tolerances of the optimisers of
    0.1.10 do not stop them before the first iteration (see ``example/scripts/fwi.py``).

    Attributes
    ----------
    kind : str
        ``"fwi"`` (``PyFWI.fwi.FWI``) or ``"tlfwi"`` (``PyFWI.tl_fwi.TimeLapse``).
    method : str
        For ``"fwi"``, the optimiser (``FWI_METHODS``); for ``"tlfwi"``, the strategy (``TL_METHODS``).
    iterations : int
        Iterations of the single multi-scale stage (25 Hz).
    """
    kind: str
    method: str
    iterations: int = 2
    nz: int = 30
    nx: int = 40
    ns: int = 2

    @property
    def name(self):
        """Identifier such as ``fwi_lbfgs`` or ``tlfwi_cc``."""
        return f"{self.kind}_{self.method}"

    def as_dict(self):
        """All attributes as a plain dict (stored in the golden-master manifest)."""
        return asdict(self)


@dataclass(frozen=True)
class ExampleCase:
    """The numerical outputs of one script of ``example/scripts/``, run with ``main(size="small")``.

    Attributes
    ----------
    script : str
        File name of the example without ``.py``.
    """
    script: str

    @property
    def name(self):
        """Identifier such as ``example_forward_surface``."""
        return f"example_{self.script}"

    def as_dict(self):
        """All attributes as a plain dict (stored in the golden-master manifest)."""
        return asdict(self)


def inversion_cases():
    """Two FWI cases (one per optimiser, 2 iterations) and eight time-lapse cases (1 iteration each)."""
    return ([InversionCase("fwi", m) for m in FWI_METHODS]
            + [InversionCase("tlfwi", m, iterations=1) for m in TL_METHODS])


def example_cases():
    """One case per example script, in alphabetical order."""
    import glob
    import os
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example", "scripts")
    return [ExampleCase(os.path.splitext(os.path.basename(p))[0])
            for p in sorted(glob.glob(os.path.join(folder, "*.py"))) if not os.path.basename(p).startswith("_")]


def all_cases():
    """Every golden-master case: propagation, inversion and example cases."""
    return propagation_cases() + inversion_cases() + example_cases()
