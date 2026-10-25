"""Results of PyFWI must not depend on the NumPy version (NumPy 2 changed type promotion, NEP 50).

NumPy 1.x promoted a ``numpy.float32`` scalar combined with a Python number to float64; NumPy 2
keeps float32. PyFWI 0.1.10 relied on the old rule in two places, both fixed in step 1.1 by
writing the float64 conversion explicitly:

* the PML damping profile (``fwi_tools.pml_delta_calculation``): checked bit for bit against the
  profile of 0.1.10 in ``test_known_bugs.py::test_pml_profile_does_not_depend_on_numpy``;
* the L2 misfit (``fwi_tools.CostFunction``): its value and its float64 type, below.

The golden-master characterisation (``test_characterization.py``) checks everything else bit for
bit, including the misfit returned by the examples and the inversions.
"""
import numpy as np


def test_l2_misfit_is_float64_with_the_value_of_0_1_10():
    import PyFWI.fwi_tools as tools

    estimate = {"taux": np.arange(12, dtype=np.float32).reshape(3, 4) / 7}
    observed = {"taux": np.ones((3, 4), np.float32)}
    misfit, adjoint = tools.CostFunction("l2")(estimate, observed)
    residual = (estimate["taux"] - observed["taux"]).astype(np.float32)
    expected = 0.5 * np.float64(residual.reshape(-1) @ residual.reshape(-1))  # what NumPy 1.x computed
    assert type(misfit) is np.float64 and misfit == expected
    assert adjoint["taux"].dtype == np.float32
    np.testing.assert_array_equal(adjoint["taux"], residual)
