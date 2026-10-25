"""Unit tests of the geometry helpers in ``test/helpers.py`` (no OpenCL device needed).

The helpers model where the kernels of PyFWI 0.1.10 really record and inject
(``kernel_receiver_cells``), find the grid points that trigger race conditions
(``coincidences``) and move sources and receivers away from them
(``race_free_geometry``). Golden masters and examples rely on them to be
reproducible, so they are tested on their own.
"""
import numpy as np
import pytest

from helpers import coincidences, grid_cell, kernel_receiver_cells, race_free_geometry

#: Geometry of the small test cases: 40x40 grid, 10 m spacing, a receiver every 10 m.
DH, REC_DIS, NZ, NX = 10.0, 10.0, 40, 40


def _default_geometry(acq_type, ns=2, sdo=4):
    """Default PyFWI acquisition for the small grid: ``(src_loc, rec_loc, n_well_rec)``."""
    import PyFWI.acquisition as acq
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(ns, REC_DIS, DH * NX, DH * NZ, DH, sdo, acq_type=acq_type)
    return np.asarray(src_loc, np.float32), np.asarray(rec_loc, np.float32), n_well_rec


def test_grid_cell_truncates_like_pyfwi():
    assert grid_cell(np.array([[0.0, 0.0], [19.9, 30.0], [40.0, 5.0]]), 10.0) == [(0, 0), (1, 3), (4, 0)]


def test_surface_line_is_contiguous_from_its_first_receiver():
    rec = np.array([[20.0, 40.0], [30.0, 40.0], [999.0, 999.0]])  # the kernel ignores the 3rd position
    assert kernel_receiver_cells(rec, 0, DH, REC_DIS, NZ) == [(2, 4), (3, 4), (4, 4)]


def test_crosswell_well_runs_to_the_bottom_of_the_grid():
    _, rec, n_well = _default_geometry(acq_type=0)
    cells = kernel_receiver_cells(rec, n_well, DH, REC_DIS, NZ)
    first = grid_cell(rec[:1], DH)[0]
    assert len(cells) > n_well  # unbounded: more cells than receivers
    assert cells == [(first[0], z) for z in range(first[1], NZ)]


def test_both_acquisition_has_left_well_surface_and_unbounded_right_well():
    _, rec, n_well = _default_geometry(acq_type=2)
    cells = kernel_receiver_cells(rec, n_well, DH, REC_DIS, NZ)
    n_surface = rec.shape[0] - 2 * n_well
    left, surface, right = cells[:n_well], cells[n_well:n_well + n_surface], cells[n_well + n_surface:]
    assert len({c[0] for c in left}) == 1 and len({c[1] for c in surface}) == 1 and len({c[0] for c in right}) == 1
    assert right[-1][1] == NZ - 1


@pytest.mark.parametrize("acq_type, key", [(1, "source_on_receiver"), (2, "shared_receiver_cell")])
def test_default_geometry_of_0_1_10_has_race_points(acq_type, key):
    # Documents why the golden masters cannot use the default geometry of acq_parameters.
    src, rec, n_well = _default_geometry(acq_type)
    assert coincidences(src, rec, n_well, DH, REC_DIS, NZ)[key]


@pytest.mark.parametrize("acq_type", [0, 1, 2])
@pytest.mark.parametrize("sdo", [4, 8])
@pytest.mark.parametrize("ns", [1, 2, 4])
def test_race_free_geometry_removes_every_race_point(acq_type, sdo, ns):
    src, rec, n_well = _default_geometry(acq_type, ns, sdo)
    src2, rec2 = race_free_geometry(src, rec, n_well, DH, REC_DIS, NZ)
    assert coincidences(src2, rec2, n_well, DH, REC_DIS, NZ) == {"source_on_receiver": [], "shared_receiver_cell": []}
    assert src2.shape == src.shape  # every source is kept (only moved)
    assert np.all(src2 >= 0) and np.all(src2[:, 0] < DH * NX) and np.all(src2[:, 1] < DH * NZ)  # inside the model
    if acq_type != 2:
        np.testing.assert_array_equal(rec2, rec)  # receivers are only changed for acq_type 2
    else:
        # Wells unchanged; the surface line keeps exactly the receivers strictly between the well columns.
        left, surface, right = rec[:n_well], rec[n_well:-n_well], rec[-n_well:]
        inside = surface[(surface[:, 0] > left[0, 0]) & (surface[:, 0] < right[0, 0])]
        np.testing.assert_array_equal(rec2, np.vstack([left, inside, right]))
