"""Unit tests of the golden-master case matrix (``test/golden_cases.py``), no device needed."""
import itertools

import pytest

from golden_cases import FACTORS, PropagationCase, pairwise_rows, propagation_cases


def _pairs(rows, factors):
    """Set of ((factor, value), (factor, value)) pairs that appear together in ``rows``."""
    names = list(factors)
    return {((a, row[a]), (b, row[b])) for row in rows for i, a in enumerate(names) for b in names[i + 1:]}


def _all_pairs(factors):
    names = list(factors)
    return {((a, va), (b, vb)) for i, a in enumerate(names) for b in names[i + 1:]
            for va in factors[a] for vb in factors[b]}


@pytest.mark.parametrize("factors", [
    FACTORS,
    {"x": (1, 2), "y": (1, 2), "z": (1, 2)},
    {"a": (1, 2, 3, 4), "b": ("p", "q"), "c": (True, False), "d": (0, 1, 2)},
])
def test_pairwise_rows_cover_every_pair(factors):
    rows = pairwise_rows(factors)
    assert _pairs(rows, factors) == _all_pairs(factors)
    assert all(set(row) == set(factors) and all(row[k] in factors[k] for k in row) for row in rows)


def test_pairwise_matrix_is_much_smaller_than_the_full_product():
    full = len(list(itertools.product(*FACTORS.values())))
    rows = pairwise_rows(FACTORS)
    largest_two = sorted((len(v) for v in FACTORS.values()), reverse=True)[:2]
    assert largest_two[0] * largest_two[1] <= len(rows) <= 20 < full  # 15 is the theoretical minimum here


def test_pairwise_rows_are_deterministic():
    assert pairwise_rows(FACTORS) == pairwise_rows(dict(FACTORS))


def test_propagation_cases_have_unique_names():
    cases = propagation_cases()
    names = [c.name for c in cases]
    assert len(set(names)) == len(names)
    assert cases[0].name.startswith("a") and "_chpr" in cases[0].name


def test_case_as_dict_round_trips():
    case = propagation_cases()[0]
    assert PropagationCase(**case.as_dict()) == case
