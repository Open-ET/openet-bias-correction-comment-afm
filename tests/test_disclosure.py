"""Tests for the joint disclosure search (etcomment.disclosure) on synthetic systems; no test reads data files."""
import numpy as np
import pandas as pd
import pytest

from etcomment import disclosure as dc
from etcomment import openet_compare as oc


def _rows(sets: list, n: int, weights=None) -> np.ndarray:
    a = np.zeros((len(sets), n))
    for i, s in enumerate(sets):
        a[i, list(s)] = 1.0 if weights is None else weights[list(s)]
    return a


def test_disjoint_large_sums_are_certified():
    a = _rows([range(10 * i, 10 * i + 10) for i in range(6)], 60)
    out = dc.small_support_search(a, list(range(60)))
    assert out["confirmed"] == [] and out["m"] >= dc.MIN_SITE_MONTHS


def test_planted_differences_and_single_values_are_found():
    s = list(range(10))
    a = _rows([s + [10, 11], s, list(range(12, 30)), s + [30]], 40)
    out = dc.small_support_search(a, list(range(40)))
    assert out["confirmed"] == [(30,), (10, 11)]   # row 4 - row 2 gives x30; row 1 - row 2 gives x10 + x11
    # with real-valued weights (gradients of nonlinear statistics) the same structure is found
    w = np.linspace(0.3, 2.1, 40)
    out_w = dc.small_support_search(_rows([s + [10, 11], s, s + [30]], 40, w), list(range(40)))
    assert out_w["confirmed"] == [(30,), (10, 11)]


def test_groups_hold_several_unknowns():
    """Two unknowns of one site-month (e.g. two towers in one gridMET cell) count once."""
    groups = [j // 2 for j in range(40)]  # columns 2g and 2g + 1 are site-month g
    s = list(range(20))
    a = _rows([s + [20, 21, 22, 23, 24, 25], s], 40)  # differ by columns 20-25: site-months 10, 11, 12
    assert dc.small_support_search(a, groups)["confirmed"] == [(10, 11, 12)]


@pytest.mark.parametrize("seed", range(8))
def test_agrees_with_the_enumeration_of_openet_compare(seed):
    rng = np.random.default_rng(seed)
    n = 28
    sets = [rng.choice(n, size=rng.integers(6, 16), replace=False) for _ in range(7)]
    sets.append(np.setdiff1d(sets[0], sets[0][: rng.integers(1, 4)]))  # a planted small difference
    a = _rows(sets, n)
    groups = [j // 2 if j < 8 else j for j in range(n)]
    fast = dc.small_support_search(a, groups)["confirmed"]
    gmap = dict(zip(map(str, range(n)), groups))
    slow = oc.small_supports(pd.DataFrame(a, columns=list(gmap)), group=lambda c: gmap[c])
    assert fast == dc.minimal(slow) and fast  # the planted difference is always there


def test_minimal_and_empty_rows():
    assert dc.minimal([(1, 2), (1,), (3, 4), (3, 4, 5)]) == [(1,), (3, 4)]
    assert dc.small_support_search(np.zeros((2, 5)), list(range(5)))["confirmed"] == []
