import numpy as np
import pytest

from eolandscape import diversity as dv
from eolandscape import fragmentation as fr
from eolandscape import landcover as lc
from eolandscape import zonal


def test_nomenclature_has_44_classes():
    assert len(lc.CLC_NOMENCLATURE) == 44


def test_class_area_and_share_explicit_domain():
    classes = np.array([[211, 211], [311, 512]])
    tab = lc.class_area(classes, 10_000.0)  # 1 ha pixels
    assert dict(zip(tab["key"], tab["area_ha"], strict=True)) == {211: 2.0, 311: 1.0, 512: 1.0}
    share = lc.class_share(classes, 10_000.0, domain_classes=[211, 311], groups={211: "arable", 311: "forest"})
    got = dict(zip(share["key"], share["share_pct"], strict=True))
    assert got["arable"] == pytest.approx(200 / 3) and got["forest"] == pytest.approx(100 / 3)


def test_diversity_formulas_and_bounds():
    p = np.array([[0.5, 1.0, 0.25], [0.5, 0.0, 0.25], [0.0, 0.0, 0.25], [0.0, 0.0, 0.25]])
    h = dv.shannon(p)
    assert h[0] == pytest.approx(np.log(2)) and h[1] == 0 and h[2] == pytest.approx(np.log(4))
    j = dv.pielou_evenness(p)
    assert j[0] == pytest.approx(1.0) and np.isnan(j[1]) and j[2] == pytest.approx(1.0)
    d = dv.simpson(p)
    assert d[2] == pytest.approx(0.75)
    n1, n2 = dv.hill_numbers(p)
    assert n1[2] == pytest.approx(4) and n2[2] == pytest.approx(4)


def test_window_proportions_sum_to_one_and_coverage():
    rng = np.random.default_rng(4)
    classes = rng.choice([1, 2, 3, 9], size=(30, 30))
    props, cov = dv.class_proportions(classes, [1, 2, 3], radius_m=300, pixel_size_m=100)
    s = np.nansum(props, axis=0)
    assert np.allclose(s[np.isfinite(props[0])], 1.0)
    assert np.all((cov >= 0) & (cov <= 1))
    h = dv.shannon(props)
    assert np.nanmax(h) <= np.log(3) + 1e-12


def test_distance_uses_real_pixel_size():
    focal = np.zeros((7, 7), bool)
    focal[1:6, 1:6] = True
    d10 = fr.distance_to_edge(focal, 10.0)
    d100 = fr.distance_to_edge(focal, 100.0)
    assert d10[3, 3] == pytest.approx(30.0)
    assert d100[3, 3] == pytest.approx(300.0)
    assert d10[1, 1] == pytest.approx(10.0)


def test_core_edge_partition():
    focal = np.zeros((9, 9), bool)
    focal[1:8, 1:8] = True
    core, edge = fr.core_edge(focal, 30.0, edge_width_m=60.0)
    assert not np.any(core & edge)
    assert np.array_equal(core | edge, focal)
    assert core.sum() == 9  # the 3x3 centre is > 60 m from non-focal


def test_riitters_pf_pff():
    f = np.ones((5, 5), bool)
    pf, pff = fr.riitters_pf_pff(f, 3)
    assert np.allclose(pf, 1.0) and np.nanmin(pff) == 1.0
    checker = (np.indices((5, 5)).sum(axis=0) % 2 == 0)
    pf, pff = fr.riitters_pf_pff(checker, 5)
    assert pff[2, 2] == 0.0  # no focal-focal cardinal pairs
    with pytest.raises(ValueError):
        fr.riitters_pf_pff(f, 4)


def test_edge_density():
    f = np.zeros((2, 2), bool)
    f[:, 0] = True
    # two shared vertical sides of 10 m = 20 m over 400 m^2 = 0.04 ha
    assert fr.edge_density(f, 10.0) == pytest.approx(500.0)


def test_zonal_stats():
    values = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    labels = np.array([[1, 1, 1], [2, 2, 0]])
    out = zonal.zonal_stats(values, labels)
    assert out.loc[1, "n_total"] == 3 and out.loc[1, "n_valid"] == 2
    assert out.loc[1, "valid_fraction"] == pytest.approx(2 / 3)
    assert out.loc[2, "mean"] == pytest.approx(4.5)
    assert 0 not in out.index


def test_all_focal_has_no_edge():
    d = fr.distance_to_edge(np.ones((3, 3), bool), 10.0)
    assert np.all(np.isinf(d))
    core, edge = fr.core_edge(np.ones((3, 3), bool), 10.0, edge_width_m=10.0)
    assert core.all() and not edge.any()


def test_fft_window_counts_match_direct_count():
    rng = np.random.default_rng(9)
    classes = rng.choice([1, 2], size=(15, 15))
    props, _ = dv.class_proportions(classes, [1, 2], radius_m=20, pixel_size_m=10)
    k = dv.disk_kernel(2)
    win = classes[5:10, 5:10][k == 1]
    assert props[0, 7, 7] == pytest.approx(np.mean(win == 1))
