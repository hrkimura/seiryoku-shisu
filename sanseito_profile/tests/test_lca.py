import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import figures  # noqa: E402
import lca  # noqa: E402
import subgroups as sg  # noqa: E402


def test_encode_maps_answers_and_missing():
    df = pd.DataFrame({c: [1, 5, 99, np.nan] for c in lca.ITEMS})
    x = lca.encode(df)
    assert x.dtype.kind == "i"
    assert x[:, 0].tolist() == [0, 4, -1, -1]


def test_onehot_rows_and_missing():
    x = np.array([[0, -1], [4, 2]])
    oh = lca._onehot(x)
    assert oh.shape == (2, 2, 5)
    assert oh[0, 0].tolist() == [1, 0, 0, 0, 0]
    assert oh[0, 1].sum() == 0
    assert oh[1, 0, 4] == 1 and oh[1, 1, 2] == 1 and oh.sum() == 3


def _two_class_data(n=600, j=8, frac=0.3, missing=0.0, seed=0):
    rng = np.random.default_rng(seed)
    z = (rng.random(n) < frac).astype(int)
    probs = {0: [0.7, 0.2, 0.06, 0.02, 0.02], 1: [0.02, 0.02, 0.06, 0.2, 0.7]}
    x = np.array([[rng.choice(5, p=probs[zi]) for _ in range(j)] for zi in z])
    x[rng.random(x.shape) < missing] = -1
    return x, z


@pytest.mark.parametrize("missing", [0.0, 0.2])
def test_em_recovers_two_classes(missing):
    x, z = _two_class_data(missing=missing)
    ll, pi, theta, resp = lca.fit(x, 2, n_starts=3)
    labels = resp.argmax(axis=1)
    assert adjusted_rand_score(z, labels) > 0.95
    assert sorted(pi) == pytest.approx([0.3, 0.7], abs=0.05)
    assert np.allclose(theta.sum(axis=2), 1, atol=1e-6)
    assert np.allclose(resp.sum(axis=1), 1)
    hi = int(np.argmax(pi))
    assert theta[hi, :, 0].mean() == pytest.approx(0.7, abs=0.06)


def test_em_loglik_matches_direct_computation():
    x, _ = _two_class_data(n=200, j=4, seed=1)
    ll, pi, theta, _ = lca.em(x, 2, seed=3)
    oh = lca._onehot(x)
    logp = np.log(pi)[None, :] + np.einsum("njc,kjc->nk", oh, np.log(theta + lca.EPS))
    direct = float(np.log(np.exp(logp).sum(axis=1)).sum())
    # em は最後の更新の前の対数尤度を返すので、更新後の値以下で、差は収束判定の幅程度
    assert ll <= direct + 1e-6
    assert direct - ll < 1e-3


def test_fit_keeps_best_start():
    x, _ = _two_class_data(n=200, j=4, seed=2)
    best = lca.fit(x, 3, n_starts=4, seed=0)
    singles = [lca.em(x, 3, s)[0] for s in range(4)]
    assert best[0] == pytest.approx(max(singles))


def test_n_params_and_bic():
    assert lca.n_params(2) == 1 + 2 * len(lca.ITEMS) * 4
    assert lca.bic(-100.0, 2, 50) == pytest.approx(200 + lca.n_params(2) * math.log(50))


def test_sanseito_by_cluster_counts_voters_only():
    df = pd.DataFrame({"Q4": [6, 6, 1, 13, 6, 2, 2, 99]})
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    t = lca.sanseito_by_cluster(df, labels).set_index("cluster")
    assert t.loc[0, "n"] == 4 and t.loc[0, "n_voted"] == 3 and t.loc[0, "n_sanseito"] == 2
    assert t.loc[1, "n_voted"] == 3 and t.loc[1, "n_sanseito"] == 1
    assert t.loc[0, "share"] == pytest.approx(2 / 3) and t.loc[0, "share_rest"] == pytest.approx(1 / 3)
    assert t.loc[0, "share_of_all_sanseito"] == pytest.approx(2 / 3)
    assert t.loc[0, "p"] == pytest.approx(stats.fisher_exact([[2, 1], [1, 2]])[1])
    assert t["p_holm"].tolist() == pytest.approx(lca.holm(t["p"].to_numpy()).tolist())


def test_share_valid_sets():
    s = pd.Series([1, 2, 3, 99, np.nan])
    assert sg.share(s, {1}, None) == (1, 3)
    assert sg.share(s, {1, 2}, {1, 2}) == (2, 2)


def test_compare_values_and_fisher():
    a = pd.DataFrame({"Q2": [7, 7, 1, 99]})
    b = pd.DataFrame({"Q2": [1, 1, 7, 1]})
    t = sg.compare(a, b, {"参政党を支持": ("Q2", {7}, None)}).iloc[0]
    assert (t.n_a, t.p_a, t.n_b, t.p_b) == (3, pytest.approx(2 / 3), 4, pytest.approx(0.25))
    assert t.p == pytest.approx(stats.fisher_exact([[2, 1], [1, 3]])[1])


def test_compare_thermo_excludes_nonresponse():
    a = pd.DataFrame({c: [10, 8, 99] for c in sg.THERMO.values()})
    b = pd.DataFrame({c: [2, 4, 6] for c in sg.THERMO.values()})
    t = sg.compare_thermo(a, b)
    assert t["n_a"].tolist() == [2, 2, 2] and t["mean_a"].tolist() == [9.0, 9.0, 9.0]
    assert t["mean_b"].tolist() == [4.0, 4.0, 4.0]
    assert t["p"].iloc[0] == pytest.approx(stats.mannwhitneyu([10, 8], [2, 4, 6]).pvalue)


def test_mean_ci_t_interval():
    m, lo, hi = figures.mean_ci(pd.Series([1.0, 2.0, 3.0]))
    h = stats.t.ppf(0.975, 2) / math.sqrt(3)
    assert (m, lo, hi) == (2.0, pytest.approx(2 - h), pytest.approx(2 + h))


def test_encode_every_answer_code():
    df = pd.DataFrame({c: [1, 2, 3, 4, 5, 0, 6] for c in lca.ITEMS})
    assert lca.encode(df)[:, 3].tolist() == [0, 1, 2, 3, 4, -1, -1]


def test_sanseito_by_cluster_asymmetric_and_empty():
    q4 = [6, 6, 6, 1] + [6, 2, 2, 2, 2, 2, 2] + [13, 99]
    labels = np.array([0] * 4 + [1] * 7 + [2] * 2)
    t = lca.sanseito_by_cluster(pd.DataFrame({"Q4": q4}), labels).set_index("cluster")
    assert t.loc[0, "share_rest"] == pytest.approx(1 / 7)
    assert t.loc[0, "p"] == pytest.approx(stats.fisher_exact([[3, 1], [1, 6]])[1])
    assert t.loc[1, "p"] == pytest.approx(stats.fisher_exact([[1, 6], [3, 1]])[1])
    assert t.loc[2, "n_voted"] == 0 and math.isnan(t.loc[2, "share"])


def test_compare_uses_valid_set_and_columns():
    a = pd.DataFrame({"Q40": [6, 7, 10, 99, 16]})
    b = pd.DataFrame({"Q40": [10, 11, 6, 99, 20]})
    items = {"40〜59歳": ("Q40", set(range(6, 10)), set(range(1, 16)))}
    t = sg.compare(a, b, items)
    r = t.iloc[0]
    assert r["label"] == "40〜59歳" and (r.n_a, r.n_b) == (3, 3)
    assert r.p_a == pytest.approx(2 / 3) and r.p_b == pytest.approx(1 / 3)
    assert t["p_holm"].tolist() == pytest.approx(lca.holm(t["p"].to_numpy()).tolist())


def test_compare_thermo_bounds_and_columns():
    a = pd.DataFrame({c: [0, 10, 11, -1] for c in sg.THERMO.values()})
    b = pd.DataFrame({c: [0, 0, 10, 11] for c in sg.THERMO.values()})
    t = sg.compare_thermo(a, b)
    assert t["label"].tolist() == list(sg.THERMO)
    assert t["n_a"].tolist() == [2] * 3 and t["mean_a"].tolist() == [5.0] * 3
    assert t["n_b"].tolist() == [3] * 3 and t["mean_b"].tolist() == pytest.approx([10 / 3] * 3)
    assert t["p_holm"].tolist() == pytest.approx(lca.holm(t["p"].to_numpy()).tolist())
