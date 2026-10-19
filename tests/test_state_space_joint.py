import math
from datetime import date, timedelta

import numpy as np
import pytest

from seiryoku import state_space_glm as glm
from seiryoku import state_space_joint as joint
from seiryoku.trend import Event

START = date(2018, 1, 7)


def _events(seed=0, M=40, years=5, parties=("A",), base=(0.3,)):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 0.8, (M, len(parties)))
    events = []
    for d in range(0, years * 365, 7):
        day = START + timedelta(days=d)
        for m in range(M):
            if (d // 7 + 5 * m) % 104 == 0:
                n = int(rng.integers(5, 30))
                eta = np.array([math.log(b / (1 - sum(base))) for b in base]) + u[m]
                p = np.exp(eta) / (1 + np.exp(eta).sum())
                k = rng.multinomial(n, list(p) + [1 - p.sum()])
                sh = {q: k[a] / n for a, q in enumerate(parties)}
                sh["他"] = k[-1] / n
                events.append(Event(day, "市区町村議会", 1e5, float(n), sh, f"市{m}"))
        if d % 364 == 0:
            sh = {q: round(b * 465) / 465 for q, b in zip(parties, base)}
            sh["他"] = 1 - sum(sh.values())
            events.append(Event(day, "衆院選", 1.2e8, 465.0, sh, ""))
    return events


def test_one_party_joint_model_equals_the_party_model():
    events = _events()
    first = {"A": START}
    data = joint.joint_data(events, ["A"], first)
    obs = glm.glm_observations(events, "A", START, filter_levels=False)
    gp = glm.GlmParams(q_x=0.03, phi={"市区町村議会": 1e9}, ell=0.6, tau2={"市区町村議会": 0.7})
    jp = joint.JointParams(q={"A": 0.03}, ell={"A": 0.6}, tau2={("A", "市区町村議会"): 0.7})
    rg = glm.run_filter(obs, gp, state_levels=["市区町村議会"])
    rj = joint.run_filter(data, jp, levels=["市区町村議会"])
    assert rj.mean[0] == pytest.approx(rg.mean[0], abs=1e-6)
    assert rj.cov[0, 0] == pytest.approx(rg.cov[0, 0], rel=1e-5)
    assert rj.loglik == pytest.approx(rg.loglik, abs=1e-4)


def test_joint_smoother_returns_the_filter_value_at_the_last_observation():
    events = _events(parties=("A", "B"), base=(0.3, 0.2), seed=1)
    data = joint.joint_data(events, ["A", "B"], {"A": START, "B": START})
    jp = joint.JointParams(q={"A": 0.03, "B": 0.05}, ell={"A": 0.6, "B": 0.8},
                           tau2={("A", "市区町村議会"): 0.6, ("B", "市区町村議会"): 0.6})
    r = joint.run_filter(data, jp)
    (m, C), = joint.smooth(data, jp, [data.t[-1]])[0]
    assert np.allclose(m, r.mean)
    assert np.allclose(C, r.cov)


def test_set_likelihood_reduces_to_a_single_winner_and_has_correct_derivatives():
    eta = np.array([0.4, -0.3, 0.1])
    single = joint._set_loglik(np.array([False, True, False]), False)(eta)
    counts = joint._counts_loglik(np.array([0.0, 1.0, 0.0]), 0.0, 1.0)(eta)
    for a, b in zip(single, counts):
        assert np.allclose(a, b)
    f = joint._set_loglik(np.array([True, True, False]), True)
    ll, g, W = f(eta)
    h = 1e-6
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        assert g[j] == pytest.approx((f(eta + e)[0] - f(eta - e)[0]) / (2 * h), abs=1e-6)
        assert np.allclose(-W[:, j], (f(eta + e)[1] - f(eta - e)[1]) / (2 * h), atol=1e-5)


def test_joint_allocation_of_a_shared_mayor_follows_party_strength():
    """自公の相乗りの当選は、党勢の強い党にほぼ配分される。"""
    eta = np.array([math.log(0.6 / 0.3), math.log(0.05 / 0.3)])
    mu, mu0 = joint._softmax(eta)
    _, g, _ = joint._set_loglik(np.array([True, True]), False)(eta)
    r = g + mu
    assert r[0] == pytest.approx(0.6 / 0.65)
    assert r[1] == pytest.approx(0.05 / 0.65)


def test_negatively_correlated_moves_pass_one_party_surprise_to_the_other():
    """党勢の増分に負の相関があると、Aが地方選挙で予想より勝ったとき、Bの党勢も下に修正される。"""
    events = _events(parties=("A", "B"), base=(0.3, 0.2), seed=2)
    last = events[-1].vote_date
    events.append(Event(last + timedelta(days=200), "市区町村議会", 1e5, 30.0, {"A": 0.9, "B": 0.1}, "市0"))
    data = joint.joint_data(events, ["A", "B"], {"A": START, "B": START})
    base = dict(q={"A": 0.05, "B": 0.05}, ell={"A": 1.0, "B": 1.0},
                tau2={("A", "市区町村議会"): 0.3, ("B", "市区町村議会"): 0.3})
    lay = joint._Layout(["A", "B"], ["市区町村議会"])
    zi = [lay.z(0), lay.z(1)]
    dz = {}
    for rho in (0.0, -0.9):
        jp = joint.JointParams(corr=np.array([[1.0, rho], [rho, 1.0]]), **base)
        before = joint.run_filter(data, jp, until=data.t[-2]).mean
        after = joint.run_filter(data, jp).mean
        dz[rho] = (after - before)[zi]
    assert dz[0.0][0] > 0
    assert dz[-0.9][1] < dz[0.0][1] - 0.01
    np.testing.assert_allclose(joint.JointParams(**base).noise(["A", "B"]), np.diag([0.05, 0.05]))
