"""外国人の割合の増加と、参政党の得票率の伸び(2022→2025参院比例)の関係。

重み付き最小二乗と、シフトシェアを操作変数にした二段階最小二乗。標準誤差は都道府県で群をなす頑健なもの。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "output"


@dataclass
class Fit:
    coef: float
    se: float
    n: int
    first_stage_f: float = float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return (self.coef - 1.96 * self.se, self.coef + 1.96 * self.se)


def design(t: pd.DataFrame, controls: list[str]) -> np.ndarray:
    """統制変数と都道府県の固定効果(定数項を含む)。"""
    fe = pd.get_dummies(t["pref"], prefix="p", drop_first=True, dtype=float)
    return np.column_stack([np.ones(len(t)), t[controls].to_numpy(float), fe.to_numpy()])


def cluster_vcov(x: np.ndarray, u: np.ndarray, w: np.ndarray, groups: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """群で頑健な分散(小標本の補正つき)。x は推定に使った説明変数、u は残差。"""
    meat = np.zeros((x.shape[1], x.shape[1]))
    for g in np.unique(groups):
        idx = groups == g
        s = (x[idx] * (w[idx] * u[idx])[:, None]).sum(axis=0)
        meat += np.outer(s, s)
    n, k, g = len(u), x.shape[1], len(np.unique(groups))
    c = g / (g - 1) * (n - 1) / (n - k)
    return c * bread @ meat @ bread


def wls(y, x, w, groups) -> tuple[np.ndarray, np.ndarray]:
    xw = x * w[:, None]
    bread = np.linalg.pinv(x.T @ xw)
    b = bread @ (xw.T @ y)
    return b, cluster_vcov(x, y - x @ b, w, groups, bread)


def complete(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """使う変数に欠損のない行だけを残す(双葉町など、統計のない単位がある)。"""
    return t.dropna(subset=cols)


def ols(t: pd.DataFrame, y: str, d: str, controls: list[str], weight: str = "valid25") -> Fit:
    t = complete(t, [y, d, weight] + controls)
    x = np.column_stack([t[d].to_numpy(float), design(t, controls)])
    b, v = wls(t[y].to_numpy(float), x, t[weight].to_numpy(float), t["pref"].to_numpy())
    return Fit(b[0], float(np.sqrt(v[0, 0])), len(t))


def tsls(t: pd.DataFrame, y: str, d: str, z: str, controls: list[str], weight: str = "valid25") -> Fit:
    t = complete(t, [y, d, z, weight] + controls)
    w = t[weight].to_numpy(float)
    g = t["pref"].to_numpy()
    xc = design(t, controls)
    zx = np.column_stack([t[z].to_numpy(float), xc])
    dv = t[d].to_numpy(float)
    pi, vpi = wls(dv, zx, w, g)
    f = pi[0] ** 2 / vpi[0, 0]
    xhat = np.column_stack([zx @ pi, xc])
    xw = xhat * w[:, None]
    bread = np.linalg.pinv(xhat.T @ xw)
    yv = t[y].to_numpy(float)
    b = bread @ (xw.T @ yv)
    u = yv - np.column_stack([dv, xc]) @ b
    v = cluster_vcov(xhat, u, w, g, bread)
    return Fit(b[0], float(np.sqrt(v[0, 0])), len(t), f)


CONTROLS = ["share22", "fshare22", "log_pop22"]
REGION = ["elderly", "manuf", "secondary", "unemp", "log_density"]
SPECS = {"統制なし": [], "基本の統制": CONTROLS, "基本+地域の特徴": CONTROLS + REGION}


def prepare(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["log_pop22"] = np.log(t["pop22"])
    t["dfor"] = t["for25"] - t["for22"]
    return t


def own_effects(t: pd.DataFrame) -> pd.DataFrame:
    """自分の単位で外国人の割合が1ポイント増えたときの、参政党の得票率の伸び(ポイント)。"""
    rows = []
    for spec, ctrl in SPECS.items():
        for method, fit in (("最小二乗", ols(t, "d_share", "d_fshare", ctrl)),
                            ("操作変数", tsls(t, "d_share", "d_fshare", "pred_fshare", ctrl))):
            rows.append({"target": "自分の市区町村", "spec": spec, "method": method, "coef": fit.coef,
                         "lo": fit.ci[0], "hi": fit.ci[1], "n": fit.n, "first_stage_F": fit.first_stage_f})
    return pd.DataFrame(rows)


def neighbor_effects(t: pd.DataFrame, radius_km: float = 10.0) -> pd.DataFrame:
    """半径内の他の市区町村で外国人の割合が1ポイント増えたときの伸び(自分の増加と近隣の水準を統制)。"""
    import geo

    office = geo.offices(sorted(geo.RAW.glob("p05/*.zip")))
    loc = [geo.unit_location(c, office) for c in t["codes"]]
    t = t.assign(lon=[x[0] for x in loc], lat=[x[1] for x in loc]).dropna(subset=["lon"]).reset_index(drop=True)
    dist = geo.distances_km(t["lon"].to_numpy(), t["lat"].to_numpy())
    t["nb_d"] = geo.neighbor_change(t, dist, radius_km, "dfor", "pop22")
    t["nb_lvl"] = geo.neighbor_change(t, dist, radius_km, "for22", "pop22")
    t["nb_pred"] = geo.neighbor_change(t, dist, radius_km, "pred", "pop22")
    rows = []
    for spec, ctrl in list(SPECS.items())[1:]:
        c2 = ctrl + ["d_fshare", "nb_lvl"]
        for method, fit in (("最小二乗", ols(t, "d_share", "nb_d", c2)),
                            ("操作変数", tsls(t, "d_share", "nb_d", "nb_pred", c2))):
            rows.append({"target": f"半径{radius_km:g}km以内の他の市区町村", "spec": spec, "method": method,
                         "coef": fit.coef, "lo": fit.ci[0], "hi": fit.ci[1], "n": fit.n,
                         "first_stage_F": fit.first_stage_f})
    return pd.DataFrame(rows)


def main() -> None:
    t = prepare(pd.read_csv(OUT / "units.csv", dtype={"codes": str}))
    r = pd.concat([own_effects(t), neighbor_effects(t)], ignore_index=True)
    r.to_csv(OUT / "main_results.csv", index=False)
    print(r.round(3).to_string(index=False))
    print("参政党の伸びの加重平均", round(np.average(t["d_share"], weights=t["valid25"]), 2),
          "外国人の割合の増加の加重平均", round(np.average(t["d_fshare"], weights=t["valid25"]), 2))


if __name__ == "__main__":
    main()
