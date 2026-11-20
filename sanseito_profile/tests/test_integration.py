"""実データで集計から図までを通す。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import figures  # noqa: E402
import party_distance as pdist  # noqa: E402
import profile_utas as pu  # noqa: E402

pytestmark = pytest.mark.skipif(not pu.DATA.exists(), reason="東大朝日2026の個票がない")


def test_pipeline_on_real_data(tmp_path):
    df = pu.load()
    assert len(df) == 1449 and int(df.sanseito.sum()) == 103
    t = pu.profile_table(df)
    assert len(t) == len(pu.ITEMS)
    edu = t.set_index("label").loc["大学・大学院卒"]
    assert edu.p_s == pytest.approx(32 / 102)

    d = df[df.Q4.isin(pdist.PARTIES)]
    ci = figures.distance_with_ci(d, pdist.DEMO, n_boot=20)
    assert len(ci) == 28
    figures.setup()
    figures.fig_distance(ci, tmp_path / "a.png")
    figures.fig_profile(df, tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 10000 and (tmp_path / "b.png").stat().st_size > 10000


def test_lca_outputs_and_figures(tmp_path):
    import lca
    import subgroups as sg
    t = pd.read_csv(pu.OUT / "utas2026_lca_sanseito.csv")
    assert len(t) == len(lca.CLASS_NAMES)
    # 名前の付け方が今の番号と合っていることを確かめる(参政党の割合が最大なのが「強く反対する右派」)
    assert lca.CLASS_NAMES[int(t.sort_values("share").iloc[-1]["cluster"])].startswith("外国人の受け入れ")
    figures.setup()
    figures.fig_lca(t, dict(zip(t["cluster"], t["n"])), tmp_path / "c.png")
    df = pu.load()
    s = df[df.Q4 == 6]
    figures.fig_thermo(s[s.Q3 == sg.Q3_SANSEITO], s[s.Q3 == sg.Q3_LDP], tmp_path / "d.png")
    assert (tmp_path / "c.png").stat().st_size > 10000 and (tmp_path / "d.png").stat().st_size > 10000
    figures.fig_typical(pd.read_csv(pu.OUT / "utas2026_profile.csv"), tmp_path / "e.png")
    figures.fig_survey_vs_actual(figures.survey_vs_actual(df), tmp_path / "f.png")
    raw = pd.read_csv(pu.DATA)
    figures.fig_class_attitudes(raw, pd.read_csv(pu.OUT / "utas2026_lca.csv"), 4, tmp_path / "g.png")
    figures.fig_stacked([("a", figures.composition(raw["Q2"], {"参政党": {7}}, "その他"))],
                        {"参政党": figures.C_SANSEI, "その他": figures.GRAYS[0]}, "t", tmp_path / "h.png")
    figures.fig_dislike_cabinet(pd.read_csv(pu.OUT / "utas2026_dislike_vs_cabinet.csv"), tmp_path / "i.png")
    assert all((tmp_path / f"{x}.png").stat().st_size > 5000 for x in "efghi")
