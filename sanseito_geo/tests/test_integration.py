import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402

pytestmark = pytest.mark.skipif(not (dataset.RAW / "sangiin_2025").exists(), reason="生データが未取得")


def test_build_matches_official_totals():
    t = dataset.build()
    # 総務省の全国計(比例代表、参政党)と住民基本台帳の外国人住民数
    assert t["votes25"].sum() == pytest.approx(7425053, abs=2)
    assert t["votes22"].sum() == pytest.approx(1768385, abs=2)
    assert t["for25"].sum() == pytest.approx(3677463, rel=1e-4)
    assert t["unit"].is_unique or t.duplicated(["pref", "unit"]).sum() == 0
    assert t[["share22", "share25", "fshare22", "pred_fshare"]].notna().all().all()
    assert len(t) == 1883
