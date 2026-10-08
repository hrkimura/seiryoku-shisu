import pytest

from seiryoku.fetch import wikipedia_governor


@pytest.fixture(autouse=True)
def _isolate_wikipedia(request, tmp_path, monkeypatch):
    """単体テストでは手元の取得結果やネットワークに依存させない(結合テストは実データを使う)。"""
    if "integration" in request.node.fspath.basename:
        return
    monkeypatch.setattr(wikipedia_governor, "CACHE_PATH", tmp_path / "wiki_cache.json")
    monkeypatch.setattr(wikipedia_governor, "MANUAL_PATH", tmp_path / "wiki_manual.json")

    def offline(title):
        raise ConnectionError("unit tests do not access Wikipedia")

    monkeypatch.setattr(wikipedia_governor, "_fetch_article", offline)
