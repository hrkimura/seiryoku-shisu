"""役場の位置(国土数値情報 P05)から、各単位の近くの単位を求め、近隣での外国人の増え方を計算する。"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw"
EARTH_KM = 6371.0
# 施設分類1は本庁(市役所・区役所・町村役場)
HEAD_OFFICE = "1"


def offices(paths: list[Path]) -> dict[str, tuple[float, float]]:
    """5桁の市区町村コード → 本庁の (経度, 緯度)。同じコードに複数あれば平均する。"""
    pts: dict[str, list[tuple[float, float]]] = {}
    for f in paths:
        z = zipfile.ZipFile(f)
        name = next(n for n in z.namelist() if n.endswith(".geojson"))
        for ft in json.loads(z.read(name))["features"]:
            p = ft["properties"]
            if p["P05_002"] == HEAD_OFFICE:
                pts.setdefault(p["P05_001"], []).append(tuple(ft["geometry"]["coordinates"]))
    return {c: tuple(np.mean(v, axis=0)) for c, v in pts.items()}


def unit_location(codes: str, office: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """「01101+01102」のような単位のコードから位置。コードに本庁がなければ、上4桁が同じ本庁の平均
    (区を再編した浜松市のように、市全体のコードに本庁がない場合)。"""
    pts = []
    for c in codes.split("+"):
        if c in office:
            pts.append(office[c])
        else:
            near = [v for k, v in office.items() if k[:4] == c[:4]]
            if near:
                pts.append(tuple(np.mean(near, axis=0)))
    return tuple(np.mean(pts, axis=0)) if pts else (np.nan, np.nan)


def distances_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lo, la = np.radians(lon), np.radians(lat)
    dlo = lo[:, None] - lo[None, :]
    dla = la[:, None] - la[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def neighbor_change(t: pd.DataFrame, dist: np.ndarray, radius_km: float, num: str, den: str) -> np.ndarray:
    """半径内の他の単位をまとめた割合の変化(%ポイント)。num は増加数、den は基準の人口。近くに単位がなければ NaN。"""
    near = (dist <= radius_km) & ~np.eye(len(t), dtype=bool)
    n = near @ t[num].to_numpy(float)
    d = near @ t[den].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / d * 100, np.nan)
