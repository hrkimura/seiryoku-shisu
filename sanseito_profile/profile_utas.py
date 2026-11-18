"""東大谷口研・朝日2026年有権者調査で、2025年参院選の比例で参政党に投票した人のプロフィルを、他党の投票者と比べる。"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "utas" / "2026UTASV20260602.csv"
OUT = HERE / "output"

SANSEITO = 6
VOTED = set(range(1, 13))
NONRESPONSE = 99

PARTY_Q4 = {1: "自民", 2: "立憲", 3: "国民", 4: "公明", 5: "維新", 6: "参政", 7: "共産",
            8: "れいわ", 9: "保守", 10: "社民", 11: "みらい", 12: "その他"}

PRIORITY_Q37 = {1: "外交・安全保障", 2: "産業・雇用", 3: "財政・税制・金融", 4: "年金・医療・介護",
                5: "こども・子育て・教育", 6: "環境・原発・エネルギー", 7: "憲法", 8: "政治・行政改革",
                9: "地方・インフラ・防災", 10: "外国人・多文化共生", 11: "ジェンダー・家族", 12: "治安・司法"}


@dataclass(frozen=True)
class Item:
    group: str
    label: str
    col: str
    yes: frozenset[int]
    valid: frozenset[int] | None = None  # None は無回答(99)以外すべてを分母にする


def _r(a: int, b: int) -> frozenset[int]:
    return frozenset(range(a, b + 1))


AGREE = _r(1, 2)
ITEMS: list[Item] = [
    Item("属性", "男性", "Q39", frozenset({1}), frozenset({1, 2})),
    Item("属性", "18〜39歳", "Q40", _r(1, 5)),
    Item("属性", "40〜59歳", "Q40", _r(6, 9)),
    Item("属性", "60歳以上", "Q40", _r(10, 15)),
    Item("属性", "大学・大学院卒", "Q43", frozenset({5, 6}), _r(1, 6)),
    Item("属性", "中学・高校卒", "Q43", frozenset({1, 2}), _r(1, 6)),
    Item("属性", "事務・技術職", "Q41_1", frozenset({1})),
    Item("属性", "製造・サービス職", "Q41_1", frozenset({2})),
    Item("属性", "自営業", "Q41_1", frozenset({3})),
    Item("属性", "主婦・主夫", "Q41_1", frozenset({5})),
    Item("属性", "無職", "Q41_1", frozenset({7})),
    # 雇用形態は Q41_1 が 1・2 の人だけに聞いている(66 は非該当)
    Item("属性", "非正規雇用(雇用者のうち)", "Q41_2", frozenset({3}), _r(1, 4)),
    Item("属性", "既婚(配偶者あり)", "Q42", frozenset({2}), _r(1, 3)),
    Item("属性", "生活水準が中の下以下", "Q46", _r(5, 7), _r(1, 7)),
    Item("情報源", "新聞", "Q49_1", frozenset({1})),
    Item("情報源", "テレビ", "Q49_2", frozenset({1})),
    Item("情報源", "雑誌", "Q49_3", frozenset({1})),
    Item("情報源", "ネットニュース", "Q49_4", frozenset({1})),
    Item("情報源", "SNS", "Q49_5", frozenset({1})),
    Item("情報源", "動画サイト", "Q49_6", frozenset({1})),
    Item("政治", "高市内閣を支持", "Q1", frozenset({1})),
    Item("政治", "憲法を変える必要がある", "Q16", frozenset({1})),
    Item("政治", "自衛隊明記の改憲に賛成", "Q15", frozenset({1})),
    Item("政治", "緊急政令は改憲で対応すべき", "Q19_1", frozenset({1})),
    Item("政治", "AIの未来に悲観的", "Q30", frozenset({2})),
    Item("政治", "妊娠・出産での不利益が残る", "Q21", AGREE),
    Item("政治", "理工系の女性枠に納得できない", "Q22", frozenset({2})),
    Item("政策", "防衛力を強化すべき", "Q35_1", AGREE),
    Item("政策", "首相は靖国に参拝してほしい", "Q35_2", AGREE),
    Item("政策", "日米安保を強化すべき", "Q35_3", AGREE),
    Item("政策", "小さな政府の方が良い", "Q35_4", AGREE),
    Item("政策", "公共事業による雇用確保は必要", "Q35_5", AGREE),
    Item("政策", "財政再建より財政出動", "Q35_6", AGREE),
    Item("政策", "大阪を副首都に", "Q35_7", AGREE),
    Item("政策", "外国人労働者の受け入れを進める", "Q35_8", AGREE),
    Item("政策", "治安のための権利制約は当然", "Q35_9", AGREE),
    Item("政策", "同性婚を認めるべき", "Q35_10", AGREE),
    Item("政策", "選択的夫婦別姓を認めるべき", "Q35_11", AGREE),
    Item("政策", "消費税率を引き下げるべき", "Q35_12", AGREE),
    Item("政策", "政治は腐敗、強い指導者が必要", "Q35_13", AGREE),
    Item("政策", "中国は脅威", "Q36_1", AGREE),
    Item("政策", "格差より経済競争力", "Q36_2", AGREE),
    Item("政策", "原発をすぐ廃止", "Q36_3", AGREE),
    Item("政策", "気候変動のため暮らしを見直す", "Q36_4", AGREE),
    Item("政策", "身を切る改革は定数削減を優先", "Q36_5", AGREE),
    Item("政策", "在留外国人で社会が豊かになる", "Q36_6", AGREE),
] + [Item("優先課題", name, "Q37", frozenset({code}), _r(1, 13)) for code, name in PRIORITY_Q37.items()]


def indicator(s: pd.Series, item: Item) -> pd.Series:
    """該当なら1、非該当なら0、分母に入らない回答はNaN。"""
    valid = s.isin(item.valid) if item.valid is not None else s.notna() & (s != NONRESPONSE)
    return s.isin(item.yes).astype(float).where(valid)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """割合の差 p1-p2 と、Newcombe の方法10(Wilson区間の合成)による95%区間。"""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def age_group(q40: pd.Series) -> pd.Series:
    """年齢を10歳刻み(18〜29, 30代, …, 70歳以上)にまとめる。調整用。"""
    bins = {**{c: "18-29" for c in (1, 2, 3)}, 4: "30s", 5: "30s", 6: "40s", 7: "40s", 8: "50s", 9: "50s",
            10: "60s", 11: "60s", **{c: "70+" for c in range(12, 16)}}
    return q40.map(bins)


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Q4"].isin(VOTED)].copy()
    df["sanseito"] = (df["Q4"] == SANSEITO).astype(int)
    df["age_g"] = age_group(df["Q40"])
    df["male"] = df["Q39"].map({1: 1, 2: 0})
    return df


def adjusted_or(y: pd.Series, df: pd.DataFrame) -> tuple[float, float, float, float]:
    """年齢・性別で調整した、参政党投票者であることのオッズ比(95%区間とp値)。"""
    import statsmodels.formula.api as smf

    d = pd.DataFrame({"y": y, "sanseito": df["sanseito"], "age_g": df["age_g"], "male": df["male"]}).dropna()
    # 該当者が参政党側で0人・全員などのとき、最尤推定が発散して係数が意味をなさない
    if min(d.loc[d.sanseito == 1, "y"].sum(), (1 - d.loc[d.sanseito == 1, "y"]).sum()) < 3:
        return (math.nan,) * 4
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = smf.logit("y ~ sanseito + C(age_g) + male", data=d).fit(disp=0)
        except Exception:
            return (math.nan,) * 4
    b, se = fit.params["sanseito"], fit.bse["sanseito"]
    if not fit.mle_retvals.get("converged", False) or not np.isfinite(se) or abs(b) > 10:
        return (math.nan,) * 4
    return math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se), float(fit.pvalues["sanseito"])


def profile_table(df: pd.DataFrame, items: list[Item] = ITEMS, adjust: bool = True) -> pd.DataFrame:
    rows = []
    s = df["sanseito"] == 1
    for it in items:
        y = indicator(df[it.col], it)
        a, b = y[s].dropna(), y[~s].dropna()
        k1, n1, k2, n2 = int(a.sum()), len(a), int(b.sum()), len(b)
        d, lo, hi = newcombe_diff(k1, n1, k2, n2)
        p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
        row = dict(group=it.group, label=it.label, n_s=n1, p_s=k1 / n1, n_o=n2, p_o=k2 / n2,
                   diff=d, diff_lo=lo, diff_hi=hi, p=p)
        if adjust:
            row.update(zip(("or_adj", "or_lo", "or_hi", "p_adj"), adjusted_or(y, df)))
        rows.append(row)
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    if adjust:
        t["p_adj_holm"] = holm(t["p_adj"].to_numpy())
    return t


def by_party(df: pd.DataFrame, items: list[Item], parties: list[int], min_n: int = 30) -> pd.DataFrame:
    """主要政党ごとの割合(Wilson区間つき)。人数が min_n 未満の党は除く。"""
    rows = []
    for code in parties:
        sub = df[df["Q4"] == code]
        if len(sub) < min_n:
            continue
        for it in items:
            y = indicator(sub[it.col], it).dropna()
            k, n = int(y.sum()), len(y)
            lo, hi = wilson(k, n)
            rows.append(dict(party=PARTY_Q4[code], label=it.label, n=n, p=k / n, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    print(f"比例で投票した回答者 {len(df)} 人、うち参政党 {int(df.sanseito.sum())} 人 "
          f"({df.sanseito.mean():.1%})")
    t = profile_table(df)
    t.to_csv(OUT / "utas2026_profile.csv", index=False)
    bp = by_party(df, ITEMS, [1, 2, 3, 4, 5, 6, 7, 8])
    bp.to_csv(OUT / "utas2026_by_party.csv", index=False)
    pd.set_option("display.width", 250)
    show = t.assign(**{c: (t[c] * 100).round(1) for c in ("p_s", "p_o", "diff", "diff_lo", "diff_hi")})
    print(show[["group", "label", "n_s", "p_s", "p_o", "diff", "diff_lo", "diff_hi", "p_holm",
                "or_adj", "or_lo", "or_hi", "p_adj_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
