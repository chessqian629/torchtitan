#!/usr/bin/env python3
"""Refresh US/KR tech 3-tier valuation master table."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

ASOF = dt.date.today().isoformat()

# fiscal year end month (1-12) for NTM weight
FISCAL_END_MONTH = {
    "MU": 8,
    "SNDK": 8,
    "WDC": 8,
    "000660.KS": 12,
    "005930.KS": 12,
}

LEGACY_MASTER = "/workspace/us_tech_3tier_master_20260818.csv"
# Recent traded forward-PE lookback for bear/base/bull bands
PE_LOOKBACK_DAYS = 183  # ~6 months

NBIS_ROW = {
    "ticker": "NBIS",
    "name": "Nebius",
    "currency": "USD",
    "note": "Nebius财报解读终稿; 跳过PE",
    "arr_bear": 14.0,
    "arr_base": 17.0,
    "arr_bull": 22.0,
    "ev_arr_bear": 3.0,
    "ev_arr_base": 5.0,
    "ev_arr_bull": 7.0,
}

# INTC: FY2 analyst dispersion + inverted PE (no beat rate)
INTC_MODEL = {
    "e_bear": 1.27,
    "e_base": 1.79,
    "e_bull": 2.84,
    "pe_bear": 55.0,
    "pe_base": 50.0,
    "pe_bull": 45.0,
    "fy27_rev_b": 71.3,
    "net_debt_b": 20.8,
}


@dataclass
class Config:
    ticker: str
    name: str
    currency: str
    kind: str  # street | storage | story0y | nbis
    eps_col: str = "+1y"
    beat_cap: float = 15.0
    intc_beat_cap: bool = False


CONFIGS = [
    Config("MU", "美光", "USD", "storage"),
    Config("SNDK", "闪迪", "USD", "storage"),
    Config("WDC", "西数", "USD", "storage"),
    Config("AAOI", "AAOI", "USD", "street"),
    Config("NOK", "诺基亚", "USD", "street"),
    Config("NVDA", "英伟达", "USD", "street"),
    Config("LITE", "Lumentum", "USD", "story0y", "0y"),
    Config("AVGO", "博通", "USD", "street"),
    Config("MRVL", "Marvell", "USD", "street"),
    Config("AMD", "AMD", "USD", "street"),
    Config("COHR", "Coherent", "USD", "story0y", "0y"),
    Config("INTC", "英特尔", "USD", "intc"),
    Config("000660.KS", "海力士", "KRW", "storage"),
    Config("SKHY", "海力士ADR", "USD", "skhy"),
    Config("005930.KS", "三星", "KRW", "storage"),
    Config("AMZN", "亚马逊", "USD", "street"),
    Config("MSFT", "微软", "USD", "street"),
]


def fiscal_weight(
    ticker: str,
    today: dt.date | None = None,
    eps0: float | None = None,
    eps1: float | None = None,
) -> float:
    """Weight on +1y in NTM = (1-w)*0y + w*+1y.

    Yahoo often keeps 0y/+1y labels until the Q4 print (~30-45d after
    fiscal year-end). During that lag (+1y/0y still very steep), clamp w
    near year-end. Once labels roll (0y jumps to the new FY), use normal
    early-year weight so NTM stays on the new 0y.
    """
    today = today or dt.date.today()
    end_month = FISCAL_END_MONTH[ticker]
    if today.month > end_month or (today.month == end_month and today.day >= 28):
        fy_end = dt.date(today.year, end_month, 28)
    else:
        fy_end = dt.date(today.year - 1, end_month, 28)
    elapsed = (today - fy_end).days
    labels_rolled = (
        eps0 is not None
        and eps1 is not None
        and eps0 > 0
        and (eps1 / eps0) < 1.5
    )
    # Keep forward-year weight until Yahoo rolls 0y (can lag 60-90d after FY-end).
    if elapsed >= 0 and not labels_rolled and elapsed <= 90:
        return 0.97
    w = max(0.0, min(1.0, elapsed / 365.0))
    return round(w, 2)


LEGACY_BEAT_OVERRIDE = {"AAOI": 14.65}


def calc_beat(ticker: str, cap: float | None = 15.0) -> float:
    if ticker in LEGACY_BEAT_OVERRIDE:
        return LEGACY_BEAT_OVERRIDE[ticker]
    try:
        ed = yf.Ticker(ticker).earnings_dates
    except Exception:
        ed = None
    if ed is not None and not ed.empty and "Surprise(%)" in ed.columns:
        df = ed.dropna(subset=["Surprise(%)"])
        if not df.empty:
            surprises = df["Surprise(%)"].values
            # Exclude the latest print when it is an extreme one-off beat (>50%)
            if len(surprises) >= 5 and surprises[0] > 50:
                surprises = surprises[1:5]
            else:
                surprises = surprises[:4]
            if cap is not None:
                surprises = np.clip(surprises, None, cap)
            return float(np.mean(surprises))
    # Fallback: reuse last master-table beat if Yahoo scrape fails
    try:
        with open("/workspace/us_tech_3tier_master_latest.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["ticker"] == ticker and row.get("beat_b_pct") not in ("", None):
                    return float(row["beat_b_pct"])
    except Exception:
        pass
    return 0.0


def load_legacy_pe(ticker: str) -> tuple[float, float, float, str]:
    with open(LEGACY_MASTER, newline="") as f:
        for row in csv.DictReader(f):
            if row["ticker"] == ticker:
                return (
                    float(row["pe_bear"]),
                    float(row["pe_base"]),
                    float(row["pe_bull"]),
                    row["pe_source"],
                )
    raise KeyError(ticker)


def eps_tiers(c: float, beat_pct: float) -> tuple[float, float, float]:
    b = beat_pct / 100.0
    return c, c * (1 + 0.5 * b), c * (1 + b)


def street_wing_pe(street_tgt: float, c: float) -> tuple[float, float, float]:
    mid = street_tgt / c
    return mid * 0.75, mid, mid * 1.125


def ntm_eps(eps0: float, eps1: float, w: float) -> float:
    return (1 - w) * eps0 + w * eps1


def fetch_vjn_forward_pe(ticker: str) -> Optional[float]:
    sym = ticker.replace(".KS", "")
    url = f"https://api.vjn.ai/public/stock/{sym}/valuation"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return float(data.get("forwardPE") or data.get("forward_pe") or 0) or None
    except Exception:
        return None


def _eps_at_lag(trend, period: str, days_ago: int) -> float:
    row = trend.loc[period]
    cur = float(row["current"])
    def g(col: str) -> float:
        v = row.get(col)
        try:
            return float(v) if v not in (None, 0, 0.0) else cur
        except Exception:
            return cur

    if days_ago >= 82:
        return g("90daysAgo")
    if days_ago >= 45:
        return g("60daysAgo")
    if days_ago >= 20:
        return g("30daysAgo")
    if days_ago >= 5:
        return g("7daysAgo")
    return cur


def finalize_pe_bands(pes: list[float], spot_pe: float) -> tuple[float, float, float, str]:
    """Map recent PE sample → bear/base/bull with downside vs spot when compressed."""
    arr = np.array(pes, dtype=float)
    p10, p25, p50, p75, p90 = [float(x) for x in np.percentile(arr, [10, 25, 50, 75, 90])]
    mid = p50
    bull = max(p75, mid * 1.12)
    bear = p25
    # Spot already below recent p25 → force 熊PE under spot so downside is real
    if spot_pe > 0 and spot_pe < p25:
        bear = min(p10, spot_pe * 0.88)
        if bear >= spot_pe * 0.98:
            bear = spot_pe * 0.85
    elif spot_pe > 0 and bear > spot_pe * 0.95:
        bear = min(bear, spot_pe * 0.90)
    if bear >= mid * 0.98:
        bear = mid * 0.75
    if bull <= mid * 1.02:
        bull = mid * 1.15
    # Keep ordering bear < mid < bull
    if bear >= mid:
        bear = mid * 0.75
    if bull <= mid:
        bull = mid * 1.15
    src = (
        f"近{PE_LOOKBACK_DAYS}d交易前瞻PE "
        f"p25/50/75={p25:.2f}/{p50:.2f}/{p75:.2f}; now={spot_pe:.2f}"
    )
    return round(bear, 1), round(mid, 1), round(bull, 1), src


def recent_pe_bands_simple(
    ticker: str, period: str, lookback_days: int = PE_LOOKBACK_DAYS
) -> tuple[float, float, float, float, str]:
    """PE bands from px / Yahoo period EPS (0y or +1y), lag-adjusted."""
    tk = yf.Ticker(ticker)
    trend = tk.get_eps_trend()
    info = tk.info or {}
    hist = tk.history(period="1y")
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1])
    c_now = float(trend.loc[period, "current"])
    spot_pe = px / c_now if c_now else 0.0
    end = hist.index[-1].date()
    pes: list[float] = []
    for idx, row in hist.iterrows():
        dago = (end - idx.date()).days
        if dago > lookback_days:
            continue
        e = _eps_at_lag(trend, period, dago)
        if e > 0:
            pes.append(float(row["Close"]) / e)
    if len(pes) < 20:
        # fallback to street-wing if history thin
        street = float(info.get("targetMeanPrice") or 0)
        if street and c_now:
            b, m, u = street_wing_pe(street, c_now)
            return round(b, 1), round(m, 1), round(u, 1), spot_pe, "fallback street-wing (thin PE hist)"
        return 15.0, 20.0, 25.0, spot_pe, "fallback flat PE"
    b, m, u, src = finalize_pe_bands(pes, spot_pe)
    return b, m, u, spot_pe, src


def dynamic_pe_bands(
    ticker: str, w: float, lookback_days: int = PE_LOOKBACK_DAYS
) -> tuple[float, float, float, float, str]:
    """Storage: lag-adjusted NTM PE over ~6M → p25/50/75 bands."""
    tk = yf.Ticker(ticker)
    trend = tk.get_eps_trend()
    info = tk.info or {}
    hist = tk.history(period="1y")
    eps0 = float(trend.loc["0y", "current"])
    eps1 = float(trend.loc["+1y", "current"])
    ntm = (1 - w) * eps0 + w * eps1
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1])
    spot_pe = px / ntm if ntm else 0.0
    end = hist.index[-1].date()
    pes: list[float] = []
    for idx, row in hist.iterrows():
        d = idx.date()
        dago = (end - d).days
        if dago > lookback_days:
            continue
        e0 = _eps_at_lag(trend, "0y", dago)
        e1 = _eps_at_lag(trend, "+1y", dago)
        ww = fiscal_weight(ticker, today=d, eps0=e0, eps1=e1)
        n = (1 - ww) * e0 + ww * e1
        if n > 0:
            pes.append(float(row["Close"]) / n)
    if len(pes) < 20:
        try:
            return (*load_legacy_pe(ticker)[:3], spot_pe, "legacy PE (thin hist)")
        except Exception:
            return 5.0, 8.0, 12.0, spot_pe, "fallback flat PE"
    b, m, u, src = finalize_pe_bands(pes, spot_pe)
    return b, m, u, spot_pe, src


def build_skhy_row(cfg: Config) -> dict:
    """SKHY Nasdaq ADR: 10 ADR = 1 ORD. Recent +1y traded PE bands."""
    tk = yf.Ticker("SKHY")
    info = tk.info or {}
    trend = tk.get_eps_trend()
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    street_tgt = float(info.get("targetMeanPrice") or 0)
    fwd_pe = float(info.get("forwardPE") or 0)
    c = float(trend.loc["+1y", "current"])
    # Beat from KR ordinary (ADR surprise feed is noisy); same skip-extreme rule
    beat = calc_beat("000660.KS", cap=None)
    e_bear, e_base, e_bull = eps_tiers(c, beat)
    pe_bear_r, pe_base_r, pe_bull_r, spot_pe, pe_src = recent_pe_bands_simple("SKHY", "+1y")
    pe_tiers = f"熊{pe_bear_r}/中{pe_base_r}/牛{pe_bull_r}×"
    tp_bear = e_bear * pe_bear_r
    tp_base = e_base * pe_base_r
    tp_bull = e_bull * pe_bull_r

    # Parity vs KR ordinary for note
    try:
        krw_px = float((yf.Ticker("000660.KS").info or {}).get("currentPrice") or 0)
        fx = float((yf.Ticker("USDKRW=X").info or {}).get("regularMarketPrice") or 0)
        parity = (krw_px / fx) / 10.0 if fx else 0.0
        prem = (px / parity - 1) * 100 if parity else 0.0
        parity_note = f"平价${parity:.0f}/溢价{prem:+.0f}%"
    except Exception:
        parity_note = "10ADR=1ORD"

    note = (
        f"Nasdaq ADR 10:1; 近6M +1y交易PE; "
        f"beat取自000660.KS; {parity_note}; {pe_tiers}"
    )
    pe_source = f"{pe_src}; {pe_tiers}"
    return row_dict(
        cfg, px, street_tgt, c, "SKHY +1y/FY2 USD", beat,
        e_bear, e_base, e_bull, pe_bear_r, pe_base_r, pe_bull_r,
        pe_tiers, pe_source, spot_pe, None, fwd_pe,
        tp_bear, tp_base, tp_bull, note,
    )


def build_intc_row(cfg: Config) -> dict:
    """INTC custom: FY27 EPS dispersion × inverted PE tiers."""
    tk = yf.Ticker("INTC")
    info = tk.info or {}
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    street_tgt = float(info.get("targetMeanPrice") or 0)
    fwd_pe = float(info.get("forwardPE") or 0)
    m = INTC_MODEL
    e_bear, e_base, e_bull = m["e_bear"], m["e_base"], m["e_bull"]
    pe_bear_r, pe_base_r, pe_bull_r = m["pe_bear"], m["pe_base"], m["pe_bull"]
    tp_bear = pe_bear_r * e_bear
    tp_base = pe_base_r * e_base
    tp_bull = pe_bull_r * e_bull
    pe_tiers = f"熊{pe_bear_r:.0f}/中{pe_base_r:.0f}/牛{pe_bull_r:.0f}×(倒挂)"
    pe_source = (
        "FY27离散度×PE倒挂(熊高倍/牛低倍); "
        f"EV/Sales≈6.2/8.0/11.8×(@FY27 Rev ${m['fy27_rev_b']}B)"
    )
    eps_src = "FY2离散度(分析师low/avg/high): $1.27/$1.79/$2.84; 不用beat"
    note = (
        f"{pe_tiers}; CapEx>$20B/2027更高; "
        f"净负债${m['net_debt_b']}B; SCIP回购$14.2B; 以色列KG厂地缘风险"
    )
    return row_dict(
        cfg, px, street_tgt, e_base, eps_src, "", e_bear, e_base, e_bull,
        pe_bear_r, pe_base_r, pe_bull_r, pe_tiers, pe_source, None, None,
        fwd_pe, tp_bear, tp_base, tp_bull, note,
    )


def build_street_row(cfg: Config) -> dict:
    tk = yf.Ticker(cfg.ticker)
    info = tk.info or {}
    trend = tk.get_eps_trend()
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    street_tgt = float(info.get("targetMeanPrice") or 0)
    fwd_pe = float(info.get("forwardPE") or 0)

    cap = 15.0 if cfg.ticker == "INTC" else None
    beat = calc_beat(cfg.ticker, cap=cap)

    if cfg.kind == "story0y":
        period = "0y"
        c = float(trend.loc["0y", "current"])
        eps_src = "FY27=0y(已报FY26)"
    else:
        period = "+1y"
        c = float(trend.loc["+1y", "current"])
        eps_src = "+1y/FY2"

    e_bear, e_base, e_bull = eps_tiers(c, beat)
    pe_bear_r, pe_base_r, pe_bull_r, spot_pe, pe_src = recent_pe_bands_simple(
        cfg.ticker, period
    )
    tp_bear = e_bear * pe_bear_r
    tp_base = e_base * pe_base_r
    tp_bull = e_bull * pe_bull_r
    pe_tiers = f"熊{pe_bear_r}/中{pe_base_r}/牛{pe_bull_r}×"
    pe_source = f"{pe_src}; {pe_tiers}"

    note = f"近6M交易前瞻PE重标定; {pe_tiers}"
    if cfg.ticker == "NVDA":
        note = f"Q2FY27已报; {pe_tiers}; fwdPE≈{round(fwd_pe, 1)}×"
    elif cfg.ticker == "MRVL":
        note = (
            f"Q2FY27已报: Rev$2.74B(+37%YoY) NG EPS$0.94; "
            f"Q3指引Rev$3.15B±5% EPS$1.10±0.05; 上调FY27/28收入展望; {pe_tiers}"
        )
    elif cfg.ticker == "AVGO":
        note = (
            f"Q3FY26已报(9/2): NG EPS$3.32 vs $3.24; "
            f"C用+1y; {pe_tiers}"
        )

    return row_dict(cfg, px, street_tgt, c, eps_src, beat, e_bear, e_base, e_bull,
                    pe_bear_r, pe_base_r, pe_bull_r, pe_tiers, pe_source, spot_pe, None,
                    fwd_pe, tp_bear, tp_base, tp_bull, note)


def build_storage_row(cfg: Config) -> dict:
    tk = yf.Ticker(cfg.ticker)
    info = tk.info or {}
    trend = tk.get_eps_trend()
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    street_tgt = float(info.get("targetMeanPrice") or 0)
    fwd_pe = float(info.get("forwardPE") or 0)
    eps0 = float(trend.loc["0y", "current"])
    eps1 = float(trend.loc["+1y", "current"])
    w = fiscal_weight(cfg.ticker, eps0=eps0, eps1=eps1)
    c = ntm_eps(eps0, eps1, w)
    beat = calc_beat(cfg.ticker, cap=None if cfg.ticker != "INTC" else 15.0)
    e_bear, e_base, e_bull = eps_tiers(c, beat)
    pe_bear, pe_base, pe_bull, dyn_now, pe_src = dynamic_pe_bands(cfg.ticker, w)
    pe_bear_r, pe_base_r, pe_bull_r = round(pe_bear, 1), round(pe_base, 1), round(pe_bull, 1)
    pe_tiers = f"熊{pe_bear_r}/中{pe_base_r}/牛{pe_bull_r}×"
    tp_bear = e_bear * pe_bear_r
    tp_base = e_base * pe_base_r
    tp_bull = e_bull * pe_bull_r
    vjn = fetch_vjn_forward_pe(cfg.ticker)
    note = f"近6M NTM交易PE重标定; {pe_tiers}"
    if cfg.ticker == "MU":
        note = (
            f"Q4FY26已报: NG EPS$33.42 vs~$31.8; "
            f"Q1FY27指引Rev$61.5B±1.5B EPS$38.15±1; "
            f"FY27=0y已滚动; {pe_tiers}"
        )
    return row_dict(cfg, px, street_tgt, c, f"动态NTM w={w}", beat, e_bear, e_base, e_bull,
                    pe_bear_r, pe_base_r, pe_bull_r, pe_tiers, pe_src, dyn_now, vjn, fwd_pe,
                    tp_bear, tp_base, tp_bull, note)


def build_nbis_row() -> dict:
    tk = yf.Ticker("NBIS")
    info = tk.info or {}
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    street_tgt = float(info.get("targetMeanPrice") or 0)
    n = NBIS_ROW
    tp_bear = n["arr_bear"] * n["ev_arr_bear"] * 10  # placeholder wrong

    # EV/ARR model: target = EV/ARR × ARR (in $B) × shares factor
    # From prior table: ARR $14/17/22B × EV/ARR 3/5/7 → $140/$291/$533
    tp_bear, tp_base, tp_bull = 140.0, 291.0, 533.0
    up = lambda t: (t / px - 1) * 100 if px else 0
    return {
        "ticker": n["ticker"],
        "name": n["name"],
        "currency": n["currency"],
        "px": px,
        "street_tgt": street_tgt,
        "C": "",
        "eps_src": "YE27 ARR $B (非EPS)",
        "beat_b_pct": "",
        "e_bear": n["arr_bear"],
        "e_base": n["arr_base"],
        "e_bull": n["arr_bull"],
        "pe_bear": n["ev_arr_bear"],
        "pe_base": n["ev_arr_base"],
        "pe_bull": n["ev_arr_bull"],
        "pe_tiers": f"熊{n['ev_arr_bear']}/中{n['ev_arr_base']}/牛{n['ev_arr_bull']}×(EV/ARR)",
        "pe_source": "bc-01a016f0终稿 EV/ARR 3/5/7 × ARR $14/17/22B → $140/$291/$533",
        "dynPE_now": "",
        "vjn_forward_pe": "",
        "yahoo_fwdPE": "",
        "tp_bear": tp_bear,
        "tp_base": tp_base,
        "tp_bull": tp_bull,
        "up_bear_pct": round(up(tp_bear), 2),
        "up_base_pct": round(up(tp_base), 2),
        "up_bull_pct": round(up(tp_bull), 2),
        "note": n["note"],
        "asof": ASOF,
    }


def row_dict(cfg, px, street_tgt, c, eps_src, beat, e_bear, e_base, e_bull,
             pe_bear, pe_base, pe_bull, pe_tiers, pe_source, dyn_now, vjn, fwd_pe,
             tp_bear, tp_base, tp_bull, note) -> dict:
    up = lambda t: (t / px - 1) * 100 if px else 0
    return {
        "ticker": cfg.ticker,
        "name": cfg.name,
        "currency": cfg.currency,
        "px": px,
        "street_tgt": street_tgt,
        "C": c,
        "eps_src": eps_src,
        "beat_b_pct": round(beat, 2) if beat != "" else "",
        "e_bear": e_bear,
        "e_base": e_base,
        "e_bull": e_bull,
        "pe_bear": pe_bear,
        "pe_base": pe_base,
        "pe_bull": pe_bull,
        "pe_tiers": pe_tiers,
        "pe_source": pe_source,
        "dynPE_now": dyn_now if dyn_now is not None else "",
        "vjn_forward_pe": vjn if vjn is not None else "",
        "yahoo_fwdPE": fwd_pe,
        "tp_bear": tp_bear,
        "tp_base": tp_base,
        "tp_bull": tp_bull,
        "up_bear_pct": round(up(tp_bear), 2),
        "up_base_pct": round(up(tp_base), 2),
        "up_bull_pct": round(up(tp_bull), 2),
        "note": note,
        "asof": ASOF,
    }


def main():
    rows = []
    for cfg in CONFIGS:
        print(f"Processing {cfg.ticker}...")
        if cfg.kind == "storage":
            rows.append(build_storage_row(cfg))
        elif cfg.kind == "intc":
            rows.append(build_intc_row(cfg))
        elif cfg.kind == "skhy":
            rows.append(build_skhy_row(cfg))
        else:
            rows.append(build_street_row(cfg))
    rows.insert(11, build_nbis_row())  # after COHR

    fieldnames = [
        "ticker", "name", "currency", "px", "street_tgt", "C", "eps_src", "beat_b_pct",
        "e_bear", "e_base", "e_bull",
        "dynPE_now", "vjn_forward_pe", "yahoo_fwdPE", "tp_bear", "tp_base", "tp_bull",
        "up_bear_pct", "up_base_pct", "up_bull_pct", "note",
        "pe_bear", "pe_base", "pe_bull", "pe_tiers", "pe_source", "asof",
    ]

    dated = f"/workspace/us_tech_3tier_master_{ASOF.replace('-', '')}.csv"
    latest = "/workspace/us_tech_3tier_master_latest.csv"
    for path in (dated, latest):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")

    for sym in ("NVDA", "SKHY", "000660.KS"):
        row = next(r for r in rows if r["ticker"] == sym)
        print(f"\n=== {sym} ===")
        for k in ["px", "street_tgt", "C", "beat_b_pct", "pe_tiers",
                  "tp_bear", "tp_base", "tp_bull",
                  "up_bear_pct", "up_base_pct", "up_bull_pct", "note"]:
            print(f"  {k}: {row[k]}")


if __name__ == "__main__":
    main()
