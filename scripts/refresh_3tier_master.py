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

MU_BEAR_PE_ANCHOR = 5.25
LEGACY_MASTER = "/workspace/us_tech_3tier_master_20260818.csv"

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
    Config("INTC", "英特尔", "USD", "street", beat_cap=15.0),
    Config("000660.KS", "海力士", "KRW", "storage"),
    Config("005930.KS", "三星", "KRW", "storage"),
    Config("AMZN", "亚马逊", "USD", "street"),
    Config("MSFT", "微软", "USD", "street"),
]


def fiscal_weight(ticker: str, today: dt.date | None = None) -> float:
    today = today or dt.date.today()
    end_month = FISCAL_END_MONTH[ticker]
    # months since last fiscal year end
    if today.month > end_month:
        fy_end = dt.date(today.year, end_month, 28)
    else:
        fy_end = dt.date(today.year - 1, end_month, 28)
    days_in_fy = 365
    elapsed = (today - fy_end).days
    w = max(0.0, min(1.0, elapsed / days_in_fy))
    return round(w, 2)


LEGACY_BEAT_OVERRIDE = {"AAOI": 14.65}


def calc_beat(ticker: str, cap: float | None = 15.0) -> float:
    if ticker in LEGACY_BEAT_OVERRIDE:
        return LEGACY_BEAT_OVERRIDE[ticker]
    ed = yf.Ticker(ticker).earnings_dates
    if ed is None or ed.empty:
        return 0.0
    df = ed.dropna(subset=["Surprise(%)"])
    surprises = df["Surprise(%)"].values
    # Exclude the latest print when it is an extreme one-off beat (>50%)
    if len(surprises) >= 5 and surprises[0] > 50:
        surprises = surprises[1:5]
    else:
        surprises = surprises[:4]
    if cap is not None:
        surprises = np.clip(surprises, None, cap)
    return float(np.mean(surprises))


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


def dynamic_pe_bands(ticker: str, w: float, lookback_days: int = 365) -> tuple[float, float, float, float, str]:
    """Reuse calibrated PE bands; refresh dynPE_now only."""
    pe_bear, pe_base, pe_bull, pe_src = load_legacy_pe(ticker)
    tk = yf.Ticker(ticker)
    trend = tk.get_eps_trend()
    eps0 = float(trend.loc["0y", "current"])
    eps1 = float(trend.loc["+1y", "current"])
    ntm = (1 - w) * eps0 + w * eps1
    info = tk.info or {}
    hist = tk.history(period="5d")
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1])
    dyn_now = px / ntm if ntm else 0.0
    return pe_bear, pe_base, pe_bull, float(dyn_now), pe_src


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
        c = float(trend.loc["0y", "current"])
        eps_src = "FY27=0y(已报FY26)"
    else:
        c = float(trend.loc["+1y", "current"])
        eps_src = "+1y/FY2"

    e_bear, e_base, e_bull = eps_tiers(c, beat)
    pe_bear, pe_base, pe_bull = street_wing_pe(street_tgt, c)
    tp_bear = e_bear * pe_bear
    tp_base = e_base * pe_base
    tp_bull = e_bull * pe_bull

    pe_bear_r, pe_base_r, pe_bull_r = round(pe_bear, 1), round(pe_base, 1), round(pe_bull, 1)
    pe_tiers = f"熊{pe_bear_r}/中{pe_base_r}/牛{pe_bull_r}×"
    pe_source = f"street-wing mid=tgt/C; {pe_tiers}"

    note = "全量刷新"
    if cfg.ticker == "NVDA":
        note = f"Q2FY27已报; {pe_tiers}; fwdPE≈{round(fwd_pe, 1)}×"
    elif cfg.ticker == "MRVL":
        note = f"MRVL待财报(~8/27); {pe_tiers}"
    elif cfg.ticker == "AVGO":
        note = f"下刊~9/2; {pe_tiers}"

    return row_dict(cfg, px, street_tgt, c, eps_src, beat, e_bear, e_base, e_bull,
                    pe_bear_r, pe_base_r, pe_bull_r, pe_tiers, pe_source, None, None,
                    fwd_pe, tp_bear, tp_base, tp_bull, note)


def build_storage_row(cfg: Config) -> dict:
    tk = yf.Ticker(cfg.ticker)
    info = tk.info or {}
    trend = tk.get_eps_trend()
    px = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    street_tgt = float(info.get("targetMeanPrice") or 0)
    fwd_pe = float(info.get("forwardPE") or 0)
    w = fiscal_weight(cfg.ticker)
    eps0 = float(trend.loc["0y", "current"])
    eps1 = float(trend.loc["+1y", "current"])
    c = ntm_eps(eps0, eps1, w)
    beat = calc_beat(cfg.ticker, cap=None if cfg.ticker != "INTC" else 15.0)
    e_bear, e_base, e_bull = eps_tiers(c, beat)
    pe_bear, pe_base, pe_bull, dyn_now, pe_src = dynamic_pe_bands(cfg.ticker, w)
    if cfg.ticker == "MU":
        pe_bear = MU_BEAR_PE_ANCHOR
    pe_bear_r, pe_base_r, pe_bull_r = round(pe_bear, 1), round(pe_base, 1), round(pe_bull, 1)
    pe_tiers = f"熊{pe_bear_r}/中{pe_base_r}/牛{pe_bull_r}×"
    tp_bear = e_bear * pe_bear_r
    tp_base = e_base * pe_base_r
    tp_bull = e_bull * pe_bull_r
    vjn = fetch_vjn_forward_pe(cfg.ticker)
    note = f"全量刷新; {pe_tiers}"
    if cfg.ticker == "MU":
        note = f"下刊~9/23; 熊PE锚定5.25; {pe_tiers}"
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
        "beat_b_pct": round(beat, 2),
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
        else:
            rows.append(build_street_row(cfg))
    rows.insert(11, build_nbis_row())  # after COHR

    fieldnames = [
        "ticker", "name", "currency", "px", "street_tgt", "C", "eps_src", "beat_b_pct",
        "e_bear", "e_base", "e_bull", "pe_bear", "pe_base", "pe_bull", "pe_tiers", "pe_source",
        "dynPE_now", "vjn_forward_pe", "yahoo_fwdPE", "tp_bear", "tp_base", "tp_bull",
        "up_bear_pct", "up_base_pct", "up_bull_pct", "note", "asof",
    ]

    dated = f"/workspace/us_tech_3tier_master_{ASOF.replace('-', '')}.csv"
    latest = "/workspace/us_tech_3tier_master_latest.csv"
    for path in (dated, latest):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")

    # Print NVDA summary
    nvda = next(r for r in rows if r["ticker"] == "NVDA")
    print("\n=== NVDA ===")
    for k in ["px", "street_tgt", "C", "beat_b_pct", "tp_bear", "tp_base", "tp_bull",
              "up_bear_pct", "up_base_pct", "up_bull_pct"]:
        print(f"  {k}: {nvda[k]}")


if __name__ == "__main__":
    main()
