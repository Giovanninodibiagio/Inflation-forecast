"""
Eurozone Inflation API — FastAPI server

Wraps EurozoneInflationAnalyzer and exposes JSON endpoints consumed by
the React frontend.  Run with:

    uvicorn api:app --reload --port 8000

Data sources (priority order, no API key required for the first two):

  1. ECB Statistical Data Warehouse (SDW) — primary source for all historical data.
     Free, official, no registration.  Fetches 5 series via CSV:
       000000 = headline HICP       SERV00 = services
       IGXE00 = non-energy goods    FOOD00 = food/alcohol/tobacco
       NRGY00 = energy
     Endpoint: https://data-api.ecb.europa.eu/service/data/ICP/M.U2.N.{code}.4.ANR

  2. FRED (optional fallback) — set FRED_API_KEY env var.

  3. Hardcoded simulated data — last resort if all APIs are unreachable.

Endpoints:
    GET /api/health               server status + feature flags
    GET /api/history              last 36 months of HICP component data (4 components)
    GET /api/subindices           11 HICP COICOP sub-category time series
    GET /api/leading-indicators   PPI, Import Prices, Labor Costs (ECB SDW)
    GET /api/forecast             probabilistic fan chart (query: periods, methodology)
    GET /api/commodities          commodity signals panel (Brent, EU gas, EUR/USD)
    GET /api/trueflation          US real-time inflation as EA leading signal (optional)
"""

import io
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from eurozone_inflation_analyzer import EurozoneInflationAnalyzer, LeadingIndicatorFetcher

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Eurozone Inflation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)

FRED_API_KEY: Optional[str] = os.getenv("FRED_API_KEY")
TRUFLATION_API_KEY: Optional[str] = os.getenv("TRUFLATION_API_KEY")

_leading_fetcher: Optional[LeadingIndicatorFetcher] = None


def _get_leading_fetcher() -> LeadingIndicatorFetcher:
    global _leading_fetcher
    if _leading_fetcher is None:
        _leading_fetcher = LeadingIndicatorFetcher()
    return _leading_fetcher

# ---------------------------------------------------------------------------
# Simulated historical data (mirrors React's hardcoded arrays)
# ---------------------------------------------------------------------------

# Simulated HICP dataset: Jan 2023 – Feb 2026 (38 months)
#
# 2023 (months 1-12):  post-peak disinflation, services/food elevated
# 2024 (months 13-24): continued disinflation, energy turning negative
# 2025 (months 25-36): stabilisation + energy recovery (Middle East oil shock)
#                       services sticky ~4%, energy flips from -4% to +2%
# 2026 Jan-Feb (37-38): headline re-accelerating to ~2.7-2.8%
#
# Component weights: services 45.7%, NEIG 25.6%, food 19.3%, energy 9.4%
# Each headline value is approximately consistent with the weighted-sum of components.

_SIM_BASELINE = [
    # 2023
    4.2, 4.1, 4.3, 4.2, 4.1, 4.0, 3.8, 3.5, 3.4, 3.5, 3.6, 3.4,
    # 2024
    3.2, 3.3, 3.0, 2.9, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 2.5, 2.0,
    # 2025: energy base-effect reversal lifts headline back toward 2.5-2.7%
    2.5, 2.4, 2.2, 2.3, 2.2, 2.5, 2.6, 2.4, 2.3, 2.5, 2.6, 2.7,
    # 2026 Jan-Feb: commodity pressure pushes headline to ~2.8%
    2.8, 2.7,
]
_SIM_COMPONENTS: Dict[str, List[float]] = {
    # Services: sticky wage-driven component; bounces back above 4% in 2025
    "services": [
        # 2023
        4.8, 4.9, 5.0, 4.8, 4.7, 4.6, 4.4, 4.2, 4.1, 4.0, 4.1, 4.2,
        # 2024
        4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.5, 3.4, 3.5, 3.4, 3.2,
        # 2025: sticky, re-accelerates as wage rounds settle above ECB target
        4.0, 3.9, 3.8, 3.9, 4.0, 4.1, 4.2, 4.1, 4.0, 4.1, 4.2, 4.3,
        # 2026
        4.4, 4.3,
    ],
    # Non-energy industrial goods: ongoing goods deflation
    "non_energy_goods": [
        # 2023
        2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1,
        # 2024
        1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.4,
        # 2025: stabilises at low positive level
        0.6, 0.5, 0.4, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.5, 0.6,
        # 2026
        0.6, 0.5,
    ],
    # Food, alcohol & tobacco: normalising after 2022-23 food crisis
    "food_alcohol_tobacco": [
        # 2023
        4.5, 4.6, 4.7, 4.5, 4.3, 4.1, 3.9, 3.7, 3.5, 3.4, 3.2, 3.1,
        # 2024
        3.0, 3.1, 3.2, 3.3, 3.4, 3.2, 3.1, 3.0, 2.8, 2.6, 2.5, 2.6,
        # 2025: ticks up as energy-intensive food production costs rise
        2.7, 2.6, 2.5, 2.6, 2.5, 2.7, 2.8, 2.7, 2.6, 2.7, 2.8, 2.9,
        # 2026
        3.0, 2.9,
    ],
    # Energy: the key story — reverses from -4% (Dec 2024) on Middle East oil shock
    "energy": [
        # 2023
        3.2,  2.8,  2.0,  0.5, -1.5, -2.3, -1.5, -3.2, -2.8, -1.5,  0.2, -0.5,
        # 2024: deep negative as high 2022 base fades
        -2.0, -1.5, -2.5, -3.2, -3.0, -2.0, -1.0, -0.8, -1.5, -2.0, -1.9, -4.0,
        # 2025: base effect fades, Brent rises on Middle East tensions
        -1.0, -0.5,  0.0,  0.5,  0.0,  1.5,  2.0,  1.5,  1.0,  2.0,  2.0,  2.0,
        # 2026: sustained commodity pressure
        2.5,  2.0,
    ],
}

# Default weights (overridden by Eurostat fetch at startup)
_DEFAULT_WEIGHTS = {
    "services": 0.457,
    "non_energy_goods": 0.256,
    "food_alcohol_tobacco": 0.193,
    "energy": 0.094,
}


def _month_offset(start: date, i: int) -> date:
    m = start.month - 1 + i
    return date(start.year + m // 12, m % 12 + 1, 1)


def _build_simulated_history(weights: Dict[str, float]) -> List[Dict]:
    """Return simulated HICP months in React-friendly format."""
    start = date(2023, 1, 1)
    today = date.today()
    core_denom = weights["services"] + weights["non_energy_goods"]
    months: List[Dict] = []

    for i, h in enumerate(_SIM_BASELINE):
        d = _month_offset(start, i)
        if d > today:
            break
        s   = _SIM_COMPONENTS["services"][i]
        neg = _SIM_COMPONENTS["non_energy_goods"][i]
        f   = _SIM_COMPONENTS["food_alcohol_tobacco"][i]
        e   = _SIM_COMPONENTS["energy"][i]
        core = (s * weights["services"] + neg * weights["non_energy_goods"]) / core_denom

        # Simulated sub-indices derived from parent components
        months.append({
            "date":              d.strftime("%Y-%m"),
            "headline":          round(h, 2),
            "core":              round(core, 2),
            "services":          round(s, 2),
            "nonEnergyGoods":    round(neg, 2),
            "foodAlcoholTobacco":round(f, 2),
            "energy":            round(e, 2),
            "subIndices": {
                "actualRents":       round(s * 0.85, 2),
                "electricityGas":    round(e * 0.60, 2),
                "fuelsLubricants":   round(e * 0.40, 2),
                "restaurantsHotels": round(s * 1.05, 2),
                "recreationCulture": round(s * 0.70, 2),
                "clothingFootwear":  round(neg * 0.90, 2),
                "healthcare":        round(s * 0.80, 2),
                "transportServices": round(s * 0.75, 2),
                "newVehicles":       round(neg * 0.75, 2),
                "communications":    round(neg * 0.40, 2),
                "education":         round(s * 0.95, 2),
            },
        })
    return months


def _sim_history_to_df(months: List[Dict]) -> pd.DataFrame:
    """Convert simulated history list → analyzer-compatible DataFrame."""
    df = pd.DataFrame(months)
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={
        "nonEnergyGoods":    "non_energy_goods",
        "foodAlcoholTobacco":"food_alcohol_tobacco",
        "headline":          "hicp_headline",
        "core":              "hicp_core",
    })


# ---------------------------------------------------------------------------
# Lazy analyzer singleton + simple TTL cache
# ---------------------------------------------------------------------------

_analyzer: Optional[EurozoneInflationAnalyzer] = None
_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 3600  # 1 hour


def _get_analyzer() -> EurozoneInflationAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = EurozoneInflationAnalyzer(fred_api_key=FRED_API_KEY)
    return _analyzer


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < _CACHE_TTL:
        return entry["data"]
    return None


def _cache_set(key: str, data: Any) -> None:
    _cache[key] = {"data": data, "ts": time.time()}


# ---------------------------------------------------------------------------
# ECB Statistical Data Warehouse — real HICP data, no API key required
# ---------------------------------------------------------------------------

# ECB SDW ICP_ITEM codes for the 4 HICP components + headline
_ECB_SERIES = {
    "headline":            "000000",  # All-items HICP
    "services":            "SERV00",  # Services
    "non_energy_goods":    "IGXE00",  # Non-energy industrial goods
    "food_alcohol_tobacco":"FOOD00",  # Food, alcohol & tobacco
    "energy":              "NRGY00",  # Energy
}

_ECB_BASE = "https://data-api.ecb.europa.eu/service/data/ICP/M.U2.N.{code}.4.ANR"

# HICP COICOP sub-category item codes (same ECB SDW dataset, no API key required)
_ECB_SUBINDICES = {
    "actualRents":        "CP041",   # Actual rentals for housing       ~6.5%
    "electricityGas":     "CP045",   # Electricity, gas & other fuels   ~4.5%
    "fuelsLubricants":    "CP0722",  # Fuels & lubricants (transport)   ~3.1%
    "restaurantsHotels":  "CP11",    # Restaurants & hotels             ~9.2%
    "recreationCulture":  "CP09",    # Recreation & culture             ~8.7%
    "clothingFootwear":   "CP03",    # Clothing & footwear              ~5.7%
    "healthcare":         "CP06",    # Healthcare                       ~4.4%
    "transportServices":  "CP073",   # Transport services (excl. fuels) ~3.2%
    "newVehicles":        "CP0711",  # New vehicles                     ~2.1%
    "communications":     "CP08",    # Communications                   ~2.4%
    "education":          "CP10",    # Education                        ~1.1%
}


def _fetch_ecb_subindices(start: str = "2022-01") -> Dict[str, Optional[pd.Series]]:
    """
    Fetch all HICP sub-index series from ECB SDW in parallel.
    Returns a dict mapping sub-index name → pd.Series (None on failure).
    Uses ThreadPoolExecutor to keep latency ~1-2s instead of 11s serial.
    """
    results: Dict[str, Optional[pd.Series]] = {}

    def fetch_one(name: str, code: str) -> tuple:
        return name, _fetch_ecb_sdw_series(code, start)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_one, name, code): name
                   for name, code in _ECB_SUBINDICES.items()}
        for future in as_completed(futures):
            name, series = future.result()
            results[name] = series

    return results


def _fetch_ecb_sdw_series(code: str, start: str = "2022-01") -> Optional[pd.Series]:
    """
    Fetch one monthly EA HICP series from the ECB Statistical Data Warehouse.

    Returns a pd.Series indexed by period string ("YYYY-MM") or None on failure.
    No API key or registration required.
    """
    url = f"{_ECB_BASE.format(code=code)}?format=csvdata&startPeriod={start}&detail=dataonly"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df = df.dropna()
        return df.set_index("TIME_PERIOD")["OBS_VALUE"]
    except Exception as exc:
        print(f"ECB SDW fetch failed for {code}: {exc}")
        return None


def _fetch_ecb_hicp_real(start: str = "2022-01") -> Optional[List[Dict]]:
    """
    Fetch all 5 HICP series + 11 sub-indices from the ECB SDW.

    Returns None only if the headline series itself is unavailable.
    Missing component/sub-index series are left as None (React handles gracefully).
    Sub-indices are fetched in parallel using ThreadPoolExecutor.
    """
    # Fetch main 5 series and 11 sub-indices concurrently
    series: Dict[str, Optional[pd.Series]] = {}
    sub_series: Dict[str, Optional[pd.Series]] = {}

    def fetch_main(name: str, code: str) -> tuple:
        return name, _fetch_ecb_sdw_series(code, start)

    with ThreadPoolExecutor(max_workers=12) as executor:
        main_futures = {executor.submit(fetch_main, name, code): name
                        for name, code in _ECB_SERIES.items()}
        sub_futures  = {executor.submit(fetch_main, name, code): name
                        for name, code in _ECB_SUBINDICES.items()}
        for future in as_completed({**main_futures, **sub_futures}):
            name, s = future.result()
            if name in _ECB_SERIES:
                series[name] = s
            else:
                sub_series[name] = s

    if series["headline"] is None:
        return None  # Can't build anything without the headline

    # Align all series to the headline index
    idx = series["headline"].index
    aligned: Dict[str, pd.Series] = {}
    for name, s in series.items():
        aligned[name] = s.reindex(idx) if s is not None else pd.Series(np.nan, index=idx)

    # Align sub-index series
    aligned_sub: Dict[str, pd.Series] = {}
    for name, s in sub_series.items():
        aligned_sub[name] = s.reindex(idx) if s is not None else pd.Series(np.nan, index=idx)

    # HICP weights for core calculation
    w_s  = 0.457   # services
    w_n  = 0.256   # non-energy goods
    core_denom = w_s + w_n

    months: List[Dict] = []
    for period in idx:
        h   = aligned["headline"].get(period, np.nan)
        s   = aligned["services"].get(period, np.nan)
        neg = aligned["non_energy_goods"].get(period, np.nan)
        f   = aligned["food_alcohol_tobacco"].get(period, np.nan)
        e   = aligned["energy"].get(period, np.nan)

        # Core = weighted average of services + NEIG (excl. food & energy)
        core = (s * w_s + neg * w_n) / core_denom if not (np.isnan(s) or np.isnan(neg)) else np.nan

        def _v(x: float) -> Optional[float]:
            return round(float(x), 2) if not np.isnan(x) else None

        # Build sub-indices dict (only include keys where data exists)
        sub_dict = {}
        for sub_name, sub_aligned in aligned_sub.items():
            val = sub_aligned.get(period, np.nan)
            sub_dict[sub_name] = _v(val)

        months.append({
            "date":              period,
            "headline":          _v(h),
            "core":              _v(core),
            "services":          _v(s),
            "nonEnergyGoods":    _v(neg),
            "foodAlcoholTobacco":_v(f),
            "energy":            _v(e),
            "subIndices":        sub_dict,
        })

    return months


# ---------------------------------------------------------------------------
# FRED history fetch helpers (secondary source — requires API key)
# ---------------------------------------------------------------------------

_FRED_SERIES = {
    "headline": "CP0000EZ19M086NEST",
    "services": "CP0600EZ19M086NEST",
    "food":     "CP0111EZ19M086NEST",
    "energy":   "CP0500EZ19M086NEST",
}


def _fetch_fred_history(analyzer: EurozoneInflationAnalyzer) -> Optional[List[Dict]]:
    """
    Fetch EA HICP series from FRED and return in React-friendly format.
    Returns None if any series fails (caller falls back to simulated data).
    """
    frames: Dict[str, pd.Series] = {}
    for key, sid in _FRED_SERIES.items():
        df = analyzer.fetch_fred_data(sid, start_date="2022-01-01")
        if df is None:
            return None
        frames[key] = df.set_index("date")["value"]

    aligned = pd.DataFrame(frames).dropna(how="all").sort_index().tail(36)
    w = analyzer.HICP_WEIGHTS
    cw = w["services"] + w["non_energy_goods"]

    months: List[Dict] = []
    for dt, row in aligned.iterrows():
        h = row.get("headline", np.nan)
        s = row.get("services",  np.nan)
        f = row.get("food",      np.nan)
        e = row.get("energy",    np.nan)

        # Estimate NEIG from energy/food/services if headline available:
        # headline ≈ s*w_s + neig*w_n + f*w_f + e*w_e
        neig = np.nan
        if not any(np.isnan(x) for x in [h, s, f, e]):
            neig_val = (h - s * w["services"] - f * w["food_alcohol_tobacco"] - e * w["energy"])
            if w["non_energy_goods"] > 0:
                neig = neig_val / w["non_energy_goods"]

        core = np.nan
        if not any(np.isnan(x) for x in [s, neig]):
            core = (s * w["services"] + neig * w["non_energy_goods"]) / cw

        def _fmt(v: float) -> Optional[float]:
            return round(float(v), 2) if not np.isnan(v) else None

        months.append({
            "date":              dt.strftime("%Y-%m"),
            "headline":          _fmt(h),
            "core":              _fmt(core),
            "services":          _fmt(s),
            "nonEnergyGoods":    _fmt(neig),
            "foodAlcoholTobacco":_fmt(f),
            "energy":            _fmt(e),
        })
    return months


# ---------------------------------------------------------------------------
# Commodity helpers
# ---------------------------------------------------------------------------

_SIM_COMMODITY_HISTORY = {
    "brent":  [74.2, 76.5, 79.8, 83.1, 85.4, 88.3],
    "eugas":  [28.4, 30.1, 33.7, 36.2, 38.9, 41.1],
    "eurusd": [1.095, 1.087, 1.079, 1.071, 1.063, 1.058],
}

_COMMODITY_META = {
    "brent":  {"label": "Brent Crude",    "unit": "$/bbl",  "note": "Every $10 rise ≈ +0.2pp EA energy CPI"},
    "eugas":  {"label": "EU Natural Gas", "unit": "€/MWh",  "note": "Direct energy + industrial input cost"},
    "eurusd": {"label": "EUR/USD",        "unit": "",       "note": "Weaker EUR → pricier USD-denominated commodity imports"},
}


def _build_commodity_payload(
    snapshot_data: Optional[Dict],
) -> Dict:
    """
    Build the commodity signals payload consumed by CommoditySignalsPanel.
    Falls back to simulated values if no live data available.
    """
    if snapshot_data:
        hist_b  = snapshot_data.get("brent_history",  _SIM_COMMODITY_HISTORY["brent"])
        hist_g  = snapshot_data.get("gas_history",    _SIM_COMMODITY_HISTORY["eugas"])
        hist_fx = snapshot_data.get("eurusd_history", _SIM_COMMODITY_HISTORY["eurusd"])
        source = "live"
    else:
        hist_b  = _SIM_COMMODITY_HISTORY["brent"]
        hist_g  = _SIM_COMMODITY_HISTORY["eugas"]
        hist_fx = _SIM_COMMODITY_HISTORY["eurusd"]
        source = "simulated"

    def _last2(hist: List[float]):
        return hist[-1], hist[-2] if len(hist) >= 2 else hist[-1]

    b_cur,  b_prev  = _last2(hist_b)
    g_cur,  g_prev  = _last2(hist_g)
    fx_cur, fx_prev = _last2(hist_fx)

    b_chg  = (b_cur  - b_prev)  / b_prev  * 100 if b_prev  else 0
    g_chg  = (g_cur  - g_prev)  / g_prev  * 100 if g_prev  else 0
    fx_chg = (fx_cur - fx_prev) / fx_prev * 100 if fx_prev else 0

    upside   = (b_chg > 1.5) + (g_chg > 1.5) + (fx_chg < -0.5)
    downside = (b_chg < -1.5) + (g_chg < -1.5) + (fx_chg > 0.5)

    if upside >= 2:
        pressure = "upside"
        sigma_mult = 1.35
    elif downside >= 2:
        pressure = "downside"
        sigma_mult = 0.85
    else:
        pressure = "neutral"
        sigma_mult = 1.0

    commodities = [
        {**_COMMODITY_META["brent"],
         "id": "brent",   "current": round(b_cur, 1),  "prev": round(b_prev, 1),
         "change": round(b_chg, 2),   "history": [round(v, 1) for v in hist_b]},
        {**_COMMODITY_META["eugas"],
         "id": "eugas",   "current": round(g_cur, 1),  "prev": round(g_prev, 1),
         "change": round(g_chg, 2),   "history": [round(v, 1) for v in hist_g]},
        {**_COMMODITY_META["eurusd"],
         "id": "eurusd",  "current": round(fx_cur, 4), "prev": round(fx_prev, 4),
         "change": round(fx_chg, 2),  "history": [round(v, 4) for v in hist_fx],
         "inflationary": fx_chg < 0},
    ]

    return {
        "pressure":        pressure,
        "sigmaMultiplier": sigma_mult,
        "source":          source,
        "commodities":     commodities,
    }


def _extract_commodity_history(
    commodity_df: pd.DataFrame,
) -> Optional[Dict]:
    """Pull trailing 6-month price history from the commodity DataFrame."""
    if commodity_df.empty:
        return None

    def _tail6(col: str) -> Optional[List[float]]:
        if col not in commodity_df.columns:
            return None
        s = commodity_df[col].dropna().tail(6)
        return [round(float(v), 4) for v in s.tolist()] if len(s) >= 2 else None

    brent_h  = _tail6("brent_usd")
    gas_h    = _tail6("eu_gas_usd")
    eurusd_h = _tail6("eurusd")

    if not brent_h and not gas_h and not eurusd_h:
        return None

    return {
        "brent_history":  brent_h  or _SIM_COMMODITY_HISTORY["brent"],
        "gas_history":    gas_h    or _SIM_COMMODITY_HISTORY["eugas"],
        "eurusd_history": eurusd_h or _SIM_COMMODITY_HISTORY["eurusd"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict:
    az = _get_analyzer()
    return {
        "status":              "ok",
        "fred_available":      bool(FRED_API_KEY),
        "weight_source":       az._weight_source,
        "weights":             az.HICP_WEIGHTS,
        "subindices_available": True,
        "subindices_count":    len(_ECB_SUBINDICES),
    }


@app.get("/api/history")
def history() -> Dict:
    """
    Return historical HICP component data (monthly, year-on-year %).

    Data source priority:
      1. ECB Statistical Data Warehouse — real official data, no API key needed.
         Fetches: headline, services, NEIG, food, energy via 5 separate series.
      2. FRED — if FRED_API_KEY env var is set (secondary fallback).
      3. Hardcoded simulated data — last resort if all APIs are unreachable.
    """
    cached = _cache_get("history")
    if cached:
        return cached

    az = _get_analyzer()
    months: Optional[List[Dict]] = None
    data_source = "simulated"

    # Layer 1: ECB SDW (no key required)
    try:
        months = _fetch_ecb_hicp_real(start="2022-01")
        if months:
            data_source = "ecb_sdw"
            print(f"ECB SDW: loaded {len(months)} months of real HICP data "
                  f"({months[0]['date']} → {months[-1]['date']})")
    except Exception as exc:
        print(f"ECB SDW fetch failed: {exc}")

    # Layer 2: FRED (requires key)
    if not months and FRED_API_KEY:
        try:
            months = _fetch_fred_history(az)
            if months:
                data_source = "fred"
        except Exception as exc:
            print(f"FRED fetch failed: {exc}")

    # Layer 3: simulated fallback
    if not months:
        print("All real data sources unavailable — using simulated fallback data.")
        months = _build_simulated_history(az.HICP_WEIGHTS)

    result = {
        "months":      months,
        "source":      data_source,
        "weightSource":az._weight_source,
        "weights": {
            "services":           az.HICP_WEIGHTS["services"],
            "nonEnergyGoods":     az.HICP_WEIGHTS["non_energy_goods"],
            "foodAlcoholTobacco": az.HICP_WEIGHTS["food_alcohol_tobacco"],
            "energy":             az.HICP_WEIGHTS["energy"],
        },
    }
    _cache_set("history", result)
    return result


@app.get("/api/forecast")
def forecast(
    periods:     int = Query(default=6,     ge=1, le=12),
    methodology: str = Query(default="ecb", pattern="^(ecb|boe)$"),
) -> Dict:
    """
    Run the Python ensemble (Exp Smoothing + ARIMA + ARIMAX) + Monte Carlo
    and return probabilistic fan chart bands.

    Returns:
        dates       list of "YYYY-MM" strings
        central     point forecast
        p10..p90    Monte Carlo percentile bands
        fanBands    pre-computed Recharts stacked-area format
        methodology_note  description of models used
    """
    cache_key = f"forecast:{periods}:{methodology}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    az = _get_analyzer()

    # Get historical data (re-use cached history if available)
    hist = _cache_get("history")
    if not hist:
        hist = history()  # populates cache as side-effect

    months = hist["months"]

    # Build DataFrame for the analyzer
    if hist["source"] == "fred":
        df = pd.DataFrame(months)
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={
            "nonEnergyGoods":    "non_energy_goods",
            "foodAlcoholTobacco":"food_alcohol_tobacco",
            "headline":          "hicp_headline",
            "core":              "hicp_core",
        })
    else:
        df = _sim_history_to_df(months)

    # Fetch commodity data for ARIMAX (best-effort; non-fatal if unavailable)
    try:
        az.fetch_commodity_data()
    except Exception:
        pass

    # Run ensemble + Monte Carlo
    result = az.forecast_with_uncertainty(df, periods=periods, methodology=methodology)

    # Add pre-computed Recharts stacked-area format so React doesn't need to transform
    fan_bands = []
    for i, d in enumerate(result["dates"]):
        p10 = result["p10"][i]
        p25 = result["p25"][i]
        p75 = result["p75"][i]
        p90 = result["p90"][i]
        fan_bands.append({
            "date":       d,
            "central":    round(result["central"][i], 3),
            "p50":        round(result["p50"][i], 3),
            "outerLow":   round(p10, 3),
            "outerWidth": round(p90 - p10, 3),
            "innerLow":   round(p25, 3),
            "innerWidth": round(p75 - p25, 3),
            "isForecast": True,
        })

    result["fanBands"] = fan_bands
    _cache_set(cache_key, result)
    return result


@app.get("/api/commodities")
def commodities() -> Dict:
    """
    Return commodity signals panel data: Brent crude, EU gas, EUR/USD with
    trailing 6-month history, MoM % changes, and inflation pressure signal.

    Fetches live data via FRED (if key set) + yfinance; falls back to the
    same simulated values previously embedded in the React component.
    """
    cached = _cache_get("commodities")
    if cached:
        return cached

    az = _get_analyzer()
    commodity_df = pd.DataFrame()

    try:
        commodity_df = az.fetch_commodity_data()
    except Exception as exc:
        print(f"Commodity fetch failed: {exc}")

    snapshot_data = _extract_commodity_history(commodity_df)
    result = _build_commodity_payload(snapshot_data)

    # Also include the snapshot pressure from Python's own logic
    if az._commodity_snapshot:
        snap = az._commodity_snapshot
        result["snapshot"] = {
            "brent_usd":     snap.brent_usd,
            "eu_gas_usd":    snap.eu_gas_usd,
            "eurusd":        snap.eurusd,
            "food_index":    snap.food_index,
            "brent_mom_pct": snap.brent_mom_pct,
            "gas_mom_pct":   snap.gas_mom_pct,
            "as_of":         snap.as_of,
            "source":        snap.source,
            "pressure":      snap.commodity_pressure(),
        }

    _cache_set("commodities", result)
    return result


@app.get("/api/subindices")
def subindices() -> Dict:
    """
    Return HICP sub-index time series (11 COICOP categories).

    Fetches from ECB SDW in parallel (same free API as /api/history).
    Falls back to values derived from the 4 parent components if unavailable.
    Kept as a separate endpoint to avoid bloating /api/history payload.
    """
    cached = _cache_get("subindices")
    if cached:
        return cached

    # Try ECB SDW first
    az = _get_analyzer()
    sub_series = _fetch_ecb_subindices(start="2022-01")
    available  = [k for k, v in sub_series.items() if v is not None]
    source     = "ecb_sdw" if available else "simulated"

    if available:
        # Align to the union of all available indices
        all_indices = set()
        for s in sub_series.values():
            if s is not None:
                all_indices.update(s.index.tolist())
        idx = sorted(all_indices)

        months: List[Dict] = []
        for period in idx:
            row: Dict[str, Any] = {"date": period}
            for name, s in sub_series.items():
                if s is not None and period in s.index:
                    val = s[period]
                    row[name] = round(float(val), 2) if not np.isnan(val) else None
                else:
                    row[name] = None
            months.append(row)
    else:
        # Fallback: derive from simulated parent components
        sim = _build_simulated_history(az.HICP_WEIGHTS)
        months = [
            {"date": m["date"], **m.get("subIndices", {})}
            for m in sim
        ]

    result = {
        "months":    months,
        "source":    source,
        "available": available,
        "meta": {k: {"label": k, "source": "ecb_sdw"}
                 for k in _ECB_SUBINDICES},
    }
    _cache_set("subindices", result)
    return result


@app.get("/api/leading-indicators")
def leading_indicators() -> Dict:
    """
    Return macroeconomic leading indicator time series.

    Fetches from ECB SDW (free, no API key):
      - PPI (Producer Price Index):  leads HICP by ~2-4 months
      - Import Price Index:           leads HICP by ~1-3 months
      - Labor Costs / Negotiated wages: leads services inflation by ~3-6 months

    These are used as extra ARIMAX exogenous regressors when available.
    Returns null values if ECB SDW is unreachable.
    """
    cached = _cache_get("leading_indicators")
    if cached:
        return cached

    fetcher = _get_leading_fetcher()
    df = fetcher.get_dataframe()
    available = []
    months: List[Dict] = []

    if df is not None and not df.empty:
        available = list(df.columns)
        for dt, row in df.iterrows():
            entry: Dict[str, Any] = {"date": str(dt)[:7]}  # YYYY-MM
            for col in df.columns:
                v = row[col]
                entry[col] = round(float(v), 2) if not (v is None or (isinstance(v, float) and np.isnan(v))) else None
            months.append(entry)
        source = "ecb_sdw"
    else:
        source = "unavailable"

    result = {
        "months":    months,
        "source":    source,
        "available": available,
        "lags":      {"ppi": 3, "import_prices": 2, "labor_costs": 4},
        "note":      "Leading indicators used as extra ARIMAX regressors when available.",
    }
    _cache_set("leading_indicators", result)
    return result


@app.get("/api/trueflation")
def trueflation() -> Dict:
    """
    Return Trueflation US real-time inflation as a cross-Atlantic leading signal.

    IMPORTANT CAVEAT: Trueflation measures US inflation only (not Eurozone).
    Used here as a directional leading signal — US inflation often precedes
    EA inflation by ~2-3 months in supply-chain-driven cycles. This signal
    can diverge significantly during EUR-specific shocks.

    Requires TRUFLATION_API_KEY env var. Returns {"available": false} without it.
    """
    if not TRUFLATION_API_KEY:
        return {
            "available": False,
            "reason":    "TRUFLATION_API_KEY env var not set",
            "caveat":    "Trueflation measures US inflation only — Eurozone leading signal only.",
        }

    cached = _cache_get("trueflation")
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://api.truflation.com/inflation",
            headers={"Authorization": f"Bearer {TRUFLATION_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        us_inflation = float(data.get("yearOverYear", data.get("value", 0)))

        # Get current EA headline from cached history for comparison
        hist = _cache_get("history")
        ea_headline = None
        if hist and hist.get("months"):
            last = hist["months"][-1]
            ea_headline = last.get("headline")

        spread = round(us_inflation - ea_headline, 2) if ea_headline is not None else None
        if spread is not None:
            if spread > 1.0:
                lead_signal = "US elevated — potential EA upside in 2-3 months"
            elif spread < -1.0:
                lead_signal = "US below EA — potential EA downside in 2-3 months"
            else:
                lead_signal = "US/EA spread neutral"
        else:
            lead_signal = "insufficient data"

        result = {
            "available":          True,
            "us_realtime":        round(us_inflation, 2),
            "ea_headline":        ea_headline,
            "spread_us_minus_ea": spread,
            "lead_signal":        lead_signal,
            "lag_months":         2,
            "source":             "truflation",
            "caveat":             "US-only data — directional EA leading signal only.",
        }
        _cache_set("trueflation", result)
        return result

    except Exception as exc:
        return {
            "available": False,
            "reason":    f"Trueflation API error: {exc}",
            "caveat":    "Trueflation measures US inflation only — Eurozone leading signal only.",
        }
