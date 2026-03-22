"""
Eurozone Inflation Analyzer — Full Overhaul

Implements ECB suite-of-models ensemble forecasting with Monte Carlo
uncertainty quantification and Bank of England two-piece normal methodology.

New in this version:
  - HICPWeightFetcher: dynamic weight fetching from Eurostat/ECB APIs
  - EnsembleForecaster: inverse-RMSE weighted combination of multiple models
  - MonteCarloEngine: 10,000-path simulation for probabilistic fan charts
  - TwoPieceNormal: Bank of England asymmetric uncertainty methodology
  - CountryContributionAnalyzer: NCB-style country-level decomposition
  - CommodityFetcher: Brent crude, EU gas, food index, EUR/USD via FRED + yfinance
  - ARIMAXModel: ARIMA with commodity exogenous regressors (commodity shock channel)
"""

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Optional package detection
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install: pip install statsmodels")

try:
    from scipy.stats import norm as scipy_norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install: pip install scipy")

try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class ForecastResult:
    """Output of a single forecast model."""
    horizon: int
    central: np.ndarray        # shape (horizon,)
    residuals: np.ndarray      # shape (n_history,) — for MC volatility
    model_name: str
    fitted_values: np.ndarray  # shape (n_history,)
    rmse: float = 0.0

    def __post_init__(self):
        if len(self.residuals) > 0:
            self.rmse = float(np.sqrt(np.mean(self.residuals ** 2)))


@dataclass
class FanChartResult:
    """Output of MonteCarloEngine — probabilistic fan chart bands."""
    dates: List[str]
    central: np.ndarray
    p10: np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p90: np.ndarray
    simulation_mean: np.ndarray
    methodology_note: str = ""
    n_simulations: int = 10_000

    def to_dict(self) -> Dict:
        return {
            "dates": self.dates,
            "central": self.central.tolist(),
            "p10": self.p10.tolist(),
            "p25": self.p25.tolist(),
            "p50": self.p50.tolist(),
            "p75": self.p75.tolist(),
            "p90": self.p90.tolist(),
            "simulation_mean": self.simulation_mean.tolist(),
            "methodology_note": self.methodology_note,
            "n_simulations": self.n_simulations,
        }


@dataclass
class CommoditySnapshot:
    """Current commodity price levels for display and risk assessment."""
    brent_usd: Optional[float] = None      # Brent crude, USD/barrel
    eu_gas_usd: Optional[float] = None     # EU natural gas, USD/MMBtu
    food_index: Optional[float] = None     # World Bank food price index
    eurusd: Optional[float] = None         # EUR/USD spot rate
    brent_mom_pct: Optional[float] = None  # Brent month-on-month % change
    gas_mom_pct: Optional[float] = None    # Gas month-on-month % change
    as_of: str = ""
    source: str = ""

    def commodity_pressure(self) -> str:
        """Simple signal: 'upside', 'downside', or 'neutral'."""
        signals = []
        if self.brent_mom_pct is not None:
            signals.append(self.brent_mom_pct)
        if self.gas_mom_pct is not None:
            signals.append(self.gas_mom_pct)
        if not signals:
            return "neutral"
        avg = sum(signals) / len(signals)
        if avg > 2.0:
            return "upside"
        if avg < -2.0:
            return "downside"
        return "neutral"

    def to_dict(self) -> Dict:
        return {
            "brent_usd": self.brent_usd,
            "eu_gas_usd": self.eu_gas_usd,
            "food_index": self.food_index,
            "eurusd": self.eurusd,
            "brent_mom_pct": self.brent_mom_pct,
            "gas_mom_pct": self.gas_mom_pct,
            "as_of": self.as_of,
            "source": self.source,
            "commodity_pressure": self.commodity_pressure(),
        }


# ===========================================================================
# Phase 1 — Dynamic Weight Fetching
# ===========================================================================

class HICPWeightFetcher:
    """
    Fetches official HICP component weights from Eurostat/ECB APIs.

    Three-layer fallback:
      1. Eurostat SDMX-JSON (primary)
      2. ECB Data Portal API (secondary — structure probe)
      3. Hardcoded 2025 ECB weights (tertiary)

    Cache: ~/.inflation_cache/hicp_weights.json
    Cache invalidated in January of a new year (ECB publishes annual updates).
    """

    CACHE_DIR = Path.home() / '.inflation_cache'
    CACHE_FILE = CACHE_DIR / 'hicp_weights.json'

    # Official ECB 2025 HICP weights (fallback)
    FALLBACK_WEIGHTS: Dict = {
        'services': 0.457,
        'non_energy_goods': 0.256,
        'food_alcohol_tobacco': 0.193,
        'energy': 0.094,
        'source': 'hardcoded_2025',
        'year': 2025,
    }

    # Eurostat SDMX-JSON — division-level COICOP codes (CP01..CP12 are valid in prc_hicp_inw)
    # We extract CP01+CP02 (food) and derive a partial update; energy/services/NEIG via ECB SDW.
    EUROSTAT_URL = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_inw"
        "?geo=EA&unit=W_PPT&coicop=CP00,CP01,CP02"
        "&sinceTimePeriod=2024&format=JSON"
    )

    # ECB Statistical Data Warehouse — annual HICP weights for euro area (4-component split)
    # Series key: ICP.A.{geo}.N.{coicop}.4.ANW
    ECB_SDW_URL = (
        "https://data-api.ecb.europa.eu/service/data/ICP"
        "/A.U2.N.SERV00+NEIG00+FOOD00+NRG000+000000.4.ANW"
        "?lastNObservations=2&format=jsondata"
    )

    # Mapping of ECB SDW COICOP → component names
    ECB_SDW_COICOP_MAP = {
        'SERV00': 'services',
        'NEIG00': 'non_energy_goods',
        'FOOD00': 'food_alcohol_tobacco',
        'NRG000': 'energy',
        '000000': 'total',
    }

    def fetch(self) -> Dict:
        """
        Return HICP component weights dict with keys:
          services, non_energy_goods, food_alcohol_tobacco, energy, source, year
        """
        if self._is_cache_valid():
            cached = self._load_cache()
            if cached:
                print(f"Using cached weights (source: {cached.get('source', 'unknown')}, "
                      f"year: {cached.get('year', '?')})")
                return cached

        # Layer 1: ECB Statistical Data Warehouse (4-component annual weights)
        weights = self._fetch_ecb_sdw()
        if weights:
            weights['source'] = 'ecb_sdw'
            weights['year'] = datetime.now().year
            self._save_cache(weights)
            print(f"Fetched live weights from ECB SDW (year {weights['year']})")
            return weights

        # Layer 2: Eurostat SDMX-JSON (division-level CP01+CP02 for food only)
        weights = self._fetch_eurostat_partial()
        if weights:
            weights['source'] = 'eurostat_partial'
            weights['year'] = datetime.now().year
            self._save_cache(weights)
            print(f"Fetched partial weights from Eurostat (food updated, others hardcoded)")
            return weights

        # Layer 3: hardcoded fallback
        print("Weight APIs unavailable — using hardcoded 2025 ECB values.")
        return self.FALLBACK_WEIGHTS.copy()

    # ------------------------------------------------------------------
    def _is_cache_valid(self) -> bool:
        if not self.CACHE_FILE.exists():
            return False
        try:
            with open(self.CACHE_FILE) as f:
                cached = json.load(f)
            cached_year = cached.get('year', 0)
            now = datetime.now()
            # Invalidate in January of a new year
            if now.month == 1 and now.year > cached_year:
                return False
            return True
        except Exception:
            return False

    def _load_cache(self) -> Optional[Dict]:
        try:
            with open(self.CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, weights: Dict) -> None:
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(weights, f, indent=2)
        except Exception as e:
            print(f"Cache write failed (non-fatal): {e}")

    def _fetch_ecb_sdw(self) -> Optional[Dict]:
        """
        Fetch 4-component HICP annual weights from ECB Statistical Data Warehouse.
        Series IDs: ICP.A.U2.N.{COICOP}.4.ANW (Annual Weight, euro area).
        """
        try:
            resp = requests.get(self.ECB_SDW_URL, timeout=10,
                                headers={'Accept': 'application/json'})
            resp.raise_for_status()
            data = resp.json()

            # ECB JSON-data format: data.dataSets[0].series[key].observations
            datasets = data.get('dataSets', [])
            structure = data.get('structure', {})
            if not datasets or not structure:
                return None

            # Dimension positions: find COICOP dimension
            dims = structure.get('dimensions', {}).get('series', [])
            coicop_pos = next(
                (i for i, d in enumerate(dims) if d.get('id') == 'COICOP'), None
            )
            if coicop_pos is None:
                return None

            coicop_values = dims[coicop_pos].get('values', [])

            raw: Dict[str, float] = {}
            series_dict = datasets[0].get('series', {})

            for series_key, series_data in series_dict.items():
                key_parts = series_key.split(':')
                if len(key_parts) <= coicop_pos:
                    continue
                coicop_i = int(key_parts[coicop_pos])
                # Reverse lookup: index → ID
                coicop_id = coicop_values[coicop_i]['id'] if coicop_i < len(coicop_values) else None
                if not coicop_id:
                    continue
                component = self.ECB_SDW_COICOP_MAP.get(coicop_id)
                if not component:
                    continue
                # Take most recent observation
                obs = series_data.get('observations', {})
                if not obs:
                    continue
                latest_obs = obs[max(obs.keys(), key=int)]
                if latest_obs and latest_obs[0] is not None:
                    raw[component] = float(latest_obs[0])

            total = raw.get('total', 1000.0)
            required = ['services', 'non_energy_goods', 'food_alcohol_tobacco', 'energy']
            if not all(k in raw for k in required):
                return None

            return {k: round(raw[k] / total, 4) for k in required}

        except Exception as e:
            print(f"ECB SDW fetch failed: {e}")
            return None

    def _fetch_eurostat_partial(self) -> Optional[Dict]:
        """
        Fetch Eurostat division-level weights (CP01, CP02) to update the food component.
        Energy, services, and NEIG fall back to hardcoded 2025 ratios.
        Returns None if fetch fails entirely.
        """
        try:
            resp = requests.get(self.EUROSTAT_URL, timeout=10)
            resp.raise_for_status()
            food_weight = self._parse_food_from_sdmx(resp.json())
            if food_weight is None:
                return None
            fb = self.FALLBACK_WEIGHTS
            remaining = 1.0 - food_weight
            # Scale services, NEIG, energy proportionally from hardcoded
            orig_non_food = fb['services'] + fb['non_energy_goods'] + fb['energy']
            scale = remaining / orig_non_food
            return {
                'services': round(fb['services'] * scale, 4),
                'non_energy_goods': round(fb['non_energy_goods'] * scale, 4),
                'food_alcohol_tobacco': round(food_weight, 4),
                'energy': round(fb['energy'] * scale, 4),
            }
        except Exception as e:
            print(f"Eurostat partial fetch failed: {e}")
            return None

    def _parse_food_from_sdmx(self, data: Dict) -> Optional[float]:
        """Extract CP01+CP02 food weight from SDMX-JSON response, normalized by CP00."""
        try:
            dim_ids: List[str] = data.get('id', [])
            sizes: List[int] = data.get('size', [])
            raw_values: Dict[str, float] = data.get('value', {})
            dimensions: Dict = data.get('dimension', {})

            def dim_index(dim: str, code: str) -> Optional[int]:
                return dimensions.get(dim, {}).get('category', {}).get('index', {}).get(code)

            def flat_idx(codes: Dict[str, str]) -> Optional[int]:
                indices = []
                for dim in dim_ids:
                    code = codes.get(dim)
                    if code is None:
                        first = next(iter(
                            dimensions.get(dim, {}).get('category', {}).get('index', {}).values()
                        ), 0)
                        indices.append(first)
                    else:
                        i = dim_index(dim, code)
                        if i is None:
                            return None
                        indices.append(i)
                result = 0
                for pos, idx in enumerate(indices):
                    stride = 1
                    for k in range(pos + 1, len(sizes)):
                        stride *= sizes[k]
                    result += idx * stride
                return result

            time_cat = dimensions.get('TIME_PERIOD', {}).get('category', {})
            latest_period = min(time_cat.get('index', {}).items(), key=lambda x: x[1])[0]

            def get_val(coicop: str) -> Optional[float]:
                codes: Dict[str, str] = {'COICOP': coicop, 'TIME_PERIOD': latest_period}
                for geo in ['EA', 'EA19', 'EA20']:
                    if dim_index('GEO', geo) is not None:
                        codes['GEO'] = geo
                        break
                fi = flat_idx(codes)
                if fi is None:
                    return None
                v = raw_values.get(str(fi))
                return float(v) if v is not None else None

            total = get_val('CP00')
            cp01 = get_val('CP01')
            cp02 = get_val('CP02')
            if total and cp01 is not None and cp02 is not None:
                return (cp01 + cp02) / total
            return None
        except Exception:
            return None


# ===========================================================================
# Phase 2 — Model Refactor + Ensemble
# ===========================================================================

class BaseInflationModel(ABC):
    """Abstract base for all inflation forecast models."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def fit(self, series: pd.Series,
            exog: Optional[pd.DataFrame] = None) -> 'BaseInflationModel':
        pass

    @abstractmethod
    def forecast(self, horizon: int,
                 forecast_exog: Optional[np.ndarray] = None) -> ForecastResult:
        pass


class ExponentialSmoothingModel(BaseInflationModel):
    """Simple exponential smoothing with trend component (α=0.3)."""

    def __init__(self, alpha: float = 0.3, trend_weight: float = 0.1):
        self.alpha = alpha
        self.trend_weight = trend_weight
        self._level: Optional[float] = None
        self._trend: Optional[float] = None
        self._series: Optional[pd.Series] = None
        self._fitted_values: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"ExpSmoothing(α={self.alpha})"

    def fit(self, series: pd.Series,
            exog: Optional[pd.DataFrame] = None) -> 'ExponentialSmoothingModel':
        # exog ignored — pure time-series model
        self._series = series.dropna().reset_index(drop=True)
        n = len(self._series)
        fitted = np.zeros(n)
        level = float(self._series.iloc[0])

        for i in range(n):
            fitted[i] = level
            level = self.alpha * float(self._series.iloc[i]) + (1 - self.alpha) * level

        self._level = level
        if n >= 6:
            self._trend = float(self._series.iloc[-1] - self._series.iloc[-6])
        elif n >= 2:
            self._trend = float(self._series.iloc[-1] - self._series.iloc[0])
        else:
            self._trend = 0.0

        self._fitted_values = fitted
        return self

    def forecast(self, horizon: int,
                 forecast_exog: Optional[np.ndarray] = None) -> ForecastResult:
        # forecast_exog ignored — pure time-series model
        if self._level is None:
            raise RuntimeError("Call fit() first")

        central = np.zeros(horizon)
        level = self._level

        for i in range(horizon):
            val = level + self._trend * self.trend_weight * (i + 1)
            central[i] = val
            level = self.alpha * val + (1 - self.alpha) * level

        residuals = self._series.values - self._fitted_values

        return ForecastResult(
            horizon=horizon,
            central=central,
            residuals=residuals,
            model_name=self.name,
            fitted_values=self._fitted_values,
        )


class ARIMAModel(BaseInflationModel):
    """ARIMA(1,1,1) forecasting model (requires statsmodels)."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self._series: Optional[pd.Series] = None
        self._fitted = None
        self._fitted_values: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}"

    def fit(self, series: pd.Series,
            exog: Optional[pd.DataFrame] = None) -> 'ARIMAModel':
        # exog ignored — use ARIMAXModel for commodity-aware forecasting
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMAModel")
        self._series = series.dropna().reset_index(drop=True)
        model = ARIMA(self._series, order=self.order)
        self._fitted = model.fit()
        self._fitted_values = self._fitted.fittedvalues.values
        self._residuals = self._fitted.resid.values
        return self

    def forecast(self, horizon: int,
                 forecast_exog: Optional[np.ndarray] = None) -> ForecastResult:
        # forecast_exog ignored — use ARIMAXModel for commodity-aware forecasting
        if self._fitted is None:
            raise RuntimeError("Call fit() first")
        try:
            fc = self._fitted.get_forecast(steps=horizon)
            central = fc.predicted_mean.values
        except Exception:
            central = np.full(horizon, float(self._series.iloc[-1]))

        return ForecastResult(
            horizon=horizon,
            central=central,
            residuals=self._residuals if self._residuals is not None else np.array([0.0]),
            model_name=self.name,
            fitted_values=self._fitted_values if self._fitted_values is not None else np.array([]),
        )


class ARIMAXModel(BaseInflationModel):
    """
    ARIMAX(1,1,1) — ARIMA extended with commodity price exogenous regressors.

    Exogenous variables (lagged % changes, supplied by CommodityFetcher):
      - d_brent_lag1:  1-month lagged % change in Brent crude
      - d_eurusd_lag1: 1-month lagged % change in EUR/USD rate

    These capture the commodity shock channel that pure ARIMA misses:
      oil spike → energy component → headline inflation (1-2 month lag)
      EUR/USD depreciation → import prices → all components (1-2 month lag)

    Falls back silently to plain ARIMA when no exog data is provided.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self._series: Optional[pd.Series] = None
        self._fitted = None
        self._fitted_values: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._used_exog: bool = False

    @property
    def name(self) -> str:
        suffix = "+commodities" if self._used_exog else " (no exog)"
        return f"ARIMAX{self.order}{suffix}"

    def fit(self, series: pd.Series,
            exog: Optional[pd.DataFrame] = None) -> 'ARIMAXModel':
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMAXModel")

        self._series = series.dropna().reset_index(drop=True)
        self._used_exog = False

        if exog is not None and not exog.empty:
            exog_clean = exog.dropna()
            n_valid = min(len(self._series), len(exog_clean))
            if n_valid >= 12:  # need enough data to fit with exog
                s = self._series.iloc[-n_valid:].reset_index(drop=True)
                exog_np = exog_clean.values[-n_valid:]
                try:
                    model = ARIMA(s, order=self.order, exog=exog_np)
                    self._fitted = model.fit()
                    self._fitted_values = self._fitted.fittedvalues.values
                    self._residuals = self._fitted.resid.values
                    self._series = s
                    self._used_exog = True
                    return self
                except Exception as e:
                    print(f"ARIMAX fit with exog failed ({e}), falling back to ARIMA")

        # Fallback: plain ARIMA
        try:
            model = ARIMA(self._series, order=self.order)
            self._fitted = model.fit()
            self._fitted_values = self._fitted.fittedvalues.values
            self._residuals = self._fitted.resid.values
        except Exception as e:
            raise RuntimeError(f"ARIMAX/ARIMA fit failed: {e}")
        return self

    def forecast(self, horizon: int,
                 forecast_exog: Optional[np.ndarray] = None) -> ForecastResult:
        if self._fitted is None:
            raise RuntimeError("Call fit() first")
        try:
            if self._used_exog and forecast_exog is not None:
                fc = self._fitted.get_forecast(steps=horizon, exog=forecast_exog[:horizon])
            else:
                fc = self._fitted.get_forecast(steps=horizon)
            central = fc.predicted_mean.values
        except Exception:
            central = np.full(horizon, float(self._series.iloc[-1]))

        return ForecastResult(
            horizon=horizon,
            central=central,
            residuals=self._residuals if self._residuals is not None else np.array([0.0]),
            model_name=self.name,
            fitted_values=self._fitted_values if self._fitted_values is not None else np.array([]),
        )


class EnsembleForecaster:
    """
    Inverse-RMSE weighted ensemble of multiple forecast models.
    Lower RMSE → higher weight (better in-sample fit rewarded more).
    """

    def __init__(self, models: Optional[List[BaseInflationModel]] = None):
        if models is None:
            models = [ExponentialSmoothingModel()]
            if STATSMODELS_AVAILABLE:
                models.append(ARIMAModel())
        self.models = models
        self._results: List[ForecastResult] = []

    def fit_and_forecast(
        self,
        series: pd.Series,
        horizon: int = 6,
        exog: Optional[pd.DataFrame] = None,
        forecast_exog: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit all models and return inverse-RMSE weighted ensemble forecast.

        Parameters:
            exog: optional lagged commodity DataFrame for ARIMAX (aligned to series)
            forecast_exog: optional forward commodity matrix shape (horizon, n_vars)

        Returns:
            (central_forecast shape (horizon,), combined_residuals shape (N,))
        """
        self._results = []
        all_residuals = []

        for model in self.models:
            try:
                model.fit(series, exog=exog)
                result = model.forecast(horizon, forecast_exog=forecast_exog)
                self._results.append(result)
                if len(result.residuals) > 0:
                    all_residuals.append(result.residuals)
            except Exception as e:
                print(f"Model {model.name} skipped: {e}")

        if not self._results:
            last_val = float(series.dropna().iloc[-1])
            return np.full(horizon, last_val), np.array([0.0])

        # Inverse-RMSE weighting
        rmses = np.array([max(r.rmse, 1e-6) for r in self._results])
        inv_rmse = 1.0 / rmses
        weights = inv_rmse / inv_rmse.sum()

        central = np.zeros(horizon)
        for w, result in zip(weights, self._results):
            n = min(len(result.central), horizon)
            central[:n] += w * result.central[:n]

        combined_residuals = (
            np.concatenate(all_residuals) if all_residuals else np.array([0.0])
        )
        return central, combined_residuals

    @property
    def model_weights(self) -> Dict[str, float]:
        if not self._results:
            return {}
        rmses = np.array([max(r.rmse, 1e-6) for r in self._results])
        inv_rmse = 1.0 / rmses
        weights = inv_rmse / inv_rmse.sum()
        return {r.model_name: float(w) for r, w in zip(self._results, weights)}


# ===========================================================================
# Phase 3 — Monte Carlo Engine
# ===========================================================================

class MonteCarloEngine:
    """
    Monte Carlo simulation for probabilistic inflation fan charts.

    Headline-level: draws innovations N(0, σ²) with persistence decay.
    Component-level: draws from multivariate normal using estimated covariance.
    """

    def __init__(self, n_simulations: int = 10_000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def _estimate_vol(self, residuals: np.ndarray) -> float:
        if len(residuals) < 2:
            return 0.5
        return float(max(np.std(residuals), 0.01))

    def _estimate_covariance(self, component_changes: np.ndarray) -> np.ndarray:
        """Estimate covariance with optional Ledoit-Wolf shrinkage."""
        n, k = component_changes.shape
        if SKLEARN_AVAILABLE and n > k * 2:
            try:
                cov = LedoitWolf().fit(component_changes).covariance_
            except Exception:
                cov = np.cov(component_changes.T)
        else:
            cov = np.cov(component_changes.T) if k > 1 else np.array([[np.var(component_changes)]])
        # Diagonal regularization for near-singular matrices
        cov += 1e-6 * np.eye(cov.shape[0])
        return cov

    def run_headline_level(
        self,
        central_forecast: np.ndarray,
        residuals: np.ndarray,
        horizon: Optional[int] = None,
    ) -> 'FanChartResult':
        """
        Headline-level MC simulation.

        Uses auto-correlated innovations (AR(1) persistence with decay=0.7)
        to produce realistic widening uncertainty bands.
        """
        if horizon is None:
            horizon = len(central_forecast)

        vol = self._estimate_vol(residuals)
        decay = 0.7  # uncertainty persistence

        innovations = self.rng.normal(0.0, vol, size=(self.n_simulations, horizon))
        paths = np.zeros((self.n_simulations, horizon))

        for h in range(horizon):
            if h == 0:
                paths[:, h] = central_forecast[h] + innovations[:, h]
            else:
                shock = innovations[:, h]
                carry = decay * (paths[:, h - 1] - central_forecast[h - 1])
                paths[:, h] = central_forecast[h] + carry + shock

        return self._build_fan_chart(central_forecast, paths)

    def run_component_level(
        self,
        component_forecasts: Dict[str, np.ndarray],
        component_history: Dict[str, pd.Series],
        weights: Dict[str, float],
        horizon: Optional[int] = None,
    ) -> 'FanChartResult':
        """
        Component-level MC simulation with cross-component correlations.

        Draws innovations from multivariate normal estimated from historical
        monthly changes. Weighted sum of components gives headline path.
        """
        components = list(component_forecasts.keys())
        n_comp = len(components)

        if horizon is None:
            horizon = len(next(iter(component_forecasts.values())))

        # Align historical series lengths
        histories = [component_history[c].diff().dropna() for c in components]
        min_len = min(len(h) for h in histories)

        if min_len < 4:
            # Fall back to headline-level simulation
            headline_central = np.sum(
                [component_forecasts[c] * weights.get(c, 1.0 / n_comp) for c in components],
                axis=0,
            )
            all_residuals = np.concatenate([h.values for h in histories])
            return self.run_headline_level(headline_central, all_residuals, horizon)

        changes = np.column_stack([h.values[-min_len:] for h in histories])
        cov = self._estimate_covariance(changes)
        means = np.zeros(n_comp)
        w = np.array([weights.get(c, 1.0 / n_comp) for c in components])

        # Headline central
        headline_central = np.array([
            sum(component_forecasts[c][h] * weights.get(c, 1.0 / n_comp) for c in components)
            for h in range(horizon)
        ])

        # Simulate paths
        headline_paths = np.zeros((self.n_simulations, horizon))
        for h in range(horizon):
            comp_central = np.array([component_forecasts[c][h] for c in components])
            innovations = self.rng.multivariate_normal(means, cov, size=self.n_simulations)
            comp_paths = comp_central + innovations
            headline_paths[:, h] = comp_paths @ w

        return self._build_fan_chart(headline_central, headline_paths)

    def _build_fan_chart(
        self, central: np.ndarray, paths: np.ndarray
    ) -> 'FanChartResult':
        return FanChartResult(
            dates=[],
            central=central,
            p10=np.percentile(paths, 10, axis=0),
            p25=np.percentile(paths, 25, axis=0),
            p50=np.percentile(paths, 50, axis=0),
            p75=np.percentile(paths, 75, axis=0),
            p90=np.percentile(paths, 90, axis=0),
            simulation_mean=np.mean(paths, axis=0),
            methodology_note=(
                "ECB suite-of-models ensemble (Exp Smoothing + ARIMA), "
                "inverse-RMSE weighted, 10k-path Monte Carlo"
            ),
            n_simulations=self.n_simulations,
        )


# ===========================================================================
# Phase 3b — Bank of England Two-Piece Normal
# ===========================================================================

class TwoPieceNormal:
    """
    Bank of England two-piece normal distribution for skewed uncertainty.

    Reference: Britton, Fisher & Whitley (1998), BoE Quarterly Bulletin.

    Distribution:
      f(x) ∝ N(mode, σ_L)  for x < mode
      f(x) ∝ N(mode, σ_U)  for x ≥ mode

    σ_U > σ_L → upside-skewed uncertainty (more upside than downside risk).
    """

    def __init__(
        self, mode: float = 0.0, sigma_lower: float = 0.5, sigma_upper: float = 0.5
    ):
        self.mode = mode
        self.sigma_lower = max(sigma_lower, 1e-6)
        self.sigma_upper = max(sigma_upper, 1e-6)

    @property
    def skew(self) -> float:
        """σ_U / σ_L ratio (1.0 = symmetric, >1 = upside skew)."""
        return self.sigma_upper / self.sigma_lower

    def cdf(self, x: float) -> float:
        c_L = 2 * self.sigma_lower / (self.sigma_lower + self.sigma_upper)
        c_U = 2 * self.sigma_upper / (self.sigma_lower + self.sigma_upper)

        if SCIPY_AVAILABLE:
            if x < self.mode:
                return c_L * float(scipy_norm.cdf(x, loc=self.mode, scale=self.sigma_lower))
            else:
                return (
                    c_L * 0.5
                    + c_U * float(scipy_norm.cdf(x, loc=self.mode, scale=self.sigma_upper) - 0.5)
                )
        else:
            def _ncdf(z: float) -> float:
                return 0.5 * (1.0 + np.tanh(z * np.sqrt(np.pi / 8)))

            if x < self.mode:
                return c_L * _ncdf((x - self.mode) / self.sigma_lower)
            else:
                return c_L * 0.5 + c_U * (_ncdf((x - self.mode) / self.sigma_upper) - 0.5)

    def ppf(self, q: float) -> float:
        """Inverse CDF (percent-point function)."""
        q = float(np.clip(q, 1e-9, 1 - 1e-9))
        c_L = 2 * self.sigma_lower / (self.sigma_lower + self.sigma_upper)
        c_U = 2 * self.sigma_upper / (self.sigma_lower + self.sigma_upper)
        mode_mass = c_L * 0.5  # CDF value at mode

        if SCIPY_AVAILABLE:
            if q <= mode_mass:
                return float(scipy_norm.ppf(q / c_L, loc=self.mode, scale=self.sigma_lower))
            else:
                q_rescaled = 0.5 + (q - mode_mass) / (c_U * 0.5) * 0.5
                return float(scipy_norm.ppf(q_rescaled, loc=self.mode, scale=self.sigma_upper))
        else:
            def _norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-9, 1 - 1e-9))
                # Rational approximation (Abramowitz & Stegun)
                t = np.sqrt(-2 * np.log(min(p, 1 - p)))
                c = [2.515517, 0.802853, 0.010328]
                d = [1.432788, 0.189269, 0.001308]
                approx = t - (c[0] + c[1] * t + c[2] * t**2) / (
                    1 + d[0] * t + d[1] * t**2 + d[2] * t**3
                )
                return -approx if p < 0.5 else approx

            if q <= mode_mass:
                return self.mode + self.sigma_lower * _norm_ppf(q / c_L)
            else:
                q_rescaled = 0.5 + (q - mode_mass) / (c_U * 0.5) * 0.5
                return self.mode + self.sigma_upper * _norm_ppf(q_rescaled)

    def fan_bands(
        self, prob_intervals: List[float] = [0.3, 0.6, 0.9]
    ) -> List[Tuple[float, float]]:
        """
        Return (lower, upper) bounds for given probability intervals.

        e.g. [0.3, 0.6, 0.9] → 30%, 60%, 90% prediction intervals centred at mode.
        """
        bands = []
        for prob in prob_intervals:
            q_lo = 0.5 - prob / 2
            q_hi = 0.5 + prob / 2
            bands.append((self.ppf(q_lo), self.ppf(q_hi)))
        return bands

    @staticmethod
    def calibrate_from_mc_step(mc: FanChartResult, step: int) -> 'TwoPieceNormal':
        """Fit TwoPieceNormal from MC percentile output at a specific horizon step."""
        mode = float(mc.p50[step])
        p10 = float(mc.p10[step])
        p90 = float(mc.p90[step])
        # 1.28 ≈ z-score for 10th/90th percentile of standard normal
        sigma_lower = max((mode - p10) / 1.28, 0.01)
        sigma_upper = max((p90 - mode) / 1.28, 0.01)
        return TwoPieceNormal(mode=mode, sigma_lower=sigma_lower, sigma_upper=sigma_upper)

    @classmethod
    def boe_fan_chart(
        cls, central: np.ndarray, mc: FanChartResult
    ) -> FanChartResult:
        """
        Build BoE-style asymmetric fan chart by fitting per-step TwoPieceNormal
        to MC output, then re-computing percentile bands.

        Typically σ_U > σ_L because inflation scenarios are upside-skewed.
        """
        horizon = len(central)
        p10 = np.zeros(horizon)
        p25 = np.zeros(horizon)
        p50 = np.zeros(horizon)
        p75 = np.zeros(horizon)
        p90 = np.zeros(horizon)

        for h in range(horizon):
            tpn = cls.calibrate_from_mc_step(mc, h)
            # Apply mild upside skew (inflation risks tend to be asymmetric)
            tpn.sigma_upper = tpn.sigma_upper * 1.25
            p10[h] = tpn.ppf(0.10)
            p25[h] = tpn.ppf(0.25)
            p50[h] = tpn.ppf(0.50)
            p75[h] = tpn.ppf(0.75)
            p90[h] = tpn.ppf(0.90)

        return FanChartResult(
            dates=mc.dates,
            central=central,
            p10=p10,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            simulation_mean=mc.simulation_mean,
            methodology_note=(
                "Bank of England two-piece normal (Britton, Fisher & Whitley 1998); "
                "upside σ inflated 1.25× to reflect asymmetric inflation risks"
            ),
            n_simulations=mc.n_simulations,
        )


# ===========================================================================
# Phase 4 — Country Contribution Analyzer
# ===========================================================================

class CountryContributionAnalyzer:
    """
    NCB-style country-level contribution decomposition.

    Contribution_i = country_weight_i × country_HICP_i
    Cross-check: Σ contributions ≈ EA headline (residual = data quality indicator)
    """

    # Eurostat expenditure share weights (2024 EA composition)
    COUNTRY_WEIGHTS: Dict[str, float] = {
        'DE': 0.269, 'FR': 0.207, 'IT': 0.180, 'ES': 0.122,
        'NL': 0.058, 'BE': 0.034, 'AT': 0.030, 'PT': 0.018,
        'FI': 0.017, 'IE': 0.014, 'HR': 0.009, 'SK': 0.009,
        'LT': 0.006, 'LU': 0.006, 'SI': 0.005, 'LV': 0.004,
        'EE': 0.003, 'CY': 0.003, 'MT': 0.002,
    }

    COUNTRY_NAMES: Dict[str, str] = {
        'DE': 'Germany', 'FR': 'France', 'IT': 'Italy', 'ES': 'Spain',
        'NL': 'Netherlands', 'BE': 'Belgium', 'AT': 'Austria', 'PT': 'Portugal',
        'FI': 'Finland', 'IE': 'Ireland', 'HR': 'Croatia', 'SK': 'Slovakia',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'SI': 'Slovenia', 'LV': 'Latvia',
        'EE': 'Estonia', 'CY': 'Cyprus', 'MT': 'Malta',
    }

    # FRED series IDs for major EA countries
    FRED_COUNTRY_SERIES: Dict[str, str] = {
        'DE': 'CP0000DEW086NEST',
        'FR': 'CP0000FRW086NEST',
        'IT': 'CP0000ITW086NEST',
        'ES': 'CP0000ESW086NEST',
        'NL': 'CP0000NLW086NEST',
    }

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key

    def get_contributions(
        self, country_inflation: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Compute country contributions to EA headline.

        Returns DataFrame with:
          - {country}_contribution columns
          - ea_headline_computed = Σ contributions
          - residual = ea_headline_computed vs official (if provided)
        """
        frames = {
            c: s.rename(c)
            for c, s in country_inflation.items()
            if c in self.COUNTRY_WEIGHTS
        }
        if not frames:
            return pd.DataFrame()

        aligned = pd.concat(frames.values(), axis=1).dropna(how='all')
        result = pd.DataFrame(index=aligned.index)

        for country in aligned.columns:
            w = self.COUNTRY_WEIGHTS.get(country, 0.0)
            result[f'{country}_contribution'] = aligned[country] * w

        result['ea_headline_computed'] = result.sum(axis=1)
        return result

    def fetch_fred_country_data(
        self, start_date: str = '2022-01-01'
    ) -> Dict[str, pd.Series]:
        """Fetch country HICP data from FRED API."""
        if not self.fred_api_key:
            print("FRED API key required for country data fetch.")
            return {}

        country_data: Dict[str, pd.Series] = {}
        for country, series_id in self.FRED_COUNTRY_SERIES.items():
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&api_key={self.fred_api_key}"
                    f"&file_type=json&observation_start={start_date}"
                )
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                obs = resp.json()['observations']
                df = pd.DataFrame(obs)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna()
                country_data[country] = df.set_index('date')['value']
                print(f"Fetched {country} HICP ({len(df)} observations)")
            except Exception as e:
                print(f"Failed to fetch {country} HICP: {e}")

        return country_data

    def summary(self, contributions: pd.DataFrame) -> None:
        """Print contribution table."""
        if contributions.empty:
            return
        latest = contributions.iloc[-1]
        print("\n" + "=" * 55)
        print("COUNTRY CONTRIBUTIONS TO EA HEADLINE HICP")
        print("=" * 55)
        for country in self.COUNTRY_WEIGHTS:
            col = f'{country}_contribution'
            if col in latest:
                name = self.COUNTRY_NAMES.get(country, country)
                w = self.COUNTRY_WEIGHTS[country] * 100
                print(f"  {name:<15} ({w:4.1f}%): {latest[col]:+.3f}pp")
        total = latest.get('ea_headline_computed', 0)
        print(f"\n  EA Computed:        {total:.2f}%")
        print("=" * 55)


# ===========================================================================
# Commodity Data Fetcher
# ===========================================================================

class CommodityFetcher:
    """
    Fetches commodity price data for use as exogenous regressors in ARIMAX.

    Two-source strategy (no paid vendor required):
      1. FRED API (if api_key provided): Brent crude, EU natural gas, food index, EUR/USD
      2. yfinance (free, no key): Brent futures, EUR/USD, wheat — automatic fallback

    Also builds the forward exogenous path for forecasting by extrapolating
    recent commodity momentum (with mean-reversion decay).

    Cache: ~/.inflation_cache/commodity_data.parquet  (24-hour TTL)
    """

    CACHE_FILE = Path.home() / '.inflation_cache' / 'commodity_data.parquet'

    # FRED series: Brent (daily→monthly avg), EU gas (monthly), food index (monthly), EUR/USD (daily→monthly)
    FRED_SERIES: Dict[str, str] = {
        'brent':      'DCOILBRENTEU',   # Brent crude, USD/barrel
        'eu_gas':     'PNGASEUUSDM',    # European natural gas, USD/MMBtu
        'food_index': 'PFOODINDEXM',    # World Bank food price index
        'eurusd':     'DEXUSEU',        # EUR/USD spot rate
    }

    # yfinance tickers (fallback — no API key needed)
    YFINANCE_TICKERS: Dict[str, str] = {
        'brent':  'BZ=F',       # Brent crude front-month futures
        'eu_gas': 'NG=F',       # Henry Hub natural gas (US proxy for EU gas)
        'eurusd': 'EURUSD=X',   # EUR/USD spot
        'wheat':  'ZW=F',       # CBOT wheat futures
    }

    def fetch(self, fred_api_key: Optional[str] = None,
              start_date: str = '2018-01-01') -> pd.DataFrame:
        """
        Return monthly commodity DataFrame.
        Columns: brent, eu_gas, food_index (or wheat), eurusd.
        """
        if self._is_cache_valid():
            cached = self._load_cache()
            if cached is not None and not cached.empty:
                print(f"Commodity data: loaded from cache ({len(cached)} months)")
                return cached

        df = None
        if fred_api_key:
            df = self._fetch_fred(fred_api_key, start_date)
            if df is not None and not df.empty:
                print(f"Commodity data: fetched from FRED ({list(df.columns)})")

        if df is None or df.empty:
            df = self._fetch_yfinance(start_date)
            if df is not None and not df.empty:
                print(f"Commodity data: fetched from yfinance ({list(df.columns)})")

        if df is not None and not df.empty:
            self._save_cache(df)
            return df

        print("Commodity data unavailable — ARIMAX will fall back to plain ARIMA.")
        return pd.DataFrame()

    def get_snapshot(self, df: pd.DataFrame) -> CommoditySnapshot:
        """Extract current levels and recent momentum for display."""
        if df.empty or len(df) < 2:
            return CommoditySnapshot()
        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        def pct_chg(col: str) -> Optional[float]:
            if col in df.columns and prev[col] and prev[col] != 0:
                return round((latest[col] - prev[col]) / abs(prev[col]) * 100, 2)
            return None

        return CommoditySnapshot(
            brent_usd    = round(float(latest['brent']), 2)      if 'brent'      in df.columns else None,
            eu_gas_usd   = round(float(latest['eu_gas']), 2)     if 'eu_gas'     in df.columns else None,
            food_index   = round(float(latest.get('food_index', latest.get('wheat', 0))), 2),
            eurusd       = round(float(latest['eurusd']), 4)     if 'eurusd'     in df.columns else None,
            brent_mom_pct = pct_chg('brent'),
            gas_mom_pct   = pct_chg('eu_gas'),
            as_of        = str(df.index[-1].date()),
            source       = 'fred' if 'food_index' in df.columns else 'yfinance',
        )

    def build_exog_matrix(self, df: pd.DataFrame,
                          target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Build aligned lagged % change matrix for ARIMAX fitting.

        Columns returned:
          d_brent_lag1:  1-month lagged % change in Brent (energy pass-through)
          d_eurusd_lag1: 1-month lagged % change in EUR/USD (import price channel)

        Both are stationary (already % changes) and enter ARIMAX directly.
        """
        exog = pd.DataFrame(index=target_index)

        for col, lag, out_name in [
            ('brent',  1, 'd_brent_lag1'),
            ('eurusd', 1, 'd_eurusd_lag1'),
        ]:
            if col not in df.columns:
                continue
            pct = df[col].pct_change() * 100
            aligned = pct.reindex(target_index, method='nearest')
            exog[out_name] = aligned.shift(lag)

        return exog

    def build_forecast_exog(self, df: pd.DataFrame,
                            horizon: int) -> Optional[np.ndarray]:
        """
        Build forward exog matrix for ARIMAX forecasting.

        Strategy:
          - Fetch Brent futures momentum via yfinance if available
          - Otherwise extrapolate recent 3-month trend with 0.7× decay per step
          - EUR/USD: mean of last 6 months of % changes (slow-moving)
        """
        if df.empty:
            return None

        cols = []

        # Brent forward path
        if 'brent' in df.columns:
            brent_fwd = self._brent_forward_path(df['brent'], horizon)
            cols.append(('d_brent_lag1', brent_fwd))

        # EUR/USD forward path: mean-revert to recent average
        if 'eurusd' in df.columns:
            recent_chgs = df['eurusd'].pct_change().dropna().values[-6:] * 100
            mean_chg = float(np.mean(recent_chgs)) if len(recent_chgs) else 0.0
            eurusd_fwd = np.array([mean_chg * (0.7 ** h) for h in range(horizon)])
            cols.append(('d_eurusd_lag1', eurusd_fwd))

        if not cols:
            return None

        return np.column_stack([v for _, v in cols])

    # ------------------------------------------------------------------
    def _fetch_fred(self, api_key: str, start_date: str) -> Optional[pd.DataFrame]:
        dfs: Dict[str, pd.Series] = {}
        for name, sid in self.FRED_SERIES.items():
            try:
                url = (f"https://api.stlouisfed.org/fred/series/observations"
                       f"?series_id={sid}&api_key={api_key}"
                       f"&file_type=json&observation_start={start_date}")
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                raw = pd.DataFrame(resp.json()['observations'])
                raw['date']  = pd.to_datetime(raw['date'])
                raw['value'] = pd.to_numeric(raw['value'], errors='coerce')
                raw = raw.dropna()
                monthly = raw.set_index('date')['value'].resample('MS').mean()
                dfs[name] = monthly
            except Exception as e:
                print(f"  FRED {name} ({sid}) failed: {e}")
        return pd.concat(dfs, axis=1) if dfs else None

    def _fetch_yfinance(self, start_date: str) -> Optional[pd.DataFrame]:
        if not YFINANCE_AVAILABLE:
            print("yfinance not installed. Run: pip install yfinance")
            return None
        dfs: Dict[str, pd.Series] = {}
        for name, ticker in self.YFINANCE_TICKERS.items():
            try:
                raw = yf.download(ticker, start=start_date,
                                  progress=False, auto_adjust=True)
                if raw.empty:
                    continue
                close = raw['Close']
                if hasattr(close, 'squeeze'):
                    close = close.squeeze()
                monthly = close.resample('MS').mean()
                monthly.index = monthly.index.tz_localize(None)
                dfs[name] = monthly
            except Exception as e:
                print(f"  yfinance {ticker} failed: {e}")
        return pd.concat(dfs, axis=1) if dfs else None

    def _brent_forward_path(self, brent_series: pd.Series,
                            horizon: int) -> np.ndarray:
        """
        Estimate forward monthly % changes for Brent.
        Uses yfinance for live front-month momentum, then applies decay.
        """
        # Try live front-month price from yfinance
        current_spot = None
        if YFINANCE_AVAILABLE:
            try:
                spot = yf.download('BZ=F', period='5d',
                                   progress=False, auto_adjust=True)
                if not spot.empty:
                    current_spot = float(spot['Close'].iloc[-1])
            except Exception:
                pass

        # Recent 3-month momentum in % change terms
        recent_chgs = brent_series.pct_change().dropna().values[-3:] * 100
        momentum = float(np.mean(recent_chgs)) if len(recent_chgs) else 0.0

        # If live spot available and above recent monthly avg, boost momentum
        if current_spot is not None and len(brent_series) >= 1:
            last_monthly = float(brent_series.iloc[-1])
            spot_pct_above = (current_spot - last_monthly) / last_monthly * 100
            # Blend: 50% historical momentum + 50% spot signal
            momentum = 0.5 * momentum + 0.5 * spot_pct_above

        # Decay momentum toward zero over the horizon (mean reversion)
        return np.array([momentum * (0.7 ** h) for h in range(horizon)])

    def _is_cache_valid(self) -> bool:
        if not self.CACHE_FILE.exists():
            return False
        age_hours = (datetime.now().timestamp() - self.CACHE_FILE.stat().st_mtime) / 3600
        return age_hours < 24

    def _load_cache(self) -> Optional[pd.DataFrame]:
        try:
            return pd.read_parquet(self.CACHE_FILE)
        except Exception:
            return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        try:
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.CACHE_FILE)
        except Exception as e:
            print(f"Commodity cache write failed (non-fatal): {e}")


# ===========================================================================
# Updated Main Analyzer Class
# ===========================================================================

class EurozoneInflationAnalyzer:
    """
    Official ECB/Eurostat HICP Methodology Implementation — Overhauled.

    Now includes:
    - Dynamic weight fetching (Eurostat → ECB → fallback)
    - Ensemble forecasting (Exp Smoothing + ARIMA + ARIMAX, inverse-RMSE weighted)
    - CommodityFetcher: Brent, EU gas, food index, EUR/USD via FRED + yfinance
    - Monte Carlo fan charts (10k paths, headline and component-level)
    - Bank of England two-piece normal methodology
    - Country contribution decomposition
    """

    # FRED Series IDs for Eurozone HICP Components
    FRED_SERIES: Dict[str, str] = {
        'headline': 'CP0000EZ19M086NEST',
        'core': 'CPHPILFE',
        'energy': 'CP0500EZ19M086NEST',
        'food': 'CP0111EZ19M086NEST',
        'services': 'CP0600EZ19M086NEST',
    }

    def __init__(self, fred_api_key: Optional[str] = None, n_mc_simulations: int = 10_000):
        self.fred_api_key = fred_api_key
        self.n_mc_simulations = n_mc_simulations
        self.data: Dict = {}
        self.forecasts: Dict = {}

        # Dynamic weight fetching
        fetcher = HICPWeightFetcher()
        raw = fetcher.fetch()
        self.HICP_WEIGHTS: Dict[str, float] = {
            'services': raw['services'],
            'non_energy_goods': raw['non_energy_goods'],
            'food_alcohol_tobacco': raw['food_alcohol_tobacco'],
            'energy': raw['energy'],
        }
        self._weight_source: str = raw.get('source', 'unknown')

        # Build ensemble: Exp Smoothing + ARIMA + ARIMAX (commodity-aware)
        models: List[BaseInflationModel] = [ExponentialSmoothingModel()]
        if STATSMODELS_AVAILABLE:
            models.append(ARIMAModel())
            models.append(ARIMAXModel())   # uses exog when commodity data available
        self.ensemble = EnsembleForecaster(models)

        # Monte Carlo engine
        self.mc_engine = MonteCarloEngine(n_simulations=n_mc_simulations)

        # Country contribution analyzer
        self.country_analyzer = CountryContributionAnalyzer(fred_api_key)

        # Commodity fetcher and cached data
        self.commodity_fetcher = CommodityFetcher()
        self._commodity_df: pd.DataFrame = pd.DataFrame()
        self._commodity_snapshot: Optional[CommoditySnapshot] = None

    # ------------------------------------------------------------------
    # Data fetching (backwards compatible)
    # ------------------------------------------------------------------

    def fetch_fred_data(self, series_id: str, start_date: str = '2018-01-01') -> Optional[pd.DataFrame]:
        if not self.fred_api_key:
            raise ValueError(
                "FRED API key required. Get free key at https://fred.stlouisfed.org/docs/api/"
            )
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={self.fred_api_key}"
            f"&file_type=json&observation_start={start_date}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            obs = resp.json()['observations']
            df = pd.DataFrame(obs)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna()
            return df[['date', 'value']].sort_values('date')
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return None

    def fetch_commodity_data(self, start_date: str = '2018-01-01') -> pd.DataFrame:
        """
        Fetch commodity price data and store for use in ARIMAX forecasting.

        Call this once before forecast_with_uncertainty() to enable the
        commodity shock channel.  Works with or without a FRED API key:
          - With key:    FRED (Brent, EU gas, food index, EUR/USD)
          - Without key: yfinance (Brent futures, EUR/USD, wheat)

        Returns the commodity DataFrame and prints a snapshot of current levels.
        """
        self._commodity_df = self.commodity_fetcher.fetch(
            fred_api_key=self.fred_api_key,
            start_date=start_date,
        )
        if not self._commodity_df.empty:
            self._commodity_snapshot = self.commodity_fetcher.get_snapshot(
                self._commodity_df
            )
            snap = self._commodity_snapshot
            print("\n" + "-" * 55)
            print("COMMODITY SNAPSHOT")
            if snap.brent_usd:
                chg = f" ({snap.brent_mom_pct:+.1f}% MoM)" if snap.brent_mom_pct else ""
                print(f"  Brent crude:      ${snap.brent_usd:.1f}/bbl{chg}")
            if snap.eu_gas_usd:
                chg = f" ({snap.gas_mom_pct:+.1f}% MoM)" if snap.gas_mom_pct else ""
                print(f"  EU natural gas:   ${snap.eu_gas_usd:.2f}/MMBtu{chg}")
            if snap.eurusd:
                print(f"  EUR/USD:          {snap.eurusd:.4f}")
            if snap.food_index:
                print(f"  Food price index: {snap.food_index:.1f}")
            print(f"  Commodity pressure signal: {snap.commodity_pressure().upper()}")
            print(f"  As of: {snap.as_of}  |  Source: {snap.source}")
            print("-" * 55)
        return self._commodity_df

    # ------------------------------------------------------------------
    # HICP calculation (backwards compatible)
    # ------------------------------------------------------------------

    def calculate_hicp_inflation(self, components_df: pd.DataFrame) -> pd.DataFrame:
        result = components_df.copy()
        w = self.HICP_WEIGHTS
        result['hicp_headline'] = (
            result.get('services', 0) * w['services']
            + result.get('non_energy_goods', 0) * w['non_energy_goods']
            + result.get('food_alcohol_tobacco', 0) * w['food_alcohol_tobacco']
            + result.get('energy', 0) * w['energy']
        )
        core_w = w['services'] + w['non_energy_goods']
        result['hicp_core'] = (
            result.get('services', 0) * w['services']
            + result.get('non_energy_goods', 0) * w['non_energy_goods']
        ) / core_w
        return result

    def calculate_contributions(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for comp, weight in self.HICP_WEIGHTS.items():
            if comp in result.columns:
                result[f'{comp}_contribution'] = result[comp] * weight
        return result

    # ------------------------------------------------------------------
    # Deterministic forecasting (backwards compatible)
    # ------------------------------------------------------------------

    def exponential_smoothing_forecast(
        self, series: pd.Series, periods: int = 6, alpha: float = 0.3, trend_weight: float = 0.1
    ) -> pd.Series:
        model = ExponentialSmoothingModel(alpha=alpha, trend_weight=trend_weight)
        model.fit(series)
        result = model.forecast(periods)
        return pd.Series(result.central)

    def arima_forecast(
        self, series: pd.Series, periods: int = 6, order: Tuple = (1, 1, 1)
    ) -> pd.Series:
        if not STATSMODELS_AVAILABLE:
            return self.exponential_smoothing_forecast(series, periods)
        try:
            model = ARIMAModel(order=order)
            model.fit(series)
            result = model.forecast(periods)
            return pd.Series(result.central)
        except Exception as e:
            print(f"ARIMA failed: {e}. Using exponential smoothing.")
            return self.exponential_smoothing_forecast(series, periods)

    def forecast_inflation(
        self, data: pd.DataFrame, method: str = 'ensemble', periods: int = 6
    ) -> pd.DataFrame:
        """
        Generate inflation forecasts (backwards compatible, now defaults to ensemble).

        method: 'ensemble' | 'arima' | 'exponential'
        """
        last_date = data['date'].max()
        forecast_dates = [last_date + timedelta(days=30 * (i + 1)) for i in range(periods)]
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        forecast_data: List[Dict] = []

        for col in numeric_cols:
            series = data[col].dropna()
            if method == 'ensemble':
                vals, _ = self.ensemble.fit_and_forecast(series, periods)
                forecast_vals = pd.Series(vals)
            elif method == 'arima':
                forecast_vals = self.arima_forecast(series, periods)
            else:
                forecast_vals = self.exponential_smoothing_forecast(series, periods)

            for i, date in enumerate(forecast_dates):
                if i >= len(forecast_data):
                    forecast_data.append({'date': date})
                forecast_data[i][col] = (
                    float(forecast_vals.iloc[i]) if i < len(forecast_vals) else np.nan
                )

        forecast_df = pd.DataFrame(forecast_data)
        forecast_df['is_forecast'] = True
        results = data.copy()
        results['is_forecast'] = False
        return pd.concat([results, forecast_df], ignore_index=True)

    # ------------------------------------------------------------------
    # New: Probabilistic fan chart
    # ------------------------------------------------------------------

    def forecast_with_uncertainty(
        self,
        data: pd.DataFrame,
        periods: int = 6,
        methodology: str = 'ecb',
        component_level: bool = False,
    ) -> Dict:
        """
        Generate probabilistic fan chart forecast.

        Parameters:
            data: DataFrame with component columns and 'date'
            periods: forecast horizon in months
            methodology: 'ecb' (symmetric MC) | 'boe' (two-piece normal)
            component_level: if True, use component-level MC (requires all 4 components)

        Returns:
            dict in FanChartResult format (JSON-serializable)
        """
        if 'hicp_headline' not in data.columns:
            data = self.calculate_hicp_inflation(data)

        headline = data['hicp_headline'].dropna()

        # Build commodity exog matrices if data is available
        exog: Optional[pd.DataFrame] = None
        forecast_exog: Optional[np.ndarray] = None
        if not self._commodity_df.empty:
            target_idx = pd.DatetimeIndex(pd.to_datetime(data['date']))
            exog = self.commodity_fetcher.build_exog_matrix(
                self._commodity_df, target_idx
            )
            forecast_exog = self.commodity_fetcher.build_forecast_exog(
                self._commodity_df, periods
            )

        # Ensemble forecast (ARIMAX uses commodity exog when available)
        central, residuals = self.ensemble.fit_and_forecast(
            headline, horizon=periods, exog=exog, forecast_exog=forecast_exog
        )

        # Forecast dates
        last_date = data['date'].max()
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        dates = [
            (last_date + pd.DateOffset(months=i + 1)).strftime('%Y-%m')
            for i in range(periods)
        ]

        # Component-level MC if requested
        components = ['services', 'non_energy_goods', 'food_alcohol_tobacco', 'energy']
        if component_level and all(c in data.columns for c in components):
            comp_forecasts: Dict[str, np.ndarray] = {}
            comp_history: Dict[str, pd.Series] = {}
            for c in components:
                series = data[c].dropna()
                fc, _ = self.ensemble.fit_and_forecast(series, horizon=periods)
                comp_forecasts[c] = fc
                comp_history[c] = series
            mc_result = self.mc_engine.run_component_level(
                comp_forecasts, comp_history, self.HICP_WEIGHTS, periods
            )
        else:
            mc_result = self.mc_engine.run_headline_level(central, residuals, periods)

        mc_result.dates = dates

        # Apply BoE two-piece normal if requested
        if methodology == 'boe':
            mc_result = TwoPieceNormal.boe_fan_chart(central, mc_result)

        mc_result.methodology_note += f" | weights source: {self._weight_source}"
        return mc_result.to_dict()

    # ------------------------------------------------------------------
    # Statistics & reporting (backwards compatible)
    # ------------------------------------------------------------------

    def get_statistics(self, data: pd.DataFrame, target: float = 2.0) -> Dict:
        if data.empty:
            return {}
        stats: Dict = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 2:
                continue
            stats[col] = {
                'current': float(series.iloc[-1]),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'vs_target': float(series.iloc[-1] - target),
                'trend': 'up' if series.iloc[-1] > series.iloc[-3] else 'down',
            }
        return stats

    def print_summary(self, data: pd.DataFrame, target: float = 2.0) -> None:
        stats = self.get_statistics(data, target)
        print("\n" + "=" * 60)
        print("EUROZONE INFLATION ANALYSIS SUMMARY")
        print(f"  Weight source: {self._weight_source}")
        print("=" * 60)

        if 'hicp_headline' in stats:
            s = stats['hicp_headline']
            print(f"\nHeadline HICP:  {s['current']:.2f}%")
            print(f"  ECB Target:   {target:.2f}%")
            print(f"  Deviation:    {s['vs_target']:+.2f}%")
            print(f"  Trend:        {'↑ Rising' if s['trend'] == 'up' else '↓ Falling'}")

        if 'hicp_core' in stats:
            print(f"\nCore HICP:      {stats['hicp_core']['current']:.2f}%")

        print("\n" + "-" * 60)
        print("COMPONENT WEIGHTS")
        for comp, weight in self.HICP_WEIGHTS.items():
            print(f"  {comp.replace('_', ' ').title():<30} {weight*100:.1f}%")

        print("\n" + "-" * 60)
        print("ENSEMBLE MODEL WEIGHTS")
        ew = self.ensemble.model_weights
        if not ew and 'hicp_headline' in data.columns:
            # Fit ensemble on the fly to display weights
            self.ensemble.fit_and_forecast(data['hicp_headline'].dropna(), horizon=1)
            ew = self.ensemble.model_weights
        for model_name, w in ew.items():
            print(f"  {model_name:<30} {w*100:.1f}%")


# ===========================================================================
# Example usage
# ===========================================================================

def example_usage():
    print("\n" + "=" * 60)
    print("EUROZONE INFLATION ANALYZER — OVERHAULED EXAMPLE")
    print("=" * 60)

    analyzer = EurozoneInflationAnalyzer()

    # Jan 2023 – Feb 2026 (38 months): includes 2025 Middle East oil shock
    # Energy reverses from -4% (Dec 2024) to +2.5% by early 2026
    # Services remain sticky ~4%; headline re-accelerates to ~2.7-2.8%
    dates = pd.date_range('2023-01-01', periods=38, freq='MS')
    sample_data = pd.DataFrame({
        'date': dates,
        'services': [
            # 2023
            4.8, 4.9, 5.0, 4.8, 4.7, 4.6, 4.4, 4.2, 4.1, 4.0, 4.1, 4.2,
            # 2024
            4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.5, 3.4, 3.5, 3.4, 3.2,
            # 2025: sticky, re-accelerates on persistent wage growth
            4.0, 3.9, 3.8, 3.9, 4.0, 4.1, 4.2, 4.1, 4.0, 4.1, 4.2, 4.3,
            # 2026 Jan-Feb
            4.4, 4.3,
        ],
        'non_energy_goods': [
            # 2023
            2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1,
            # 2024
            1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.4,
            # 2025: stabilises at low positive level
            0.6, 0.5, 0.4, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.5, 0.6,
            # 2026 Jan-Feb
            0.6, 0.5,
        ],
        'food_alcohol_tobacco': [
            # 2023
            4.5, 4.6, 4.7, 4.5, 4.3, 4.1, 3.9, 3.7, 3.5, 3.4, 3.2, 3.1,
            # 2024
            3.0, 3.1, 3.2, 3.3, 3.4, 3.2, 3.1, 3.0, 2.8, 2.6, 2.5, 2.6,
            # 2025: ticks up as energy-intensive food production costs rise
            2.7, 2.6, 2.5, 2.6, 2.5, 2.7, 2.8, 2.7, 2.6, 2.7, 2.8, 2.9,
            # 2026 Jan-Feb
            3.0, 2.9,
        ],
        'energy': [
            # 2023
            3.2,  2.8,  2.0,  0.5, -1.5, -2.3, -1.5, -3.2, -2.8, -1.5,  0.2, -0.5,
            # 2024: deeply negative as 2022 price spike drops out of base
            -2.0, -1.5, -2.5, -3.2, -3.0, -2.0, -1.0, -0.8, -1.5, -2.0, -1.9, -4.0,
            # 2025: base effect fades, Brent rises on Middle East tensions
            -1.0, -0.5,  0.0,  0.5,  0.0,  1.5,  2.0,  1.5,  1.0,  2.0,  2.0,  2.0,
            # 2026 Jan-Feb: sustained upside
            2.5,  2.0,
        ],
    })

    hicp_data = analyzer.calculate_hicp_inflation(sample_data)
    contrib_data = analyzer.calculate_contributions(hicp_data)
    analyzer.print_summary(contrib_data)

    print("\n" + "-" * 60)
    print("ENSEMBLE FORECAST (Next 6 Months)")
    forecast = analyzer.forecast_inflation(contrib_data, method='ensemble', periods=6)
    forecast_only = forecast[forecast['is_forecast'] == True].head(6)
    for _, row in forecast_only.iterrows():
        print(f"  {row['date'].strftime('%Y-%m')}: Headline {row['hicp_headline']:.2f}%")

    print("\n" + "-" * 60)
    print("MONTE CARLO FAN CHART (ECB methodology)")
    fan = analyzer.forecast_with_uncertainty(hicp_data, periods=6, methodology='ecb')
    print(f"  Dates:  {fan['dates']}")
    print(f"  p10:    {[f'{v:.2f}' for v in fan['p10']]}")
    print(f"  p50:    {[f'{v:.2f}' for v in fan['p50']]}")
    print(f"  p90:    {[f'{v:.2f}' for v in fan['p90']]}")
    assert all(
        fan['p10'][i] < fan['p25'][i] < fan['p50'][i] < fan['p75'][i] < fan['p90'][i]
        for i in range(len(fan['p10']))
    ), "Fan chart ordering violated!"
    print("  ✓ p10 < p25 < p50 < p75 < p90 verified")

    print("\n" + "-" * 60)
    print("BANK OF ENGLAND TWO-PIECE NORMAL")
    fan_boe = analyzer.forecast_with_uncertainty(hicp_data, periods=6, methodology='boe')
    for i, d in enumerate(fan_boe['dates']):
        spread_up = fan_boe['p90'][i] - fan_boe['p50'][i]
        spread_dn = fan_boe['p50'][i] - fan_boe['p10'][i]
        print(f"  {d}: p50={fan_boe['p50'][i]:.2f}  ↑{spread_up:.2f} ↓{spread_dn:.2f}  "
              f"{'(upside wider ✓)' if spread_up > spread_dn else ''}")

    print("\n" + "-" * 60)
    print("COMMODITY-AWARE FORECAST (yfinance, no API key needed)")
    analyzer.fetch_commodity_data()   # pulls Brent, EUR/USD via yfinance
    if not analyzer._commodity_df.empty:
        snap = analyzer._commodity_snapshot
        if snap:
            print(f"  Commodity pressure: {snap.commodity_pressure().upper()}")
        fan_comm = analyzer.forecast_with_uncertainty(hicp_data, periods=6)
        print("  Commodity-aware p50 vs baseline p50:")
        for i, d in enumerate(fan_comm['dates']):
            diff = fan_comm['p50'][i] - fan['p50'][i]
            print(f"    {d}: {fan_comm['p50'][i]:.2f}%  (Δ {diff:+.2f}pp vs no-commodity baseline)")
    else:
        print("  Install yfinance to enable: pip install yfinance")

    print("\n" + "=" * 60)
    print("Run complete. See forecast_with_uncertainty() for JSON output.")


if __name__ == '__main__':
    example_usage()
