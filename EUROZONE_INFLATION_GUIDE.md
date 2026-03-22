# Eurozone Inflation Analysis Tool
## Full Overhaul — ECB Suite-of-Models + Monte Carlo + Bank of England Methodology

---

## What's New in This Version

| Feature | Before | After |
|---------|--------|-------|
| **Weights** | Hardcoded 2025 values | Dynamic Eurostat SDMX-JSON fetch with annual cache invalidation |
| **Forecasting** | Single Exp Smoothing | Ensemble (Exp Smoothing + ARIMA), inverse-RMSE weighted |
| **Uncertainty** | None | 10,000-path Monte Carlo fan charts |
| **BoE methodology** | None | Two-piece normal (Britton, Fisher & Whitley 1998) |
| **Country breakdown** | None | NCB-style country contribution decomposition |

---

## Official ECB HICP Methodology

### Component Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| **Services** | 45.7% | Largest component — rents, insurance, haircuts, etc. |
| **Non-Energy Industrial Goods** | 25.6% | Manufactured goods — cars, clothing, appliances |
| **Food, Alcohol & Tobacco** | 19.3% | Food & beverage, volatile |
| **Energy** | 9.4% | Fuel, electricity — most volatile |

**Headline HICP** = Services×0.457 + NEIG×0.256 + Food×0.193 + Energy×0.094

**Core HICP** = (Services×0.457 + NEIG×0.256) / 0.713

**ECB Target**: 2% over the medium term

---

## Phase 1 — Dynamic Weight Fetching (`HICPWeightFetcher`)

Weights update every January when the ECB publishes the annual HICP weight revision.

### Three-Layer Fallback

```
Layer 1: Eurostat SDMX-JSON
  GET https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_inw
      ?geo=EA&unit=W_PPT&coicop=CP00,SERV,NEIG,FOOD,NRG&sinceTimePeriod=2024&format=JSON

Layer 2: ECB Data Portal (secondary — limited public weight endpoint)

Layer 3: Hardcoded 2025 ECB weights (tertiary fallback)
```

### Cache Design

- Stored at `~/.inflation_cache/hicp_weights.json`
- **Cache invalidation rule**: `month == 1 AND year > cached_year`
  → Re-fetches in January of every new year (ECB annual weight update)

### SDMX-JSON Parsing

The Eurostat SDMX-JSON response uses a sparse flat-index array:

```python
flat_index = i0 * (s1*s2*…*sN) + i1 * (s2*…*sN) + … + iN

# Dimension order read from response["id"], NOT assumed
dim_ids = response["id"]   # e.g. ["FREQ", "UNIT", "COICOP", "GEO", "TIME_PERIOD"]
sizes   = response["size"] # e.g. [1, 1, 5, 1, 2]
values  = response["value"] # sparse dict: {"0": 1000, "4": 456, …}
```

The fetcher handles arbitrary dimension order and sparse missing values gracefully.

---

## Phase 2 — Ensemble Forecasting

### Model Suite

All models implement `BaseInflationModel` ABC with `fit(series)` and `forecast(horizon)`.

#### `ExponentialSmoothingModel(alpha=0.3)`

```
Level(t+1) = α × y(t) + (1-α) × Level(t)
Forecast(h) = Level + trend_weight × trend × h
```

- α = 0.3 (smoothing parameter)
- Trend computed from last 6 months
- Returns `residuals` = actual − fitted (used for MC calibration)

#### `ARIMAModel(order=(1,1,1))`

- Requires `statsmodels`
- Stores `model.resid` as residuals for MC calibration
- Falls back to Exp Smoothing if statsmodels unavailable

### `EnsembleForecaster` — Inverse-RMSE Weighting

```python
weight_i = (1 / RMSE_i) / Σ(1 / RMSE_j)
```

Lower RMSE → higher weight. Uses last N in-sample periods as pseudo-validation.
Concatenated residuals from all models feed the Monte Carlo engine.

---

## Phase 3 — Monte Carlo Engine

### Headline-Level Simulation

```
For i = 1..10,000:
  For h = 1..horizon:
    innovation_h ~ N(0, σ²)        # σ estimated from ensemble residuals
    path[i, h] = central[h]
               + decay × (path[i, h-1] − central[h-1])   # AR(1) persistence
               + innovation_h
```

- `decay = 0.7` — uncertainty persists across horizon (wider bands over time)
- `σ = std(residuals)` — calibrated from ensemble in-sample residuals

### Component-Level Simulation (Preferred)

Requires all 4 HICP components in the dataset:

```
Estimate covariance matrix Σ of monthly component changes
  → Ledoit-Wolf shrinkage if sklearn available, else sample covariance
  → Diagonal regularization: Σ += 1e-6 × I (for near-singular matrices)

For i = 1..10,000:
  For h = 1..horizon:
    innovations[h] ~ MVN(0, Σ)    # multivariate draw
    comp_paths[h]  = comp_central[h] + innovations[h]
    headline[h]    = Σ (comp_paths[h] × weight)
```

### Output Schema

```json
{
  "dates": ["2025-04", "2025-05", …],
  "central": [2.3, 2.2, …],
  "p10": [1.1, 0.9, …],
  "p25": [1.7, 1.5, …],
  "p50": [2.3, 2.2, …],
  "p75": [2.9, 2.8, …],
  "p90": [3.6, 3.5, …],
  "simulation_mean": [2.31, 2.21, …],
  "methodology_note": "ECB suite-of-models…",
  "n_simulations": 10000
}
```

**Verification**: `p10 < p25 < p50 < p75 < p90` for all horizon steps (asserted in `example_usage()`).

---

## Phase 3b — Bank of England Two-Piece Normal

### Theory

Reference: **Britton, Fisher & Whitley (1998)**, Bank of England Quarterly Bulletin.

The two-piece normal has mode μ, lower standard deviation σ_L, upper standard deviation σ_U:

```
f(x) = c × N(μ, σ_L)   for x < μ
f(x) = c × N(μ, σ_U)   for x ≥ μ

where c = 2 / (σ_L + σ_U)   (normalizing constant)
```

**CDF**:
```
F(x) = c_L × Φ((x-μ)/σ_L)                           for x < μ
F(x) = c_L/2 + c_U × [Φ((x-μ)/σ_U) - 0.5]          for x ≥ μ

c_L = 2σ_L/(σ_L + σ_U),  c_U = 2σ_U/(σ_L + σ_U)
```

### Calibration from Monte Carlo

```python
mode     = p50[h]
σ_L      = (p50[h] - p10[h]) / 1.28      # 1.28 ≈ z-score of 10th percentile
σ_U      = (p90[h] - p50[h]) / 1.28
σ_U     *= 1.25                           # 25% upside inflation risk premium
```

The 1.25× multiplier reflects ECB survey evidence that inflation risks are upside-skewed during
disinflation episodes (services stickiness, energy rebound).

### Fan Chart Bands (fan_bands method)

```python
tpn = TwoPieceNormal(mode=2.3, sigma_lower=0.4, sigma_upper=0.5)
bands = tpn.fan_bands([0.3, 0.6, 0.9])
# → [(lower_30, upper_30), (lower_60, upper_60), (lower_90, upper_90)]
```

Visual result: upper half of bands is wider than lower half.

---

## Phase 4 — NCB Country Contributions

### Country Weights (Eurostat EA Expenditure Shares, 2024)

| Country | Weight | Country | Weight |
|---------|--------|---------|--------|
| Germany | 26.9%  | Netherlands | 5.8% |
| France  | 20.7%  | Belgium     | 3.4% |
| Italy   | 18.0%  | Austria     | 3.0% |
| Spain   | 12.2%  | Other EA    | ~10% |

### Contribution Calculation

```
contribution_i = country_weight_i × country_HICP_i

EA headline ≈ Σ contribution_i
residual    = EA headline − Σ contribution_i   (data quality indicator, target < 0.1pp)
```

### Fetching Country Data (FRED API)

```python
analyzer = EurozoneInflationAnalyzer(fred_api_key='YOUR_KEY')
country_data = analyzer.country_analyzer.fetch_fred_country_data(start_date='2022-01-01')
contributions = analyzer.country_analyzer.get_contributions(country_data)
analyzer.country_analyzer.summary(contributions)
```

---

## React Frontend — Fan Chart Implementation

### Recharts Stacked-Area Pattern

Fan chart bands use the Recharts `stackId` pattern where two `<Area>` components per band
render an invisible baseline then a visible width:

```jsx
{/* Outer band (p10 → p90) */}
<Area stackId="outer" dataKey="outerLow"   fill="transparent" stroke="none" />  {/* invisible */}
<Area stackId="outer" dataKey="outerWidth" fill="#06b6d4" fillOpacity={0.12} />  {/* visible */}

{/* Inner band (p25 → p75) */}
<Area stackId="inner" dataKey="innerLow"   fill="transparent" stroke="none" />
<Area stackId="inner" dataKey="innerWidth" fill="#06b6d4" fillOpacity={0.22} />

{/* Central forecast */}
<Line dataKey="central" stroke="#1e40af" strokeWidth={2} />

{/* Median — shows skew vs central in BoE mode */}
<Line dataKey="p50" stroke="#a855f7" strokeDasharray="5 4" />
```

**Data structure** (forecast rows only):
```js
{
  date: "2025-04",
  central: 2.3,          // ensemble central forecast
  p50: 2.31,             // median (may differ from central if skewed)
  outerLow: 1.1,         // p10 — invisible baseline
  outerWidth: 2.5,       // p90 - p10 — visible band
  innerLow: 1.7,         // p25 — invisible baseline
  innerWidth: 1.2,       // p75 - p25 — visible band
}
```

### Methodology Selector

Three modes accessible via toggle buttons:

| Mode | Description |
|------|-------------|
| **ECB** | Symmetric MC fan chart (σ_L = σ_U), ECB suite-of-models |
| **BoE** | Two-piece normal (σ_U = 1.25 × σ_L), asymmetric upside |
| **NCB** | Country contribution stacked bar chart |

---

## Verification Checklist

1. **Weight source**: output includes `weight_source: 'eurostat'` or `'hardcoded_2025'`
2. **MC ordering**: `p10 < p25 < p50 < p75 < p90` for all horizon steps
3. **p50 ≈ central**: median within ~0.1% of ensemble central (symmetric case)
4. **BoE asymmetry**: `(p90 − p50) > (p50 − p10)` for all horizon steps (upside wider)
5. **Country residual**: `|ea_headline_computed − ea_headline| < 0.1pp`
6. **React fan chart**: shaded nested bands visible; toggling ECB↔BoE changes band shape
7. **Cache invalidation**: running in January of a new year triggers Eurostat re-fetch

Run `python eurozone_inflation_analyzer.py` to execute the full verification suite.

---

## Dependencies

### Python

```bash
# Required
pip install pandas numpy requests

# Recommended (ARIMA forecasting)
pip install statsmodels

# Optional (Ledoit-Wolf covariance, BoE ppf)
pip install scikit-learn scipy python-dateutil
```

### Frontend

No new npm packages — uses Recharts (already installed) and lucide-react.

---

## Files

| File | Description |
|------|-------------|
| `eurozone_inflation_analyzer.py` | Python backend — all model classes, MC engine, BoE, country analyzer |
| `eurozone_inflation_tool.jsx` | React dashboard — fan chart, methodology toggle, country breakdown |
| `EUROZONE_INFLATION_GUIDE.md` | This documentation |

---

## Data Sources

| Source | URL | Used For |
|--------|-----|----------|
| Eurostat SDMX-JSON | `ec.europa.eu/eurostat/api/dissemination/…` | HICP weights (primary) |
| ECB Data Portal | `data-api.ecb.europa.eu` | Weight fallback |
| FRED API | `fred.stlouisfed.org` | Historical HICP series |

---

## Recent Eurozone Inflation Context (Q1 2026)

**February 2026**:
- Headline HICP: 1.9% YoY (disinflation continues)
- Core HICP: 2.4% YoY (services stickiness persists)
- Services: 3.4% (primary above-target driver)
- Energy: −3.2% (supporting headline below target)

**Outlook**: Services disinflation pace is the key uncertainty.
ECB fan chart upside risk: energy rebound + wage agreements in Q2–Q3.

---

*Last Updated: March 2026*
*Methodology: ECB suite-of-models + Monte Carlo + Bank of England two-piece normal*
*Weights: Dynamically fetched from Eurostat, annual revision cycle*
