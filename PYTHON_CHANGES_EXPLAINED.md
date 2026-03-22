# Python File — Changes Explained

---

## What the original file did (before the overhaul)

One class (`EurozoneInflationAnalyzer`) with hardcoded 2025 ECB weights and two forecast methods that ran independently — no coordination, no uncertainty, no probability bands. You got a single number per future month.

---

## What changed — class by class

### 1. `HICPWeightFetcher` *(new)*

**What it does:** tries to fetch the official HICP component weights dynamically from the internet instead of relying on hardcoded values. Weights matter because the ECB revises them every January — services, energy, food etc. shift slightly each year.

**Three-layer fallback in order:**
1. **ECB Statistical Data Warehouse (SDW)** — tries to download annual weight series directly from the ECB's own API (`data-api.ecb.europa.eu`). This is the most authoritative source.
2. **Eurostat SDMX-JSON** — if ECB fails, tries Eurostat's dissemination API. Fetches division-level COICOP codes (CP01 = food, CP02 = alcohol/tobacco), computes the food component weight from the actual data, and scales the other three (services, energy, NEIG) proportionally from the hardcoded 2025 ratios.
3. **Hardcoded 2025 ECB values** — final fallback if both APIs are unreachable.

**Cache:** results are saved to `~/.inflation_cache/hicp_weights.json`. The cache is invalidated automatically in January of a new year (when ECB publishes new weights), so you never re-fetch unnecessarily but always get fresh data at the right time.

---

### 2. `ForecastResult` and `FanChartResult` *(new dataclasses)*

Structured containers for outputs — like typed dictionaries. Before, forecast results were loose DataFrames. Now each model returns a `ForecastResult` with:
- `central`: the point forecast array
- `residuals`: the difference between what the model predicted in-sample vs what actually happened — these are critical inputs for the Monte Carlo step
- `rmse`: root mean squared error, used for weighting in the ensemble
- `fitted_values`: what the model would have predicted on historical data

`FanChartResult` holds the full probabilistic output: `p10, p25, p50, p75, p90` plus a `to_dict()` method so it serializes cleanly to JSON for the React frontend.

---

### 3. `BaseInflationModel` ABC *(new)*

An abstract base class that enforces a shared interface: every model must implement `fit(series)` and `forecast(horizon)`. This is what makes the ensemble possible — the `EnsembleForecaster` doesn't care which model it's running, just that it has these two methods.

---

### 4. `ExponentialSmoothingModel` *(refactored from existing code)*

The same algorithm that existed before, extracted into the new class structure. Logic unchanged:

```
Level(t+1) = α × actual(t) + (1−α) × Level(t)
Forecast(h) = Level + trend × 0.1 × h
```

α = 0.3, trend computed from the last 6 months. The key addition: it now also computes and stores `residuals` (actual minus fitted) so the Monte Carlo engine can estimate volatility from this model's errors.

---

### 5. `ARIMAModel` *(refactored from existing code)*

Same ARIMA(1,1,1) as before but now wrapped in the standard interface. The important new piece: it stores `model.resid` — the ARIMA residuals — which are more statistically principled volatility estimates than those from exponential smoothing.

ARIMA(1,1,1) means:
- **1 autoregressive term** — today's inflation is partly a function of last month's
- **1 differencing** — works on changes rather than levels (makes the series stationary)
- **1 moving average term** — accounts for last month's forecast error

---

### 6. `EnsembleForecaster` *(new)*

Runs both models and combines them using **inverse-RMSE weighting**:

```
weight_i = (1 / RMSE_i) / sum(1 / RMSE_j for all j)
```

Lower RMSE = better in-sample fit = higher weight in the final forecast. On the sample data the output showed 71% Exp Smoothing / 29% ARIMA — meaning the smoother model was fitting the data better than ARIMA on that particular series.

The ensemble also concatenates the residuals from all models into a combined pool, which is handed to the Monte Carlo engine.

---

### 7. `MonteCarloEngine` *(new)*

Instead of producing one forecast number per month, it produces 10,000 alternative future paths and then reads off the percentiles.

**Headline-level simulation** (always available):
```
innovation_h ~ Normal(0, σ²)           where σ = std(ensemble residuals)
path[i, h] = central[h]
           + decay × (path[i, h−1] − central[h−1])   # AR(1) carry
           + innovation_h
```
The `decay = 0.7` term means uncertainty is persistent — a shock in month 1 carries forward partially into months 2, 3, etc., which is why the bands widen as you go further out.

**Component-level simulation** (when all 4 components available):
Instead of shocking headline inflation, it shocks each component separately (services, food, energy, NEIG) using a covariance matrix estimated from their historical co-movements. This captures the fact that energy and food often move together, while services is more independent. Then it re-weights the component paths using HICP weights to get headline paths. This is closer to how the ECB actually does it.

For the covariance matrix it uses **Ledoit-Wolf shrinkage** (from scikit-learn) if available — a statistical technique that produces better-behaved covariance matrices when you don't have a lot of data, by pulling the estimated matrix toward a simpler structure.

---

### 8. `TwoPieceNormal` *(new — Bank of England methodology)*

The standard Monte Carlo gives symmetric bands — the upside and downside are equally wide. The Bank of England rejected this in 1998 (Britton, Fisher & Whitley) because in reality inflation risks are often asymmetric: energy prices can spike much more than they can fall, services inflation is sticky upward.

The two-piece normal wraps two half-normal distributions back to back at the mode:
```
f(x) = c × N(mode, σ_lower)   for x < mode
f(x) = c × N(mode, σ_upper)   for x ≥ mode
```

When `σ_upper > σ_lower` the right tail is fatter — the fan chart bands are wider on the upside than the downside. In the implementation, `σ_upper` is inflated by ×1.25 relative to what the symmetric MC would give.

The class implements the full CDF and inverse CDF (PPF) analytically so you can read off exact probability levels without additional simulation.

---

### 9. `CountryContributionAnalyzer` *(new)*

NCB-style (national central bank) decomposition. The EA headline isn't just one number — it's a weighted average of 19 country HICPs. Germany alone is 26.9% of the index.

```
contribution_i = country_weight_i × country_HICP_i
EA headline ≈ Σ contributions
```

The residual `|EA_headline − Σ_contributions|` is a data quality indicator — it should be below 0.1 percentage points if the country data is consistent with the EA aggregate.

Country data can be fetched from FRED (requires API key). The weights are hardcoded from Eurostat's 2024 expenditure share data.

---

### 10. `EurozoneInflationAnalyzer` *(updated)*

The main class was updated to:
- On `__init__`, call `HICPWeightFetcher` to get live weights instead of reading from a class constant
- Build the ensemble automatically (Exp Smoothing always, ARIMA if statsmodels available)
- Add `forecast_with_uncertainty()` — the new main method that returns a full JSON-serializable fan chart
- Keep all existing methods (`forecast_inflation`, `calculate_hicp_inflation`, etc.) unchanged for backwards compatibility

---

## How many ways inflation is computed

There are **four distinct computations**, serving different purposes:

| | What it produces | Used for |
|---|---|---|
| **1. HICP weighted average** | One number per month: the official headline rate | The actual measurement — `Services×0.457 + NEIG×0.256 + Food×0.193 + Energy×0.094` |
| **2. Ensemble point forecast** | One central number per future month | Best-guess deterministic forecast; shown as the solid blue line in the fan chart |
| **3. Monte Carlo fan chart** | 10,000 paths → p10/p25/p50/p75/p90 per month | Probabilistic uncertainty; shows how wide the range of outcomes is and how it widens over the horizon |
| **4. BoE two-piece normal** | Asymmetric bands per month | Same as MC but with the upside uncertainty inflated, reflecting that inflation tends to surprise on the upside more than downside |

Within computation #2, the ensemble itself runs **two sub-models** (Exp Smoothing and ARIMA) and combines them.

---

## Data sources

| Source | What it provides | How it's accessed |
|---|---|---|
| **ECB Statistical Data Warehouse** | Annual HICP weights for the 4 EA components | HTTP GET, no API key, public |
| **Eurostat SDMX-JSON API** | Official HICP component weights (CP01, CP02 for food) | HTTP GET, no API key, public |
| **FRED (Federal Reserve St. Louis)** | Historical monthly HICP rates — EA headline, energy, food, services, and country-level series | HTTP GET, **free API key required** from fred.stlouisfed.org |
| **Hardcoded fallback** | 2025 ECB weights (services 45.7%, NEIG 25.6%, food 19.3%, energy 9.4%) | In-code constant, no network needed |

FRED is the source for the actual inflation time series used to fit and validate the models. ECB SDW and Eurostat are only used for the weights. The React frontend currently uses simulated data that mirrors the ECB's published historical pattern — no live API calls are made from the browser.
