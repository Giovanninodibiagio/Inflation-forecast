# Eurozone Inflation Tool — Quick Start

## Architecture overview

```
eurozone_inflation_analyzer.py   Python engine (models, Monte Carlo, data fetching)
api.py                           FastAPI server — wraps the Python engine
inflation-app/                   React frontend (Vite + Tailwind + Recharts)
```

The React app talks to the Python server via HTTP. If the server is offline the
frontend falls back to simulated data automatically.

---

## Running the full stack

You need **two terminals open at the same time**.

### Terminal 1 — Python API server

```bash
cd "/Users/giovanninodibiagio/Desktop/Inflation "
uvicorn api:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

You can verify the server is healthy by opening `http://localhost:8000/api/health`
in a browser — it returns JSON with the weight source and feature flags.

### Terminal 2 — React dev server

```bash
cd "/Users/giovanninodibiagio/Desktop/Inflation /inflation-app"
npm run dev
```

Then open `http://localhost:5173`.

The status badge in the top-right of the dashboard shows:
- **Green "Python API connected · ensemble active"** — live Python forecasts
- **Amber "API offline · local fallback"** — server not running, JS fallback active

---

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Server status, HICP weight source, feature flags |
| `GET /api/history` | Last 36 months of HICP component data |
| `GET /api/forecast?periods=6&methodology=ecb` | Ensemble + Monte Carlo fan chart |
| `GET /api/commodities` | Brent, EU gas, EUR/USD signals with pressure badge |

`methodology` accepts `ecb` (symmetric bands) or `boe` (Bank of England two-piece normal).

Interactive API docs: `http://localhost:8000/docs`

---

## Optional: real FRED data

Without a FRED API key the server uses a simulated dataset that mirrors the
official 2023-2024 ECB disinflation path. To enable real data:

1. Get a free key at `https://fred.stlouisfed.org/docs/api/` (takes ~2 minutes)
2. Set the environment variable before starting the server:

```bash
export FRED_API_KEY="your_key_here"
uvicorn api:app --reload --port 8000
```

With a key, `/api/history` fetches real EA HICP series directly from FRED and
ARIMAX uses live commodity prices from FRED + yfinance as exogenous regressors.

---

## Python dependencies

```bash
pip install fastapi uvicorn pandas numpy requests statsmodels scipy scikit-learn yfinance
```

All are free; no paid data vendor required.

---

## Forecasting models

When the Python server is running, fan chart bands come from:

1. **Exponential Smoothing** (α = 0.3) — fast, responsive to recent trend
2. **ARIMA(1,1,1)** — autoregressive model with one-step differencing
3. **ARIMAX(1,1,1)** — ARIMA extended with Brent crude and EUR/USD as
   commodity shock regressors; falls back to plain ARIMA if commodity data
   is unavailable (e.g. yfinance rate-limited)

The three models are combined via **inverse-RMSE weighting**: the model with
the lowest historical error gets the highest weight. 10,000 Monte Carlo paths
are then simulated to produce the p10/p25/p50/p75/p90 fan chart bands.

Commodity signals (Brent rising → upside pressure → bands widened by ×1.35)
are shown in the **Commodity Signals** panel and affect band width.

---

## Troubleshooting

**"Module not found: statsmodels"**
```bash
pip install statsmodels
```

**"API offline" badge even though server is running**
- Confirm the server is on port 8000 (`uvicorn api:app --port 8000`)
- Check the browser console for network errors
- Make sure you're opening `http://localhost:5173` (not the 8000 port directly)

**yfinance rate limit warning in terminal**
- Non-fatal — ARIMAX automatically falls back to plain ARIMA
- Try again after a few minutes; yfinance rate-limits rapid sequential downloads

**Port 8000 already in use**
```bash
lsof -ti:8000 | xargs kill -9
uvicorn api:app --reload --port 8000
```
