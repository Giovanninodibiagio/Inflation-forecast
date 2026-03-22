# Eurozone Inflation Forecast Tool

A full-stack application for analysing and forecasting Eurozone inflation using official ECB/Eurostat HICP methodology. Combines a Python forecasting engine with a React interactive dashboard.

## Architecture

```
eurozone_inflation_analyzer.py   Python engine (ECB models, Monte Carlo, data fetching)
api.py                           FastAPI server — exposes JSON endpoints
inflation-app/                   React frontend (Vite + Tailwind + Recharts)
```

The React app communicates with the Python backend via HTTP. If the server is offline, the frontend automatically falls back to simulated data.

## Getting Started

You need **two terminals** running simultaneously.

### Terminal 1 — Python API server

```bash
pip install fastapi uvicorn pandas numpy requests statsmodels scipy scikit-learn yfinance
uvicorn api:app --reload --port 8000
```

Verify it's running: `http://localhost:8000/api/health`

### Terminal 2 — React frontend

```bash
cd inflation-app
npm install
npm run dev
```

Then open `http://localhost:5173`.

The status badge in the top-right shows:
- **Green** — live Python forecasts active
- **Amber** — API offline, JS fallback active

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Server status and feature flags |
| `GET /api/history` | Last 36 months of HICP component data |
| `GET /api/forecast?periods=6&methodology=ecb` | Ensemble + Monte Carlo fan chart |
| `GET /api/commodities` | Brent, EU gas, EUR/USD signals |

`methodology` accepts `ecb` (symmetric bands) or `boe` (Bank of England two-piece normal).

Interactive API docs: `http://localhost:8000/docs`

## Forecasting Models

The fan chart is produced by combining three models via **inverse-RMSE weighting**:

1. **Exponential Smoothing** (α = 0.3) — fast, responsive to recent trends
2. **ARIMA(1,1,1)** — autoregressive model with one-step differencing
3. **ARIMAX(1,1,1)** — ARIMA with Brent crude and EUR/USD as exogenous regressors

10,000 Monte Carlo paths are simulated to produce p10/p25/p50/p75/p90 fan chart bands. Rising commodity prices widen the bands by ×1.35.

## HICP Component Weights (ECB 2025)

| Component | Weight |
|---|---|
| Services | 45.7% |
| Non-Energy Industrial Goods | 25.6% |
| Food / Alcohol / Tobacco | 19.3% |
| Energy | 9.4% |

## Optional: Real FRED Data

By default the server uses ECB SDW data (free, no key required). For additional commodity data via FRED:

1. Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Set the environment variable before starting the server:

```bash
export FRED_API_KEY="your_key_here"
uvicorn api:app --reload --port 8000
```

## Troubleshooting

**"Module not found" error**
```bash
pip install fastapi uvicorn pandas numpy requests statsmodels scipy scikit-learn yfinance
```

**"API offline" badge despite server running**
- Confirm the server is on port 8000
- Check the browser console for network errors
- Make sure you're opening `http://localhost:5173`

**Port 8000 already in use**
```bash
lsof -ti:8000 | xargs kill -9
uvicorn api:app --reload --port 8000
```
