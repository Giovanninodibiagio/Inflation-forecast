import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ComposedChart, Area, ReferenceLine,
} from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle, Activity, Globe, BarChart2, Flame, Droplets, DollarSign, Zap } from 'lucide-react';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HICP_WEIGHTS = {
  services: 0.457,
  nonEnergyGoods: 0.256,
  foodAlcoholTobacco: 0.193,
  energy: 0.094,
};

const COUNTRY_WEIGHTS = {
  DE: { weight: 0.269, name: 'Germany',     color: '#3b82f6' },
  FR: { weight: 0.207, name: 'France',      color: '#10b981' },
  IT: { weight: 0.180, name: 'Italy',       color: '#f59e0b' },
  ES: { weight: 0.122, name: 'Spain',       color: '#ef4444' },
  NL: { weight: 0.058, name: 'Netherlands', color: '#8b5cf6' },
  BE: { weight: 0.034, name: 'Belgium',     color: '#06b6d4' },
  AT: { weight: 0.030, name: 'Austria',     color: '#ec4899' },
  OT: { weight: 0.100, name: 'Other EA',    color: '#6b7280' },
};

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

const API_BASE = '/api';

async function apiFetch(path) {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) throw new Error(`API ${path} returned ${resp.status}`);
  return resp.json();
}

// ---------------------------------------------------------------------------
// Local fallback data (used if FastAPI server is unreachable)
// ---------------------------------------------------------------------------

function buildFallbackHistory() {
  // Jan 2023 – Feb 2026 (38 months)
  // 2025 reflects Middle East oil shock: energy recovers from -4% to +2.5%
  // services sticky ~4%, headline re-accelerates to ~2.7-2.8%
  const baseline = [
    // 2023
    4.2, 4.1, 4.3, 4.2, 4.1, 4.0, 3.8, 3.5, 3.4, 3.5, 3.6, 3.4,
    // 2024
    3.2, 3.3, 3.0, 2.9, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 2.5, 2.0,
    // 2025: energy base effect reverses, headline stabilises 2.2-2.7%
    2.5, 2.4, 2.2, 2.3, 2.2, 2.5, 2.6, 2.4, 2.3, 2.5, 2.6, 2.7,
    // 2026 Jan-Feb
    2.8, 2.7,
  ];
  const comps = {
    services:           [4.8,4.9,5.0,4.8,4.7,4.6,4.4,4.2,4.1,4.0,4.1,4.2, 4.0,3.9,3.8,3.7,3.6,3.5,3.4,3.5,3.4,3.5,3.4,3.2, 4.0,3.9,3.8,3.9,4.0,4.1,4.2,4.1,4.0,4.1,4.2,4.3, 4.4,4.3],
    nonEnergyGoods:     [2.3,2.2,2.1,2.0,1.9,1.8,1.7,1.5,1.4,1.3,1.2,1.1, 1.0,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.4,0.4, 0.6,0.5,0.4,0.5,0.4,0.5,0.6,0.5,0.4,0.5,0.5,0.6, 0.6,0.5],
    foodAlcoholTobacco: [4.5,4.6,4.7,4.5,4.3,4.1,3.9,3.7,3.5,3.4,3.2,3.1, 3.0,3.1,3.2,3.3,3.4,3.2,3.1,3.0,2.8,2.6,2.5,2.6, 2.7,2.6,2.5,2.6,2.5,2.7,2.8,2.7,2.6,2.7,2.8,2.9, 3.0,2.9],
    energy:             [3.2,2.8,2.0,0.5,-1.5,-2.3,-1.5,-3.2,-2.8,-1.5,0.2,-0.5, -2.0,-1.5,-2.5,-3.2,-3.0,-2.0,-1.0,-0.8,-1.5,-2.0,-1.9,-4.0, -1.0,-0.5,0.0,0.5,0.0,1.5,2.0,1.5,1.0,2.0,2.0,2.0, 2.5,2.0],
  };
  const startDate = new Date(2023, 0, 1);
  const today = new Date();
  const months = [];
  for (let i = 0; i < baseline.length; i++) {
    const d = new Date(startDate);
    d.setMonth(d.getMonth() + i);
    if (d > today) break;
    const s = comps.services[i], n = comps.nonEnergyGoods[i];
    const core = (s * HICP_WEIGHTS.services + n * HICP_WEIGHTS.nonEnergyGoods)
               / (HICP_WEIGHTS.services + HICP_WEIGHTS.nonEnergyGoods);
    months.push({
      date: d.toISOString().slice(0, 7),
      headline: +baseline[i].toFixed(2), core: +core.toFixed(2),
      services: +s.toFixed(2), nonEnergyGoods: +n.toFixed(2),
      foodAlcoholTobacco: +comps.foodAlcoholTobacco[i].toFixed(2),
      energy: +comps.energy[i].toFixed(2),
    });
  }
  return months;
}

/** Inverse-normal approx (Abramowitz & Stegun rational) */
function normPpf(p) {
  p = Math.max(1e-9, Math.min(1 - 1e-9, p));
  const t = Math.sqrt(-2 * Math.log(Math.min(p, 1 - p)));
  const c = [2.515517, 0.802853, 0.010328];
  const d = [1.432788, 0.189269, 0.001308];
  const approx = t - (c[0] + c[1]*t + c[2]*t*t) / (1 + d[0]*t + d[1]*t*t + d[2]*t*t*t);
  return p < 0.5 ? -approx : approx;
}

/**
 * Compute fan chart bands for a given forecast series.
 *
 * methodology:
 *   'ecb' → symmetric normal bands (σ_L = σ_U)
 *   'boe' → two-piece normal (σ_U = 1.25 × σ_L — upside-skewed)
 *
 * Returns array of objects with:
 *   date, central, p50,
 *   outerLow (invisible baseline for p10–p90 band),
 *   outerWidth (p90 - p10),
 *   innerLow (invisible baseline for p25–p75 band),
 *   innerWidth (p75 - p25)
 */
function computeFanBands(historicalData, forecastPoints, methodology, sigmaMultiplier = 1.0) {
  // Estimate volatility from historical changes
  const vals = historicalData.map(d => d.headline);
  const changes = vals.slice(1).map((v, i) => v - vals[i]);
  const variance = changes.reduce((s, d) => s + d * d, 0) / changes.length;
  const baseVol = Math.sqrt(variance) * sigmaMultiplier;

  return forecastPoints.map((pt, i) => {
    const h = i + 1;
    const sigmaL = baseVol * Math.sqrt(h);
    const sigmaU = methodology === 'boe' ? sigmaL * 1.25 : sigmaL;

    const c = pt.central;
    // For two-piece normal, upper and lower use different sigmas
    const p10 = c + normPpf(0.10) * sigmaL;
    const p25 = c + normPpf(0.25) * sigmaL;
    const p50 = methodology === 'boe'
      ? c + (sigmaU - sigmaL) * 0.1   // slight upward shift in median for BoE
      : c;
    const p75 = c + normPpf(0.75) * sigmaU;
    const p90 = c + normPpf(0.90) * sigmaU;

    return {
      date: pt.date,
      central: +c.toFixed(2),
      p50: +p50.toFixed(2),
      // Recharts stacked-area fan chart pattern:
      // Stack "outer": invisible baseline (p10) + visible width (p90-p10)
      outerLow:   +p10.toFixed(2),
      outerWidth: +(p90 - p10).toFixed(2),
      // Stack "inner": invisible baseline (p25) + visible width (p75-p25)
      innerLow:   +p25.toFixed(2),
      innerWidth: +(p75 - p25).toFixed(2),
      isForecast: true,
    };
  });
}

/** Local JS fallback forecast — used by legacy trajectory chart and when API is down. */
function buildForecastPoints(historicalData, forecastMonths) {
  const vals = historicalData.map(d => d.headline);
  const n = Math.min(6, vals.length);
  const recent = vals.slice(-n);
  const older = vals.slice(-2 * n, -n);
  const avgRecent = recent.reduce((a, b) => a + b, 0) / n;
  const avgOlder = older.length ? older.reduce((a, b) => a + b, 0) / older.length : avgRecent;
  const trend = avgRecent - avgOlder;

  const lastDate = new Date(historicalData[historicalData.length - 1].date + '-01');
  const lastVal = vals[vals.length - 1];

  return Array.from({ length: forecastMonths }, (_, i) => {
    const d = new Date(lastDate);
    d.setMonth(d.getMonth() + i + 1);
    const central = lastVal + trend * (i + 1) * 0.08;
    return {
      date: d.toISOString().slice(0, 7),
      central: +central.toFixed(2),
    };
  });
}

/**
 * Fallback commodity snapshot (used when FastAPI server is unreachable).
 * Reflects Middle-East-war upside shock scenario.
 * In normal operation this data comes from /api/commodities (Python backend).
 */
function buildFallbackCommoditySignals() {
  // Simulated 6-month trailing prices for sparkline context
  const brentHistory = [74.2, 76.5, 79.8, 83.1, 85.4, 88.3]; // $/bbl — uptrend
  const gasHistory   = [28.4, 30.1, 33.7, 36.2, 38.9, 41.1]; // €/MWh — uptrend
  const eurusdHistory = [1.095, 1.087, 1.079, 1.071, 1.063, 1.058]; // EUR/USD — weakening

  const brentCurrent  = brentHistory.at(-1);
  const brentPrev     = brentHistory.at(-2);
  const gasCurrent    = gasHistory.at(-1);
  const gasPrev       = gasHistory.at(-2);
  const eurusdCurrent = eurusdHistory.at(-1);
  const eurusdPrev    = eurusdHistory.at(-2);

  const brentChg  = ((brentCurrent  - brentPrev)  / brentPrev)  * 100;
  const gasChg    = ((gasCurrent    - gasPrev)    / gasPrev)    * 100;
  const eurusdChg = ((eurusdCurrent - eurusdPrev) / eurusdPrev) * 100;

  // Pressure logic:
  // Upside = commodity prices rising + EUR/USD weakening (imports cost more)
  // Downside = falling commodity prices
  // Neutral = mixed or low-magnitude signals
  const upsideSignals   = (brentChg > 1.5 ? 1 : 0) + (gasChg > 1.5 ? 1 : 0) + (eurusdChg < -0.5 ? 1 : 0);
  const downsideSignals = (brentChg < -1.5 ? 1 : 0) + (gasChg < -1.5 ? 1 : 0) + (eurusdChg > 0.5 ? 1 : 0);

  let pressure = 'neutral';
  if (upsideSignals >= 2)   pressure = 'upside';
  else if (downsideSignals >= 2) pressure = 'downside';

  // Sigma multiplier applied to fan chart bands: upside = wider uncertainty
  const sigmaMultiplier = pressure === 'upside' ? 1.35 : pressure === 'downside' ? 0.85 : 1.0;

  return {
    pressure,
    sigmaMultiplier,
    commodities: [
      {
        id:      'brent',
        label:   'Brent Crude',
        unit:    '$/bbl',
        current: brentCurrent,
        prev:    brentPrev,
        change:  brentChg,
        history: brentHistory,
        icon:    'droplets',
        // Inflation transmission: every $10 rise in Brent ≈ +0.15–0.25pp EA energy CPI
        note:    'Every $10 rise ≈ +0.2pp EA energy CPI',
      },
      {
        id:      'eugas',
        label:   'EU Natural Gas',
        unit:    '€/MWh',
        current: gasCurrent,
        prev:    gasPrev,
        change:  gasChg,
        history: gasHistory,
        icon:    'flame',
        note:    'Direct energy + industrial input cost',
      },
      {
        id:      'eurusd',
        label:   'EUR/USD',
        unit:    '',
        current: eurusdCurrent,
        prev:    eurusdPrev,
        // For EUR/USD: appreciation = disinflationary (imports cheaper), weakening = inflationary
        change:  eurusdChg,
        history: eurusdHistory,
        icon:    'dollar',
        // Invert the direction indicator: EUR/USD fall is an upside inflation risk
        inflationary: eurusdChg < 0,
        note:    'Weaker EUR → pricier USD-denominated commodity imports',
      },
    ],
  };
}

function buildCountryContributions(historicalData) {
  // Simulated country HICP rates calibrated to EA headline
  const countryRates = {
    DE: (h) => +(h * 0.96 + 0.1).toFixed(2),
    FR: (h) => +(h * 0.88 - 0.1).toFixed(2),
    IT: (h) => +(h * 0.82 - 0.2).toFixed(2),
    ES: (h) => +(h * 1.18 + 0.3).toFixed(2),
    NL: (h) => +(h * 1.05 + 0.2).toFixed(2),
    BE: (h) => +(h * 0.92 + 0.1).toFixed(2),
    AT: (h) => +(h * 0.97 + 0.0).toFixed(2),
  };

  return historicalData.map(d => {
    const row = { date: d.date };
    let computed = 0;
    for (const [iso, { weight }] of Object.entries(COUNTRY_WEIGHTS)) {
      if (iso === 'OT') continue;
      const rate = (countryRates[iso] || ((h) => h))(d.headline);
      row[`${iso}_contribution`] = +(rate * weight).toFixed(3);
      computed += rate * weight;
    }
    // Other EA = residual
    row['OT_contribution'] = +(d.headline - computed).toFixed(3);
    row.ea_headline = d.headline;
    row.ea_computed = +computed.toFixed(2);
    return row;
  });
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Commodity Signals Panel
// ---------------------------------------------------------------------------

const PRESSURE_CONFIG = {
  upside: {
    label:   'UPSIDE PRESSURE',
    subtitle: 'Commodity prices rising — inflation bands widened',
    bg:      'from-orange-900/40 to-red-900/30',
    border:  'border-orange-600/50',
    badge:   'bg-orange-500',
    text:    'text-orange-300',
    dot:     'bg-orange-400',
  },
  downside: {
    label:   'DOWNSIDE PRESSURE',
    subtitle: 'Commodity prices falling — inflation bands narrowed',
    bg:      'from-green-900/40 to-emerald-900/30',
    border:  'border-green-600/50',
    badge:   'bg-green-500',
    text:    'text-green-300',
    dot:     'bg-green-400',
  },
  neutral: {
    label:   'NEUTRAL',
    subtitle: 'No significant directional commodity pressure',
    bg:      'from-slate-800 to-slate-900',
    border:  'border-slate-700',
    badge:   'bg-slate-500',
    text:    'text-slate-300',
    dot:     'bg-slate-400',
  },
};

const CommodityIcon = ({ id }) => {
  const cls = 'shrink-0';
  if (id === 'brent')  return <Droplets  size={18} className={cls} />;
  if (id === 'eugas')  return <Flame     size={18} className={cls} />;
  if (id === 'eurusd') return <DollarSign size={18} className={cls} />;
  return <Zap size={18} className={cls} />;
};

/** Mini sparkline using SVG — no extra dependencies */
const Sparkline = ({ values, color }) => {
  if (!values || values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const w = 72, h = 28, pad = 2;
  const pts = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((v - min) / range) * (h - 2 * pad);
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg width={w} height={h} className="opacity-70">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
};

const CommodityCard = ({ commodity }) => {
  // For EUR/USD: weakening is inflationary (upside risk), so invert direction
  const isUp = commodity.change > 0;
  const isInflationary = commodity.inflationary !== undefined
    ? commodity.inflationary
    : isUp;

  const changeColor = isInflationary ? 'text-red-400' : 'text-green-400';
  const sparkColor  = isInflationary ? '#f87171'      : '#34d399';
  const arrowEl = isUp
    ? <TrendingUp size={14} />
    : <TrendingDown size={14} />;

  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-slate-300">
          <CommodityIcon id={commodity.id} />
          <span className="text-xs font-semibold">{commodity.label}</span>
        </div>
        <Sparkline values={commodity.history} color={sparkColor} />
      </div>

      <div className="flex items-end justify-between">
        <div>
          <span className="text-2xl font-bold text-white font-mono">
            {commodity.id === 'eurusd'
              ? commodity.current.toFixed(4)
              : commodity.current.toFixed(1)}
          </span>
          {commodity.unit && (
            <span className="text-xs text-slate-400 ml-1">{commodity.unit}</span>
          )}
        </div>
        <div className={`flex items-center gap-1 text-sm font-semibold ${changeColor}`}>
          {arrowEl}
          <span>{Math.abs(commodity.change).toFixed(1)}%</span>
          <span className="text-xs font-normal text-slate-500">MoM</span>
        </div>
      </div>

      <p className="text-xs text-slate-500 leading-tight">{commodity.note}</p>
    </div>
  );
};

const CommoditySignalsPanel = ({ signals }) => {
  const cfg = PRESSURE_CONFIG[signals.pressure];

  return (
    <div className={`bg-gradient-to-br ${cfg.bg} border ${cfg.border} rounded-2xl p-6 mb-8`}>
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className={`w-2 h-2 rounded-full ${cfg.dot} animate-pulse`} />
            <h3 className="text-sm font-bold text-slate-200 tracking-wide">COMMODITY SIGNALS</h3>
          </div>
          <p className="text-xs text-slate-400">
            Exogenous regressors — ARIMAX model + fan chart band adjustment
          </p>
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${cfg.badge} bg-opacity-90 self-start`}>
          <span className="text-white text-xs font-bold tracking-wider">{cfg.label}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5">
        {signals.commodities.map(c => (
          <CommodityCard key={c.id} commodity={c} />
        ))}
      </div>

      <div className={`pt-4 border-t border-slate-700/50 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2`}>
        <p className={`text-sm font-medium ${cfg.text}`}>
          {cfg.subtitle}
        </p>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span>
            Band multiplier:{' '}
            <span className="font-mono text-slate-300">×{signals.sigmaMultiplier.toFixed(2)}</span>
          </span>
          <span className="hidden sm:inline">·</span>
          <span>Sources: FRED + yfinance (Brent, EU gas, EUR/USD)</span>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------

const MethodologySelector = ({ value, onChange }) => {
  const options = [
    { id: 'ecb', label: 'ECB Suite-of-Models', icon: <Activity size={14} /> },
    { id: 'boe', label: 'Bank of England', icon: <TrendingUp size={14} /> },
    { id: 'ncb', label: 'NCB Country View', icon: <Globe size={14} /> },
  ];
  return (
    <div className="flex gap-2 flex-wrap">
      {options.map(opt => (
        <button
          key={opt.id}
          onClick={() => onChange(opt.id)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            value === opt.id
              ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/25'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          {opt.icon}
          {opt.label}
        </button>
      ))}
    </div>
  );
};

const FanChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload || {};

  if (row.isForecast) {
    const p10 = row.outerLow;
    const p90 = +(row.outerLow + row.outerWidth).toFixed(2);
    const p25 = row.innerLow;
    const p75 = +(row.innerLow + row.innerWidth).toFixed(2);
    return (
      <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-xs text-slate-200 shadow-xl">
        <p className="font-bold text-cyan-400 mb-1">{label} (Forecast)</p>
        <p>Central: <span className="text-white font-semibold">{row.central?.toFixed(2)}%</span></p>
        <p>Median (p50): <span className="text-purple-300">{row.p50?.toFixed(2)}%</span></p>
        <p>80% interval: {p10?.toFixed(2)}% – {p90?.toFixed(2)}%</p>
        <p>50% interval: {p25?.toFixed(2)}% – {p75?.toFixed(2)}%</p>
      </div>
    );
  }
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-xs text-slate-200 shadow-xl">
      <p className="font-bold text-cyan-400 mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i}>{p.name}: <span className="font-semibold">{(+p.value).toFixed(2)}%</span></p>
      ))}
    </div>
  );
};

const FanChartVisualization = ({ historicalData, fanBands, methodology, commoditySignals }) => {
  // Combine historical (with headline) and forecast (with fan band columns)
  const combined = [
    ...historicalData.map(d => ({ ...d })),
    ...fanBands,
  ];

  const methodologyLabel = {
    ecb: 'ECB suite-of-models ensemble (Exp Smoothing + ARIMA + ARIMAX), inverse-RMSE weighted, 10k-path Monte Carlo',
    boe: 'Bank of England two-piece normal (Britton, Fisher & Whitley 1998) — σ_upper inflated 1.25× for upside risk',
  }[methodology] || '';

  const bandColor = methodology === 'boe' ? '#f59e0b' : '#06b6d4';
  const commodityAdjusted = commoditySignals && commoditySignals.sigmaMultiplier !== 1.0;
  const pressureCfg = commoditySignals ? PRESSURE_CONFIG[commoditySignals.pressure] : null;

  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 mb-8">
      <div className="mb-6 flex flex-col sm:flex-row sm:items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold mb-1">
            Probabilistic Inflation Fan Chart
          </h2>
          <p className="text-slate-400 text-sm">
            {methodology === 'boe'
              ? 'Bank of England two-piece normal — asymmetric upside/downside uncertainty'
              : 'ECB suite-of-models — symmetric Monte Carlo fan chart (10,000 simulations)'}
          </p>
        </div>
        {commodityAdjusted && pressureCfg && (
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold ${pressureCfg.text} border ${pressureCfg.border} bg-slate-900/60 self-start`}>
            <div className={`w-1.5 h-1.5 rounded-full ${pressureCfg.dot} animate-pulse`} />
            Bands {commoditySignals.pressure === 'upside' ? 'widened' : 'narrowed'} for commodity{' '}
            {commoditySignals.pressure} risk ×{commoditySignals.sigmaMultiplier.toFixed(2)}
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={combined} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} stroke="#475569"
            interval={2} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} stroke="#475569"
            label={{ value: '% YoY', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }} />
          <Tooltip content={<FanChartTooltip />} />
          <ReferenceLine y={2.0} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1.5}
            label={{ value: 'ECB 2%', position: 'right', fill: '#ef4444', fontSize: 10 }} />

          {/* Historical headline — solid filled area */}
          <Area
            type="monotone"
            dataKey="headline"
            fill="url(#histGrad)"
            stroke="#06b6d4"
            strokeWidth={2.5}
            dot={false}
            name="Historical"
            connectNulls={false}
          />

          {/* Fan chart — outer band (p10 → p90): stacked areas */}
          {/* Invisible baseline lifts the band to start at p10 */}
          <Area
            type="monotone"
            dataKey="outerLow"
            stackId="outer"
            fill="transparent"
            stroke="none"
            dot={false}
            legendType="none"
            name=""
            connectNulls
          />
          <Area
            type="monotone"
            dataKey="outerWidth"
            stackId="outer"
            fill={bandColor}
            fillOpacity={0.12}
            stroke="none"
            dot={false}
            name="80% interval"
            connectNulls
          />

          {/* Fan chart — inner band (p25 → p75) */}
          <Area
            type="monotone"
            dataKey="innerLow"
            stackId="inner"
            fill="transparent"
            stroke="none"
            dot={false}
            legendType="none"
            name=""
            connectNulls
          />
          <Area
            type="monotone"
            dataKey="innerWidth"
            stackId="inner"
            fill={bandColor}
            fillOpacity={0.22}
            stroke="none"
            dot={false}
            name="50% interval"
            connectNulls
          />

          {/* Central forecast line */}
          <Line
            type="monotone"
            dataKey="central"
            stroke="#1e40af"
            strokeWidth={2}
            dot={false}
            name="Central forecast"
            connectNulls
          />

          {/* Median (p50) — shows skew vs central in BoE mode */}
          <Line
            type="monotone"
            dataKey="p50"
            stroke="#a855f7"
            strokeWidth={1.5}
            strokeDasharray="5 4"
            dot={false}
            name="Median (p50)"
            connectNulls
          />

          <Legend
            wrapperStyle={{ paddingTop: 16 }}
            formatter={(value) => <span style={{ color: '#cbd5e1', fontSize: 12 }}>{value}</span>}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-4 pt-4 border-t border-slate-700">
        <p className="text-xs text-slate-500 flex items-start gap-2">
          <AlertCircle size={13} className="mt-0.5 shrink-0" />
          <span>
            Cyan area: historical HICP | Shaded bands: forecast uncertainty |{' '}
            <span className="text-slate-400">{methodologyLabel}</span>
          </span>
        </p>
      </div>
    </div>
  );
};

const CountryContributionChart = ({ countryData }) => {
  if (!countryData.length) return null;
  const last12 = countryData.slice(-12);

  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 mb-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-1">Country Contributions to EA Headline HICP</h2>
        <p className="text-slate-400 text-sm">
          NCB-style decomposition — contribution_i = country_weight_i × country_HICP_i
        </p>
      </div>

      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={last12} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} stroke="#475569" />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            stroke="#475569"
            label={{ value: 'pp contribution', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
            labelStyle={{ color: '#e2e8f0' }}
            formatter={(v, name) => [`${(+v).toFixed(3)}pp`, name]}
          />
          <ReferenceLine y={0} stroke="#475569" />
          <Legend
            wrapperStyle={{ paddingTop: 12 }}
            formatter={(v) => <span style={{ color: '#cbd5e1', fontSize: 11 }}>{v}</span>}
          />
          {Object.entries(COUNTRY_WEIGHTS).map(([iso, { name, color }]) => (
            <Bar
              key={iso}
              dataKey={`${iso}_contribution`}
              stackId="contributions"
              fill={color}
              name={`${name} (${(COUNTRY_WEIGHTS[iso].weight * 100).toFixed(1)}%)`}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>

      {/* Summary table */}
      <div className="mt-6 pt-4 border-t border-slate-700">
        <p className="text-xs text-slate-400 mb-3">Latest month contribution breakdown</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(COUNTRY_WEIGHTS).map(([iso, { name, color, weight }]) => {
            const latest = countryData[countryData.length - 1];
            const val = latest ? latest[`${iso}_contribution`] : null;
            return (
              <div key={iso} className="flex items-center gap-2 text-xs">
                <div className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                <span className="text-slate-400">{name}</span>
                <span className="text-white font-mono ml-auto">
                  {val !== undefined && val !== null ? `${(+val).toFixed(2)}pp` : '—'}
                </span>
              </div>
            );
          })}
        </div>
        <p className="text-xs text-slate-600 mt-3 flex items-center gap-1">
          <AlertCircle size={11} />
          Cross-check: Σ contributions ≈ EA headline (residual &lt; 0.1pp indicates data quality)
        </p>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const EurozoneInflationTool = () => {
  const [data, setData] = useState(null);
  const [fanBands, setFanBands] = useState([]);
  const [countryData, setCountryData] = useState([]);
  const [commoditySignals, setCommoditySignals] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState(false);
  const [methodology, setMethodology] = useState('ecb');
  const [forecastMonths, setForecastMonths] = useState(6);
  const [selectedMetric, setSelectedMetric] = useState('headline');
  // Track whether fan bands came from API (Python ensemble) or local JS fallback
  const [forecastSource, setForecastSource] = useState('local');

  // Fetch fan bands from Python API
  const fetchForecast = useCallback(async (method, periods) => {
    try {
      const result = await apiFetch(`/forecast?periods=${periods}&methodology=${method}`);
      setFanBands(result.fanBands);
      setForecastSource('python');
    } catch {
      // API unavailable — fall back to local JS computation
      setData(prev => {
        if (prev) {
          const pts = buildForecastPoints(prev, periods);
          const sig = commoditySignals?.sigmaMultiplier ?? 1.0;
          setFanBands(computeFanBands(prev, pts, method, sig));
        }
        return prev;
      });
      setForecastSource('local');
    }
  }, [commoditySignals]);

  // Initial data load: history + commodities from API, fan chart separately
  useEffect(() => {
    setLoading(true);

    Promise.all([
      apiFetch('/history'),
      apiFetch('/commodities'),
    ])
      .then(([hist, comm]) => {
        const months = hist.months;
        setData(months);
        // Attach the history data source to commodity signals so the badge can show it
        setCommoditySignals({ ...comm, source: hist.source });
        setCountryData(buildCountryContributions(months));
        setApiError(false);
      })
      .catch(() => {
        // API down — use local fallback data
        const months = buildFallbackHistory();
        const comm   = buildFallbackCommoditySignals();
        setData(months);
        setCommoditySignals(comm);
        setCountryData(buildCountryContributions(months));
        setApiError(true);
      })
      .finally(() => setLoading(false));
  }, []);

  // Re-fetch forecast whenever methodology or horizon changes
  useEffect(() => {
    if (!data) return;
    fetchForecast(methodology, forecastMonths);
  }, [methodology, forecastMonths, data, fetchForecast]);

  const latestData = data?.at(-1) ?? null;
  const prevData = data?.at(-2) ?? null;

  const stats = latestData
    ? [
        { label: 'Headline Inflation', value: latestData.headline, change: latestData.headline - (prevData?.headline ?? latestData.headline), target: 2.0 },
        { label: 'Core Inflation',     value: latestData.core,     change: latestData.core - (prevData?.core ?? latestData.core),             target: 2.0 },
        { label: 'Services',           value: latestData.services,      weight: '45.7%', change: latestData.services - (prevData?.services ?? latestData.services) },
        { label: 'Energy',             value: latestData.energy,        weight: '9.4%',  change: latestData.energy - (prevData?.energy ?? latestData.energy) },
      ]
    : [];

  // Simple exponential smoothing for legacy "flat" forecast (non-fan views)
  const simpleForecast = (() => {
    if (!data) return [];
    const pts = buildForecastPoints(data, forecastMonths);
    const last = new Date(data.at(-1).date + '-01');
    return pts.map((pt, i) => {
      const d = new Date(last);
      d.setMonth(d.getMonth() + i + 1);
      return {
        date: pt.date,
        headline: pt.central,
        core: +(pt.central * 0.93).toFixed(2),
        services: +(data.at(-1).services - i * 0.02).toFixed(2),
        isForecast: true,
      };
    });
  })();

  const combinedLegacy = data ? [...data, ...simpleForecast] : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white p-8">

      {/* Header */}
      <div className="mb-10 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-5xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">
            Eurozone Inflation Analysis
          </h1>
          <p className="text-slate-400 text-lg">
            ECB Suite-of-Models · Monte Carlo Fan Charts · Bank of England Methodology
          </p>
        </div>
        <div className="flex flex-col items-end gap-1.5 self-start sm:self-auto">
          {!apiError ? (
            <>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-900/40 border border-emerald-700/50 rounded-lg text-xs text-emerald-300">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Python API connected
                {forecastSource === 'python' && <span className="text-emerald-500">· ensemble active</span>}
              </div>
              {commoditySignals?.source && (
                <div className="text-xs text-slate-500 pr-1">
                  Data: <span className="text-slate-400">
                    {commoditySignals.source === 'ecb_sdw'  ? 'ECB SDW (live)' :
                     commoditySignals.source === 'fred'     ? 'FRED (live)' :
                     'simulated fallback'}
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-900/40 border border-amber-700/50 rounded-lg text-xs text-amber-300">
              <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              API offline · local fallback
            </div>
          )}
        </div>
      </div>

      {loading ? (
        <div className="text-center py-24">
          <div className="animate-spin rounded-full h-14 w-14 border-4 border-cyan-400 border-t-transparent mx-auto mb-5" />
          <p className="text-slate-300 text-lg">Loading inflation data…</p>
          <p className="text-slate-500 text-sm mt-1">Fetching Eurostat/ECB weights</p>
        </div>
      ) : (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {stats.map((stat, idx) => (
              <div
                key={idx}
                className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-xl p-6 hover:border-cyan-500/50 transition-all"
              >
                <p className="text-slate-400 text-sm font-medium mb-2">{stat.label}</p>
                <div className="flex items-end justify-between">
                  <div>
                    <p className="text-3xl font-bold text-white">{stat.value?.toFixed(2)}%</p>
                    {stat.weight && <p className="text-xs text-slate-500 mt-1">Weight: {stat.weight}</p>}
                  </div>
                  <div className={`flex items-center gap-1 ${stat.change >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                    {stat.change >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
                    <span className="text-sm font-semibold">{Math.abs(stat.change).toFixed(2)}</span>
                  </div>
                </div>
                {stat.target != null && (
                  <div className="mt-3 pt-3 border-t border-slate-700">
                    <p className="text-xs text-slate-500">
                      ECB Target: <span className="text-cyan-400 font-semibold">{stat.target}%</span>
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Commodity signals panel */}
          {commoditySignals && (
            <CommoditySignalsPanel signals={commoditySignals} />
          )}

          {/* Methodology selector + forecast horizon */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 mb-8">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-300 mb-3">Methodology</h3>
                <MethodologySelector value={methodology} onChange={setMethodology} />
              </div>
              <div className="sm:w-56">
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Forecast Horizon: <span className="text-white font-bold">{forecastMonths} months</span>
                </label>
                <input
                  type="range" min="1" max="12" value={forecastMonths}
                  onChange={e => setForecastMonths(+e.target.value)}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            {/* Methodology info pills */}
            <div className="mt-4 pt-4 border-t border-slate-700 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-slate-400">
              {methodology === 'ecb' && (
                <>
                  <span className="px-2 py-1 bg-slate-700 rounded">✓ Exp Smoothing (α=0.3)</span>
                  <span className="px-2 py-1 bg-slate-700 rounded">✓ ARIMA(1,1,1)</span>
                  <span className="px-2 py-1 bg-slate-700 rounded">✓ ARIMAX + commodity exog</span>
                  <span className="px-2 py-1 bg-slate-700 rounded">✓ 10k MC paths</span>
                </>
              )}
              {methodology === 'boe' && (
                <>
                  <span className="px-2 py-1 bg-amber-900/40 rounded text-amber-300">✓ Two-piece normal</span>
                  <span className="px-2 py-1 bg-amber-900/40 rounded text-amber-300">✓ Asymmetric σ</span>
                  <span className="px-2 py-1 bg-amber-900/40 rounded text-amber-300">✓ Upside risk skew ×1.25</span>
                  <span className="px-2 py-1 bg-amber-900/40 rounded text-amber-300">✓ Britton et al. (1998)</span>
                </>
              )}
              {methodology === 'ncb' && (
                <>
                  <span className="px-2 py-1 bg-emerald-900/40 rounded text-emerald-300">✓ DE 26.9%</span>
                  <span className="px-2 py-1 bg-emerald-900/40 rounded text-emerald-300">✓ FR 20.7%</span>
                  <span className="px-2 py-1 bg-emerald-900/40 rounded text-emerald-300">✓ IT 18.0%</span>
                  <span className="px-2 py-1 bg-emerald-900/40 rounded text-emerald-300">✓ ES 12.2%</span>
                </>
              )}
            </div>
          </div>

          {/* Main visualisation: fan chart OR country breakdown */}
          {methodology !== 'ncb' ? (
            <FanChartVisualization
              historicalData={data}
              fanBands={fanBands}
              methodology={methodology}
              commoditySignals={commoditySignals}
            />
          ) : (
            <CountryContributionChart countryData={countryData} />
          )}

          {/* Legacy trajectory chart (Headline vs Core toggle) */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 mb-8">
            <div className="mb-6 flex items-center justify-between flex-wrap gap-3">
              <div>
                <h2 className="text-2xl font-bold mb-1">Inflation Trajectory</h2>
                <p className="text-slate-400 text-sm">
                  {selectedMetric === 'headline'
                    ? 'All-items HICP including volatile components'
                    : 'Core inflation excluding energy and food'}
                </p>
              </div>
              <div className="flex gap-2">
                {['headline', 'core'].map(m => (
                  <button
                    key={m}
                    onClick={() => setSelectedMetric(m)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      selectedMetric === m
                        ? 'bg-cyan-500 text-white'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {m === 'headline' ? 'Headline' : 'Core'} Inflation
                  </button>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart data={combinedLegacy}>
                <defs>
                  <linearGradient id="colorArea2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} stroke="#475569" interval={2} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} stroke="#475569"
                  label={{ value: '% YoY', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
                  labelStyle={{ color: '#e2e8f0' }}
                  formatter={v => `${(+v).toFixed(2)}%`}
                />
                <ReferenceLine y={2.0} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1.5}
                  label={{ value: 'ECB 2%', position: 'right', fill: '#ef4444', fontSize: 10 }} />
                <Area type="monotone" dataKey={selectedMetric} fill="url(#colorArea2)" stroke="#06b6d4"
                  strokeWidth={2.5} dot={false}
                  name={selectedMetric === 'headline' ? 'Headline' : 'Core'} />
                <Line type="monotone" dataKey="services" stroke="#f59e0b" strokeWidth={1.5}
                  strokeDasharray="5 5" dot={false} name="Services" opacity={0.7} />
                <Legend wrapperStyle={{ paddingTop: 16 }}
                  formatter={v => <span style={{ color: '#cbd5e1', fontSize: 12 }}>{v}</span>} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Component weights + component performance */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8">
              <h3 className="text-xl font-bold mb-6">HICP Component Weights</h3>
              <div className="space-y-4">
                {[
                  { name: 'Services',                     weight: 45.7, color: 'from-orange-500 to-amber-500' },
                  { name: 'Non-Energy Industrial Goods',  weight: 25.6, color: 'from-green-500 to-emerald-500' },
                  { name: 'Food, Alcohol & Tobacco',      weight: 19.3, color: 'from-red-500 to-rose-500' },
                  { name: 'Energy',                       weight: 9.4,  color: 'from-blue-500 to-cyan-500' },
                ].map((c, i) => (
                  <div key={i}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-slate-300 font-medium text-sm">{c.name}</span>
                      <span className="text-white font-bold">{c.weight}%</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-2.5 overflow-hidden">
                      <div className={`h-full bg-gradient-to-r ${c.color}`} style={{ width: `${c.weight}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6 pt-4 border-t border-slate-700">
                <p className="text-xs text-slate-500">
                  Source: ECB 2025 official weights. Dynamically updated from Eurostat SDMX-JSON
                  (invalidated each January when ECB publishes annual revision).
                </p>
              </div>
            </div>

            <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8">
              <h3 className="text-xl font-bold mb-4">Component Performance (Last 12 Months)</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data?.slice(-12) || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} stroke="#475569" />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} stroke="#475569" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
                    labelStyle={{ color: '#e2e8f0' }}
                    formatter={v => `${(+v).toFixed(2)}%`}
                  />
                  <Legend formatter={v => <span style={{ color: '#cbd5e1', fontSize: 11 }}>{v}</span>} />
                  <Bar dataKey="services"           fill="#f59e0b" name="Services" />
                  <Bar dataKey="nonEnergyGoods"     fill="#10b981" name="Non-Energy Goods" />
                  <Bar dataKey="foodAlcoholTobacco" fill="#ef4444" name="Food/Alcohol/Tobacco" />
                  <Bar dataKey="energy"             fill="#3b82f6" name="Energy" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center text-xs text-slate-600 pb-4">
            <p>
              Data: ECB/Eurostat HICP (official methodology) ·{' '}
              Forecast: ECB suite-of-models ensemble + Monte Carlo ·{' '}
              Uncertainty: {methodology === 'boe' ? 'Bank of England two-piece normal' : 'Symmetric normal bands'} ·{' '}
              March 2026
            </p>
          </div>
        </>
      )}
    </div>
  );
};

export default EurozoneInflationTool;
