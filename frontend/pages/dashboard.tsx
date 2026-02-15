import React, { useState, useEffect, useCallback } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { apiGet, apiPost, getSessionId } from '@/lib/api';
import {
  PortfolioSummary, ForecastResponse, RiskResponse,
  ScenarioResponse, BacktestResponse, CorrelationResponse,
  ProfileResponse,
} from '@/lib/types';

import Tabs, { TabName } from '@/components/Tabs';
import TopBar from '@/components/TopBar';
import KpiCard from '@/components/KpiCard';
import PositionsTable from '@/components/PositionsTable';
import ChartNav from '@/components/ChartNav';
import ChartDonut from '@/components/ChartDonut';
import ChartForecast from '@/components/ChartForecast';
import ChartHistogram from '@/components/ChartHistogram';
import ChartHeatmap from '@/components/ChartHeatmap';
import ChartScatter from '@/components/ChartScatter';
import ChartVol from '@/components/ChartVol';

function fmtUsd(n: number) {
  return '$' + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtPnl(n: number) {
  const sign = n >= 0 ? '+' : '-';
  return sign + fmtUsd(n);
}
function fmtPct(n: number) {
  return n.toFixed(2) + '%';
}

const SIGNAL_STYLES: Record<string, { label: string; cls: string }> = {
  strong_buy: { label: 'STRONG BUY', cls: 'bg-green-600 text-white' },
  buy: { label: 'BUY', cls: 'bg-green-500 text-white' },
  neutral: { label: 'NEUTRAL', cls: 'bg-gray-400 text-white' },
  sell: { label: 'SELL', cls: 'bg-red-500 text-white' },
  strong_sell: { label: 'STRONG SELL', cls: 'bg-red-700 text-white' },
};

const SCENARIOS = [
  { key: 'usd_crash', label: 'USD Crash' },
  { key: 'jpy_spike', label: 'JPY Spike' },
  { key: 'risk_off', label: 'Risk Off' },
  { key: 'em_selloff', label: 'EM Sell-Off' },
];

export default function DashboardPage() {
  const router = useRouter();
  const [sid, setSid] = useState('');
  const [tab, setTab] = useState<TabName>('Portfolio');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Top bar state
  const [pairs, setPairs] = useState<string[]>([]);
  const [selectedPair, setSelectedPair] = useState('');
  const [horizon, setHorizon] = useState('1');
  const [mode, setMode] = useState('real');
  const [riskLevel, setRiskLevel] = useState('medium');

  // Tab data
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [scenario, setScenario] = useState<ScenarioResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [corr, setCorr] = useState<CorrelationResponse | null>(null);

  // Scenario form
  const [scenType, setScenType] = useState('custom');
  const [scenPair, setScenPair] = useState('');
  const [scenShock, setScenShock] = useState(-5);

  useEffect(() => {
    const s = getSessionId();
    if (!s) { router.replace('/'); return; }
    setSid(s);
  }, [router]);

  const fetchPortfolio = useCallback(async () => {
    if (!sid) return;
    setLoading(true);
    try {
      const data = await apiGet<PortfolioSummary>('/api/portfolio/summary', { session_id: sid });
      setPortfolio(data);
      const p = data.allocation.map((a) => a.pair);
      setPairs(p);
      if (!selectedPair && p.length) setSelectedPair(p[0]);
      if (!scenPair && p.length) setScenPair(p[0]);
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, [sid, selectedPair, scenPair]);

  const fetchForecast = useCallback(async () => {
    if (!sid) return;
    setLoading(true);
    try {
      setForecast(await apiGet<ForecastResponse>('/api/forecasts', { session_id: sid }));
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, [sid]);

  const fetchRisk = useCallback(async () => {
    if (!sid) return;
    setLoading(true);
    try {
      setRisk(await apiGet<RiskResponse>('/api/risk/metrics', { session_id: sid }));
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, [sid]);

  const fetchBacktest = useCallback(async () => {
    if (!sid) return;
    setLoading(true);
    try {
      setBacktest(await apiGet<BacktestResponse>('/api/backtest/results', { session_id: sid }));
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, [sid]);

  const fetchCorrelations = useCallback(async () => {
    if (!sid) return;
    setLoading(true);
    try {
      setCorr(await apiGet<CorrelationResponse>('/api/correlations/matrix', { session_id: sid }));
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, [sid]);

  // Initial data load
  useEffect(() => { if (sid) fetchPortfolio(); }, [sid, fetchPortfolio]);

  // Load tab data when switching
  useEffect(() => {
    if (!sid) return;
    if (tab === 'AI Forecasts' && !forecast) fetchForecast();
    if (tab === 'Risk' && !risk) fetchRisk();
    if (tab === 'Backtest' && !backtest) fetchBacktest();
    if (tab === 'Correlations' && !corr) fetchCorrelations();
  }, [tab, sid, forecast, risk, backtest, corr, fetchForecast, fetchRisk, fetchBacktest, fetchCorrelations]);

  // Actions
  const handleRefresh = () => {
    setPortfolio(null); setForecast(null); setRisk(null); setBacktest(null); setCorr(null); setScenario(null);
    fetchPortfolio();
  };

  const handleRebalance = async () => {
    if (!sid) return;
    setLoading(true);
    try {
      const data = await apiPost<ProfileResponse>('/portfolio/rebalance', { session_id: sid });
      setPortfolio(null); setForecast(null); setRisk(null);
      fetchPortfolio();
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  };

  const handleRecompute = async () => {
    if (!sid) return;
    setLoading(true);
    try {
      setForecast(await apiPost<ForecastResponse>('/forecasts/recompute', { session_id: sid }));
      setTab('AI Forecasts');
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  };

  const handleScenario = async () => {
    if (!sid) return;
    setLoading(true);
    try {
      setScenario(await apiPost<ScenarioResponse>('/scenarios/apply', {
        session_id: sid,
        scenario_type: scenType,
        pair: scenType === 'custom' ? scenPair : undefined,
        shock_pct: scenType === 'custom' ? scenShock : undefined,
      }));
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  };

  // ── Render helpers ──

  const renderPortfolio = () => {
    if (!portfolio) return <LoadingState />;
    const k = portfolio.kpi;
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-5 gap-3">
          <KpiCard label="Total Value" value={fmtUsd(k.total_value)} />
          <KpiCard label="Daily PnL" value={fmtPnl(k.daily_pnl)} trend={k.daily_pnl >= 0 ? 'up' : 'down'} />
          <KpiCard label="Cumulative PnL" value={fmtPnl(k.cumulative_pnl)} trend={k.cumulative_pnl >= 0 ? 'up' : 'down'} />
          <KpiCard label="Net Exposure" value={fmtUsd(k.net_exposure_usd)} />
          <KpiCard label="HHI" value={k.concentration_hhi.toFixed(4)} />
        </div>
        <div className="grid grid-cols-3 gap-5">
          <div className="col-span-2">
            <ChartNav data={portfolio.nav_series} title="NAV (Indexed)" />
          </div>
          <ChartDonut data={portfolio.allocation} />
        </div>
        <PositionsTable rows={portfolio.allocation} />
      </div>
    );
  };

  const renderForecasts = () => {
    if (!forecast) return <LoadingState />;
    const m = forecast.metrics;
    const sig = SIGNAL_STYLES[forecast.signal] || SIGNAL_STYLES.neutral;
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-5 gap-3">
          <KpiCard label="RMSE" value={m.rmse.toFixed(2)} />
          <KpiCard label="MAE" value={m.mae.toFixed(2)} />
          <KpiCard label="R²" value={m.r2.toFixed(4)} />
          <KpiCard label="Dir. Accuracy" value={fmtPct(m.directional_accuracy * 100)} />
          <div className="bg-white rounded-xl border border-surface-3 p-5 flex flex-col items-center justify-center gap-1 shadow-sm">
            <span className="text-xs font-medium text-ink-3 uppercase tracking-wider">Signal</span>
            <span className={`text-sm font-bold px-3 py-1 rounded-full ${sig.cls}`}>{sig.label}</span>
          </div>
        </div>
        <ChartForecast data={forecast.series} />
      </div>
    );
  };

  const renderRisk = () => {
    if (!risk) return <LoadingState />;
    const c = risk.cards;
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-6 gap-3">
          <KpiCard label="VaR 95" value={fmtUsd(c.var_95)} trend="down" />
          <KpiCard label="VaR 99" value={fmtUsd(c.var_99)} trend="down" />
          <KpiCard label="ES 95" value={fmtUsd(c.es_95)} trend="down" />
          <KpiCard label="ES 99" value={fmtUsd(c.es_99)} trend="down" />
          <KpiCard label="Ann. Vol" value={fmtPct(c.annual_vol)} />
          <KpiCard label="Max DD" value={fmtPct(c.max_drawdown)} trend="down" />
        </div>
        <div className="grid grid-cols-2 gap-5">
          <ChartHistogram data={risk.histogram} title="Return Distribution (%)" color="#5c7cfa" />
          <ChartVol data={risk.vol_series} />
        </div>
      </div>
    );
  };

  const renderScenarios = () => {
    const selectCls = 'bg-surface-1 border border-surface-3 rounded-lg px-3 py-2 text-sm text-ink-0 focus:outline-none focus:ring-2 focus:ring-brand-400';
    return (
      <div className="space-y-5">
        {/* Controls */}
        <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
          <h3 className="text-sm font-semibold text-ink-1 mb-4">Stress Test</h3>
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1">Scenario</label>
              <select className={selectCls} value={scenType} onChange={(e) => setScenType(e.target.value)}>
                <option value="custom">Custom</option>
                {SCENARIOS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
              </select>
            </div>
            {scenType === 'custom' && (
              <>
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1">Pair</label>
                  <select className={selectCls} value={scenPair} onChange={(e) => setScenPair(e.target.value)}>
                    {pairs.map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1">Shock %</label>
                  <input type="number" className={selectCls + ' w-24'} value={scenShock} onChange={(e) => setScenShock(Number(e.target.value))} />
                </div>
              </>
            )}
            <button
              onClick={handleScenario}
              disabled={loading}
              className="px-5 py-2 rounded-lg bg-brand-700 text-white text-sm font-medium hover:bg-brand-800 disabled:opacity-50 transition"
            >
              Apply
            </button>
          </div>
        </div>

        {/* Results */}
        {scenario && (
          <div className="bg-white rounded-xl border border-surface-3 shadow-sm overflow-hidden">
            <div className="px-5 py-4 border-b border-surface-3 flex items-center gap-6">
              <span className="text-sm text-ink-2">Before: <span className="font-mono font-semibold text-ink-0">{fmtPnl(scenario.total_pnl_before)}</span></span>
              <span className="text-sm text-ink-2">→</span>
              <span className="text-sm text-ink-2">After: <span className={`font-mono font-semibold ${scenario.total_pnl_after >= 0 ? 'text-up' : 'text-down'}`}>{fmtPnl(scenario.total_pnl_after)}</span></span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-surface-1 text-ink-3 text-xs uppercase tracking-wider">
                    <th className="px-5 py-3 text-left font-medium">Pair</th>
                    <th className="px-5 py-3 text-right font-medium">Weight</th>
                    <th className="px-5 py-3 text-right font-medium">PnL Before</th>
                    <th className="px-5 py-3 text-right font-medium">PnL After</th>
                    <th className="px-5 py-3 text-right font-medium">Contribution</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-2">
                  {scenario.rows.map((r) => (
                    <tr key={r.pair} className="hover:bg-surface-1 transition-colors">
                      <td className="px-5 py-3 font-medium">{r.pair}</td>
                      <td className="px-5 py-3 text-right font-mono">{r.weight_before.toFixed(1)}%</td>
                      <td className="px-5 py-3 text-right font-mono">{fmtPnl(r.pnl_before)}</td>
                      <td className={`px-5 py-3 text-right font-mono ${r.pnl_after >= 0 ? 'text-up' : 'text-down'}`}>{fmtPnl(r.pnl_after)}</td>
                      <td className={`px-5 py-3 text-right font-mono ${r.contribution >= 0 ? 'text-up' : 'text-down'}`}>{fmtPnl(r.contribution)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderBacktest = () => {
    if (!backtest) return <LoadingState />;
    const c = backtest.cards;
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-6 gap-3">
          <KpiCard label="Ann. Return" value={fmtPct(c.ann_return)} trend={c.ann_return >= 0 ? 'up' : 'down'} />
          <KpiCard label="Ann. Vol" value={fmtPct(c.ann_vol)} />
          <KpiCard label="Sharpe" value={c.sharpe.toFixed(2)} />
          <KpiCard label="Sortino" value={c.sortino.toFixed(2)} />
          <KpiCard label="Calmar" value={c.calmar.toFixed(2)} />
          <KpiCard label="Max DD" value={fmtPct(c.max_drawdown)} trend="down" />
        </div>
        <div className="grid grid-cols-2 gap-5">
          <ChartNav data={backtest.nav_series} title="Backtest NAV (Indexed)" showPeak />
          <ChartHistogram data={backtest.return_histogram} title="Strategy Returns (%)" color="#0ca678" />
        </div>
      </div>
    );
  };

  const renderCorrelations = () => {
    if (!corr) return <LoadingState />;
    return (
      <div className="grid grid-cols-2 gap-5">
        <ChartHeatmap data={corr.matrix} />
        <ChartScatter data={corr.scatter} />
      </div>
    );
  };

  const renderTab = () => {
    switch (tab) {
      case 'Portfolio': return renderPortfolio();
      case 'AI Forecasts': return renderForecasts();
      case 'Risk': return renderRisk();
      case 'Scenarios': return renderScenarios();
      case 'Backtest': return renderBacktest();
      case 'Correlations': return renderCorrelations();
    }
  };

  return (
    <>
      <Head><title>Dashboard – FX Optimizer</title></Head>
      <div className="min-h-screen">
        {/* Header */}
        <header className="bg-white border-b border-surface-3 px-6 py-3 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-brand-700 flex items-center justify-center">
              <span className="text-white font-bold text-xs">FX</span>
            </div>
            <span className="font-bold text-ink-0">Portfolio Optimizer</span>
          </div>
          <div className="flex-1" />
          <button
            onClick={() => router.push('/')}
            className="text-sm text-ink-3 hover:text-ink-0 transition"
          >
            ← New Profile
          </button>
        </header>

        <main className="max-w-[1400px] mx-auto px-6 py-5 space-y-5">
          {/* Error banner */}
          {error && (
            <div className="px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex items-center justify-between">
              <span>{error}</span>
              <button onClick={() => setError('')} className="text-red-400 hover:text-red-600">✕</button>
            </div>
          )}

          {/* Top bar */}
          <TopBar
            pairs={pairs}
            selectedPair={selectedPair}
            onPairChange={setSelectedPair}
            horizon={horizon}
            onHorizonChange={setHorizon}
            mode={mode}
            onModeChange={setMode}
            riskLevel={riskLevel}
            onRiskLevelChange={setRiskLevel}
            onRefresh={handleRefresh}
            onRebalance={handleRebalance}
            onRecompute={handleRecompute}
            loading={loading}
          />

          {/* Tabs */}
          <Tabs active={tab} onChange={setTab} />

          {/* Tab content */}
          <div>{renderTab()}</div>
        </main>
      </div>
    </>
  );
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-3 border-brand-300 border-t-brand-700 rounded-full animate-spin" />
        <span className="text-sm text-ink-3">Loading data…</span>
      </div>
    </div>
  );
}
