import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { apiPost, setSessionId } from '@/lib/api';
import { ProfileResponse, AllocationRow, KpiSummary } from '@/lib/types';
import KpiCard from '@/components/KpiCard';
import PositionsTable from '@/components/PositionsTable';

const HORIZON_OPTIONS = ['1M', '3M', '6M', '1Y', 'custom'] as const;
const MODEL_OPTIONS = ['markowitz', 'entropy_pooling', 'equal_weight', 'risk_parity'] as const;
const RISK_OPTIONS = ['low', 'medium', 'high'] as const;
const UNIVERSE_OPTIONS = ['majors', 'minors', 'exotics', 'mix'] as const;

const MODEL_LABELS: Record<string, string> = {
  markowitz: 'Markowitz MVO',
  entropy_pooling: 'Entropy Pooling',
  equal_weight: 'Equal Weight',
  risk_parity: 'Risk Parity',
};

function fmtUsd(n: number) {
  return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function WizardPage() {
  const router = useRouter();
  const [step, setStep] = useState<'form' | 'result'>('form');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // form state
  const [capital, setCapital] = useState(100000);
  const [horizon, setHorizon] = useState('3M');
  const [model, setModel] = useState('markowitz');
  const [targetReturn, setTargetReturn] = useState(8);
  const [riskLevel, setRiskLevel] = useState('medium');
  const [universe, setUniverse] = useState('majors');

  // result
  const [allocation, setAllocation] = useState<AllocationRow[]>([]);
  const [kpi, setKpi] = useState<KpiSummary | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await apiPost<ProfileResponse>('/profile/generate_portfolio', {
        initial_capital: capital,
        horizon,
        optimization_model: model,
        target_return_pct: targetReturn,
        risk_aversion_level: riskLevel,
        fx_universe: universe,
      });
      setSessionId(data.session_id);
      setAllocation(data.allocation);
      setKpi(data.kpi);
      setStep('result');
    } catch (e: any) {
      setError(e.message || 'Failed to generate portfolio');
    } finally {
      setLoading(false);
    }
  };

  const selectCls =
    'w-full bg-surface-1 border border-surface-3 rounded-lg px-4 py-3 text-sm text-ink-0 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 transition';

  return (
    <>
      <Head>
        <title>FX Portfolio Optimizer</title>
      </Head>

      <div className="min-h-screen flex items-center justify-center p-6">
        {/* Decorative background */}
        <div className="fixed inset-0 -z-10">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full bg-brand-200/20 blur-[120px]" />
          <div className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full bg-brand-300/10 blur-[100px]" />
        </div>

        <div className="w-full max-w-2xl">
          {/* Logo / Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 mb-3">
              <div className="w-9 h-9 rounded-lg bg-brand-700 flex items-center justify-center">
                <span className="text-white font-bold text-sm">FX</span>
              </div>
              <span className="text-xl font-bold text-ink-0">Portfolio Optimizer</span>
            </div>
            <p className="text-ink-2 text-sm">Configure your investment profile to generate an optimized FX portfolio.</p>
          </div>

          {step === 'form' ? (
            <div className="bg-white rounded-2xl border border-surface-3 shadow-lg p-8">
              <h2 className="text-lg font-semibold text-ink-0 mb-6">Investment Profile</h2>

              <div className="grid grid-cols-2 gap-5">
                {/* Capital */}
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1.5">
                    Initial Capital (USD)
                  </label>
                  <input
                    type="number"
                    className={selectCls}
                    value={capital}
                    onChange={(e) => setCapital(Number(e.target.value))}
                    min={1000}
                  />
                </div>

                {/* Horizon */}
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1.5">
                    Horizon
                  </label>
                  <select className={selectCls} value={horizon} onChange={(e) => setHorizon(e.target.value)}>
                    {HORIZON_OPTIONS.map((h) => (
                      <option key={h} value={h}>{h}</option>
                    ))}
                  </select>
                </div>

                {/* Model */}
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1.5">
                    Optimization Model
                  </label>
                  <select className={selectCls} value={model} onChange={(e) => setModel(e.target.value)}>
                    {MODEL_OPTIONS.map((m) => (
                      <option key={m} value={m}>{MODEL_LABELS[m]}</option>
                    ))}
                  </select>
                </div>

                {/* Target return */}
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1.5">
                    Target Return (%)
                  </label>
                  <input
                    type="number"
                    className={selectCls}
                    value={targetReturn}
                    onChange={(e) => setTargetReturn(Number(e.target.value))}
                    min={0}
                    max={100}
                    step={0.5}
                  />
                </div>

                {/* Risk */}
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1.5">
                    Risk Aversion
                  </label>
                  <select className={selectCls} value={riskLevel} onChange={(e) => setRiskLevel(e.target.value)}>
                    {RISK_OPTIONS.map((r) => (
                      <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                    ))}
                  </select>
                </div>

                {/* Universe */}
                <div>
                  <label className="block text-xs font-medium text-ink-3 uppercase tracking-wider mb-1.5">
                    FX Universe
                  </label>
                  <select className={selectCls} value={universe} onChange={(e) => setUniverse(e.target.value)}>
                    {UNIVERSE_OPTIONS.map((u) => (
                      <option key={u} value={u}>{u.charAt(0).toUpperCase() + u.slice(1)}</option>
                    ))}
                  </select>
                </div>
              </div>

              {error && (
                <div className="mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">{error}</div>
              )}

              <button
                onClick={handleGenerate}
                disabled={loading}
                className="mt-6 w-full py-3 rounded-lg bg-brand-700 text-white font-semibold text-sm hover:bg-brand-800 transition disabled:opacity-50"
              >
                {loading ? 'Generating…' : 'Generate Portfolio'}
              </button>
            </div>
          ) : (
            <div className="space-y-5">
              {/* KPI cards */}
              {kpi && (
                <div className="grid grid-cols-5 gap-3">
                  <KpiCard label="Total Value" value={fmtUsd(kpi.total_value)} />
                  <KpiCard label="Daily PnL" value={fmtUsd(kpi.daily_pnl)} trend={kpi.daily_pnl >= 0 ? 'up' : 'down'} />
                  <KpiCard label="Cumulative PnL" value={fmtUsd(kpi.cumulative_pnl)} trend={kpi.cumulative_pnl >= 0 ? 'up' : 'down'} />
                  <KpiCard label="Net Exposure" value={fmtUsd(kpi.net_exposure_usd)} />
                  <KpiCard label="HHI" value={kpi.concentration_hhi.toFixed(4)} />
                </div>
              )}

              <PositionsTable rows={allocation} />

              <div className="flex gap-3">
                <button
                  onClick={() => setStep('form')}
                  className="flex-1 py-3 rounded-lg bg-surface-2 text-ink-1 font-semibold text-sm hover:bg-surface-3 transition border border-surface-3"
                >
                  ← Edit Profile
                </button>
                <button
                  onClick={() => router.push('/dashboard')}
                  className="flex-1 py-3 rounded-lg bg-brand-700 text-white font-semibold text-sm hover:bg-brand-800 transition"
                >
                  Enter Dashboard →
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
