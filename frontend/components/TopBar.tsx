import React from 'react';

interface Props {
  pairs: string[];
  selectedPair: string;
  onPairChange: (p: string) => void;
  horizon: string;
  onHorizonChange: (h: string) => void;
  mode: string;
  onModeChange: (m: string) => void;
  riskLevel: string;
  onRiskLevelChange: (r: string) => void;
  onRefresh: () => void;
  onRebalance: () => void;
  onRecompute: () => void;
  loading?: boolean;
}

export default function TopBar(props: Props) {
  const selectCls =
    'bg-surface-1 border border-surface-3 rounded-lg px-3 py-2 text-sm text-ink-0 focus:outline-none focus:ring-2 focus:ring-brand-400 focus:border-brand-400';
  const btnCls =
    'px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50';

  return (
    <div className="bg-white border border-surface-3 rounded-xl shadow-sm px-5 py-3 flex flex-wrap items-center gap-3">
      {/* Pair selector */}
      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-ink-3 uppercase tracking-wider">Pair</label>
        <select className={selectCls} value={props.selectedPair} onChange={(e) => props.onPairChange(e.target.value)}>
          {props.pairs.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </div>

      {/* Horizon */}
      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-ink-3 uppercase tracking-wider">Horizon</label>
        <select className={selectCls} value={props.horizon} onChange={(e) => props.onHorizonChange(e.target.value)}>
          <option value="1">1 Day</option>
          <option value="5">5 Days</option>
          <option value="30">30 Days</option>
        </select>
      </div>

      {/* Mode */}
      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-ink-3 uppercase tracking-wider">Mode</label>
        <select className={selectCls} value={props.mode} onChange={(e) => props.onModeChange(e.target.value)}>
          <option value="real">Real</option>
          <option value="backtest">Backtest</option>
        </select>
      </div>

      {/* Risk level */}
      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-ink-3 uppercase tracking-wider">Risk</label>
        <select className={selectCls} value={props.riskLevel} onChange={(e) => props.onRiskLevelChange(e.target.value)}>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>
      </div>

      <div className="flex-1" />

      {/* Action buttons */}
      <button
        className={`${btnCls} bg-surface-1 text-ink-1 hover:bg-surface-2 border border-surface-3`}
        onClick={props.onRefresh}
        disabled={props.loading}
      >
        Refresh Data
      </button>
      <button
        className={`${btnCls} bg-brand-600 text-white hover:bg-brand-700`}
        onClick={props.onRebalance}
        disabled={props.loading}
      >
        Rebalance
      </button>
      <button
        className={`${btnCls} bg-brand-800 text-white hover:bg-brand-900`}
        onClick={props.onRecompute}
        disabled={props.loading}
      >
        Recompute Forecasts
      </button>
    </div>
  );
}
