import React from 'react';

interface Props {
  label: string;
  value: string;
  sub?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export default function KpiCard({ label, value, sub, trend }: Props) {
  const trendColor = trend === 'up' ? 'text-up' : trend === 'down' ? 'text-down' : 'text-ink-2';
  return (
    <div className="bg-white rounded-xl border border-surface-3 p-5 flex flex-col gap-1 shadow-sm hover:shadow-md transition-shadow">
      <span className="text-xs font-medium text-ink-3 uppercase tracking-wider">{label}</span>
      <span className={`text-2xl font-bold font-mono ${trendColor}`}>{value}</span>
      {sub && <span className="text-xs text-ink-3">{sub}</span>}
    </div>
  );
}
