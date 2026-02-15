import React from 'react';
import { AllocationRow } from '@/lib/types';

interface Props {
  rows: AllocationRow[];
}

function fmt(n: number, prefix = '') {
  const s = Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const sign = n < 0 ? '-' : n > 0 ? '+' : '';
  return `${sign}${prefix}${s}`;
}

export default function PositionsTable({ rows }: Props) {
  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-surface-3">
        <h3 className="text-sm font-semibold text-ink-1">Positions</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-surface-1 text-ink-3 text-xs uppercase tracking-wider">
              <th className="px-5 py-3 text-left font-medium">Pair</th>
              <th className="px-5 py-3 text-right font-medium">Weight</th>
              <th className="px-5 py-3 text-right font-medium">Notional</th>
              <th className="px-5 py-3 text-right font-medium">Mkt Value</th>
              <th className="px-5 py-3 text-right font-medium">PnL Today</th>
              <th className="px-5 py-3 text-right font-medium">PnL Total</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-surface-2">
            {rows.map((r) => (
              <tr key={r.pair} className="hover:bg-surface-1 transition-colors">
                <td className="px-5 py-3 font-medium text-ink-0">{r.pair}</td>
                <td className="px-5 py-3 text-right font-mono text-ink-1">{r.weight.toFixed(1)}%</td>
                <td className="px-5 py-3 text-right font-mono text-ink-1">${r.notional.toLocaleString()}</td>
                <td className="px-5 py-3 text-right font-mono text-ink-1">${r.market_value.toLocaleString()}</td>
                <td className={`px-5 py-3 text-right font-mono ${r.pnl_today >= 0 ? 'text-up' : 'text-down'}`}>
                  {fmt(r.pnl_today, '$')}
                </td>
                <td className={`px-5 py-3 text-right font-mono ${r.pnl_total >= 0 ? 'text-up' : 'text-down'}`}>
                  {fmt(r.pnl_total, '$')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
