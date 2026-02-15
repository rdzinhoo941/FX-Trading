import React from 'react';

export const TAB_LIST = ['Portfolio', 'AI Forecasts', 'Risk', 'Scenarios', 'Backtest', 'Correlations'] as const;
export type TabName = (typeof TAB_LIST)[number];

interface Props {
  active: TabName;
  onChange: (t: TabName) => void;
}

export default function Tabs({ active, onChange }: Props) {
  return (
    <div className="flex gap-1 bg-surface-2 rounded-lg p-1">
      {TAB_LIST.map((t) => (
        <button
          key={t}
          onClick={() => onChange(t)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            active === t
              ? 'bg-white text-brand-700 shadow-sm'
              : 'text-ink-2 hover:text-ink-0 hover:bg-surface-0/50'
          }`}
        >
          {t}
        </button>
      ))}
    </div>
  );
}
