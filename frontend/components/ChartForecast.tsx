import React from 'react';
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { ForecastPoint } from '@/lib/types';

interface Props {
  data: ForecastPoint[];
}

export default function ChartForecast({ data }: Props) {
  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-ink-1 mb-4">NAV Forecast</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#748ffc" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#748ffc" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e9ecf5" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#8b95a8" tickLine={false} />
          <YAxis tick={{ fontSize: 10 }} stroke="#8b95a8" tickLine={false} domain={['auto', 'auto']} />
          <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e9ecf5', fontSize: 12 }} />
          <Area type="monotone" dataKey="upper" stroke="none" fill="url(#bandGrad)" />
          <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" />
          <Line type="monotone" dataKey="actual" stroke="#4263eb" strokeWidth={2} dot={false} name="Actual" />
          <Line type="monotone" dataKey="predicted" stroke="#f76707" strokeWidth={2} strokeDasharray="6 3" dot={false} name="Predicted" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
