import React from 'react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { VolPoint } from '@/lib/types';

interface Props {
  data: VolPoint[];
}

export default function ChartVol({ data }: Props) {
  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-ink-1 mb-4">Realized Volatility (%)</h3>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ae3ec9" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#ae3ec9" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e9ecf5" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#8b95a8" tickLine={false} />
          <YAxis tick={{ fontSize: 10 }} stroke="#8b95a8" tickLine={false} />
          <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e9ecf5', fontSize: 12 }} />
          <Area type="monotone" dataKey="vol" stroke="#ae3ec9" fill="url(#volGrad)" strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
