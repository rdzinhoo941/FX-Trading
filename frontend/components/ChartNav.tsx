import React from 'react';
import {
  ResponsiveContainer, AreaChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { NavPoint } from '@/lib/types';

interface Props {
  data: NavPoint[];
  title?: string;
  showPeak?: boolean;
}

export default function ChartNav({ data, title = 'NAV', showPeak = true }: Props) {
  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-ink-1 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id="navGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4263eb" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#4263eb" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e9ecf5" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#8b95a8" tickLine={false} />
          <YAxis tick={{ fontSize: 10 }} stroke="#8b95a8" tickLine={false} domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ borderRadius: 8, border: '1px solid #e9ecf5', fontSize: 12 }}
          />
          <Area type="monotone" dataKey="nav" stroke="#4263eb" fill="url(#navGrad)" strokeWidth={2} dot={false} />
          {showPeak && (
            <Line type="monotone" dataKey="peak" stroke="#8b95a8" strokeDasharray="4 4" strokeWidth={1} dot={false} />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
