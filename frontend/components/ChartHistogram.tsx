import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { HistogramBin } from '@/lib/types';

interface Props {
  data: HistogramBin[];
  title?: string;
  color?: string;
}

export default function ChartHistogram({ data, title = 'Distribution', color = '#4263eb' }: Props) {
  const chartData = data.map((b) => ({
    range: ((b.bin_start + b.bin_end) / 2).toFixed(2),
    count: b.count,
  }));
  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-ink-1 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e9ecf5" />
          <XAxis dataKey="range" tick={{ fontSize: 9 }} stroke="#8b95a8" tickLine={false} />
          <YAxis tick={{ fontSize: 10 }} stroke="#8b95a8" tickLine={false} />
          <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e9ecf5', fontSize: 12 }} />
          <Bar dataKey="count" fill={color} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
