import React from 'react';
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell, Label,
} from 'recharts';
import { ScatterPoint } from '@/lib/types';

interface Props {
  data: ScatterPoint[];
}

const COLORS = ['#4263eb', '#f76707', '#0ca678', '#ae3ec9', '#e8590c', '#1098ad', '#d6336c'];

export default function ChartScatter({ data }: Props) {
  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-ink-1 mb-4">Risk / Return</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e9ecf5" />
          <XAxis type="number" dataKey="ann_vol" tick={{ fontSize: 10 }} stroke="#8b95a8">
            <Label value="Ann. Vol (%)" position="bottom" offset={0} style={{ fontSize: 11, fill: '#8b95a8' }} />
          </XAxis>
          <YAxis type="number" dataKey="ann_return" tick={{ fontSize: 10 }} stroke="#8b95a8">
            <Label value="Ann. Return (%)" angle={-90} position="left" offset={-4} style={{ fontSize: 11, fill: '#8b95a8' }} />
          </YAxis>
          <Tooltip
            formatter={(v: number) => `${v.toFixed(2)}%`}
            contentStyle={{ borderRadius: 8, border: '1px solid #e9ecf5', fontSize: 12 }}
            labelFormatter={(_, payload) => {
              const pt = payload?.[0]?.payload;
              return pt?.pair || '';
            }}
          />
          <Scatter data={data} fill="#4263eb">
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} r={7} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
