import React from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import { AllocationRow } from '@/lib/types';

interface Props {
  data: AllocationRow[];
}

const COLORS = ['#4263eb', '#5c7cfa', '#748ffc', '#91a7ff', '#bac8ff', '#dbe4ff', '#364fc7'];

export default function ChartDonut({ data }: Props) {
  const chartData = data.map((r) => ({ name: r.pair, value: r.weight }));
  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-ink-1 mb-4">Allocation</h3>
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={100}
            paddingAngle={2}
            dataKey="value"
            stroke="none"
          >
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            formatter={(v: number) => `${v.toFixed(1)}%`}
            contentStyle={{ borderRadius: 8, border: '1px solid #e9ecf5', fontSize: 12 }}
          />
          <Legend
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: 11 }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
