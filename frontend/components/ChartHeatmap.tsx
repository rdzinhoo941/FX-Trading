import React from 'react';
import { CorrelationMatrix } from '@/lib/types';

interface Props {
  data: CorrelationMatrix;
}

function corrColor(v: number): string {
  if (v >= 0.6) return '#364fc7';
  if (v >= 0.3) return '#5c7cfa';
  if (v >= 0) return '#bac8ff';
  if (v >= -0.3) return '#ffa8a8';
  if (v >= -0.6) return '#ff6b6b';
  return '#c92a2a';
}

export default function ChartHeatmap({ data }: Props) {
  const n = data.pairs.length;
  const cell = 56;
  const labelW = 64;
  const w = labelW + n * cell;
  const h = labelW + n * cell;

  return (
    <div className="bg-white rounded-xl border border-surface-3 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-ink-1 mb-4">Correlation Matrix</h3>
      <div className="overflow-x-auto">
        <svg width={w} height={h} className="mx-auto">
          {/* Column labels */}
          {data.pairs.map((p, j) => (
            <text
              key={`cl-${j}`}
              x={labelW + j * cell + cell / 2}
              y={labelW - 8}
              textAnchor="middle"
              fontSize={9}
              fill="#5a6578"
              fontFamily="DM Sans"
            >
              {p}
            </text>
          ))}
          {/* Row labels + cells */}
          {data.pairs.map((p, i) => (
            <g key={`r-${i}`}>
              <text
                x={labelW - 6}
                y={labelW + i * cell + cell / 2 + 3}
                textAnchor="end"
                fontSize={9}
                fill="#5a6578"
                fontFamily="DM Sans"
              >
                {p}
              </text>
              {data.matrix[i].map((v, j) => (
                <g key={`c-${i}-${j}`}>
                  <rect
                    x={labelW + j * cell}
                    y={labelW + i * cell}
                    width={cell - 2}
                    height={cell - 2}
                    rx={4}
                    fill={corrColor(v)}
                    opacity={0.8}
                  />
                  <text
                    x={labelW + j * cell + cell / 2 - 1}
                    y={labelW + i * cell + cell / 2 + 3}
                    textAnchor="middle"
                    fontSize={10}
                    fill="#fff"
                    fontWeight={500}
                    fontFamily="JetBrains Mono"
                  >
                    {v.toFixed(2)}
                  </text>
                </g>
              ))}
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
