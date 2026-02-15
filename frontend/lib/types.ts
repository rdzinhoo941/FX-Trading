export interface AllocationRow {
  pair: string;
  weight: number;
  notional: number;
  market_value: number;
  pnl_today: number;
  pnl_total: number;
}

export interface KpiSummary {
  total_value: number;
  daily_pnl: number;
  cumulative_pnl: number;
  net_exposure_usd: number;
  concentration_hhi: number;
}

export interface ProfileResponse {
  session_id: string;
  allocation: AllocationRow[];
  kpi: KpiSummary;
}

export interface NavPoint {
  date: string;
  nav: number;
  peak: number;
}

export interface PortfolioSummary {
  allocation: AllocationRow[];
  kpi: KpiSummary;
  nav_series: NavPoint[];
}

export interface ForecastPoint {
  date: string;
  actual?: number | null;
  predicted?: number | null;
  upper?: number | null;
  lower?: number | null;
}

export interface ForecastMetrics {
  rmse: number;
  mae: number;
  r2: number;
  directional_accuracy: number;
}

export interface ForecastResponse {
  series: ForecastPoint[];
  metrics: ForecastMetrics;
  signal: string;
}

export interface RiskCards {
  var_95: number;
  var_99: number;
  es_95: number;
  es_99: number;
  annual_vol: number;
  max_drawdown: number;
}

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface VolPoint {
  date: string;
  vol: number;
}

export interface RiskResponse {
  cards: RiskCards;
  histogram: HistogramBin[];
  vol_series: VolPoint[];
}

export interface ScenarioRow {
  pair: string;
  weight_before: number;
  pnl_before: number;
  pnl_after: number;
  contribution: number;
}

export interface ScenarioResponse {
  rows: ScenarioRow[];
  total_pnl_before: number;
  total_pnl_after: number;
}

export interface BacktestCards {
  ann_return: number;
  ann_vol: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
}

export interface BacktestResponse {
  nav_series: NavPoint[];
  return_histogram: HistogramBin[];
  cards: BacktestCards;
}

export interface CorrelationMatrix {
  pairs: string[];
  matrix: number[][];
}

export interface ScatterPoint {
  pair: string;
  ann_return: number;
  ann_vol: number;
}

export interface CorrelationResponse {
  matrix: CorrelationMatrix;
  scatter: ScatterPoint[];
}
