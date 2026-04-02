import { AlertTriangle, Activity, Target } from 'lucide-react';

interface RiskPanelProps {
  volatility: number;
  drawdown: number;
  concentration: number;
}

function RiskMetric({ icon: Icon, label, value, unit, level }: {
  icon: typeof AlertTriangle;
  label: string;
  value: number;
  unit: string;
  level: 'low' | 'medium' | 'high';
}) {
  const colors = {
    low: 'text-success',
    medium: 'text-warning',
    high: 'text-destructive',
  };
  const barColors = {
    low: 'bg-success',
    medium: 'bg-warning',
    high: 'bg-destructive',
  };
  const pct = Math.min(100, (value / (unit === '%' ? 30 : 1)) * 100);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className={`w-4 h-4 ${colors[level]}`} />
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
        <span className={`mono text-sm font-semibold ${colors[level]}`}>
          {value}{unit}
        </span>
      </div>
      <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${barColors[level]} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function RiskPanel({ volatility, drawdown, concentration }: RiskPanelProps) {
  return (
    <div className="fintech-card h-full flex flex-col gap-5">
      <h3 className="text-sm font-semibold text-foreground">Risk Metrics</h3>
      <RiskMetric
        icon={Activity}
        label="Volatility"
        value={volatility}
        unit="%"
        level={volatility < 3 ? 'low' : volatility < 8 ? 'medium' : 'high'}
      />
      <RiskMetric
        icon={AlertTriangle}
        label="Max Drawdown"
        value={drawdown}
        unit="%"
        level={drawdown < 5 ? 'low' : drawdown < 15 ? 'medium' : 'high'}
      />
      <RiskMetric
        icon={Target}
        label="Concentration"
        value={concentration}
        unit=""
        level={concentration < 0.5 ? 'low' : concentration < 0.7 ? 'medium' : 'high'}
      />
    </div>
  );
}
