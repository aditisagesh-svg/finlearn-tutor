import { DollarSign, Wallet, TrendingUp, Brain } from 'lucide-react';

interface KpiCardsProps {
  portfolioValue: number;
  cash: number;
  initialValue: number;
  learningScore: number;
}

function formatCurrency(n: number) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(n);
}

function KpiCard({ icon: Icon, label, value, subValue, color }: {
  icon: typeof DollarSign;
  label: string;
  value: string;
  subValue?: string;
  color: string;
}) {
  return (
    <div className="fintech-card-glow flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{label}</span>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center`} style={{ background: `hsl(var(--${color}) / 0.15)` }}>
          <Icon className="w-4 h-4" style={{ color: `hsl(var(--${color}))` }} />
        </div>
      </div>
      <p className="kpi-value text-foreground">{value}</p>
      {subValue && <p className="text-xs text-muted-foreground">{subValue}</p>}
    </div>
  );
}

export function KpiCards({ portfolioValue, cash, initialValue, learningScore }: KpiCardsProps) {
  const pnl = portfolioValue - initialValue;
  const pnlPct = ((pnl / initialValue) * 100).toFixed(1);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <KpiCard icon={DollarSign} label="Portfolio Value" value={formatCurrency(portfolioValue)} color="primary" />
      <KpiCard icon={Wallet} label="Cash Balance" value={formatCurrency(cash)} color="chart-5" />
      <KpiCard
        icon={TrendingUp}
        label="Profit / Loss"
        value={formatCurrency(pnl)}
        subValue={`${pnl >= 0 ? '+' : ''}${pnlPct}%`}
        color={pnl >= 0 ? 'success' : 'destructive'}
      />
      <KpiCard icon={Brain} label="Learning Score" value={`${learningScore.toFixed(1)}`} subValue="/ 100" color="warning" />
    </div>
  );
}
