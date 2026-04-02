import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { StepResult } from '@/lib/finlearn-engine';

interface PortfolioChartProps {
  history: StepResult[];
}

export function PortfolioChart({ history }: PortfolioChartProps) {
  const data = history.map(h => ({
    step: h.step,
    Portfolio: parseFloat(h.portfolioValue.toFixed(0)),
    ALPHA: parseFloat(h.prices.ALPHA.toFixed(2)),
    BETA: parseFloat(h.prices.BETA.toFixed(2)),
    GAMMA: parseFloat(h.prices.GAMMA.toFixed(2)),
  }));

  return (
    <div className="fintech-card h-full">
      <h3 className="text-sm font-semibold text-foreground mb-4">Portfolio & Asset Prices</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 30% 18%)" />
          <XAxis dataKey="step" stroke="hsl(215 20% 55%)" fontSize={11} />
          <YAxis yAxisId="left" stroke="hsl(215 20% 55%)" fontSize={11} />
          <YAxis yAxisId="right" orientation="right" stroke="hsl(215 20% 55%)" fontSize={11} />
          <Tooltip
            contentStyle={{
              background: 'hsl(222 41% 10%)',
              border: '1px solid hsl(222 30% 18%)',
              borderRadius: '8px',
              fontSize: 12,
              color: 'hsl(210 40% 92%)',
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line yAxisId="left" type="monotone" dataKey="Portfolio" stroke="hsl(187 94% 43%)" strokeWidth={2} dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="ALPHA" stroke="hsl(152 69% 45%)" strokeWidth={1.5} dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="BETA" stroke="hsl(38 92% 50%)" strokeWidth={1.5} dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="GAMMA" stroke="hsl(262 83% 58%)" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
