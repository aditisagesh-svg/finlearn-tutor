import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ComposedChart } from 'recharts';
import type { StepResult } from '@/lib/finlearn-engine';

interface RewardChartProps {
  history: StepResult[];
}

export function RewardChart({ history }: RewardChartProps) {
  let cumulative = 0;
  const data = history.slice(1).map(h => {
    cumulative += h.reward;
    return {
      step: h.step,
      reward: h.reward,
      cumulative: parseFloat(cumulative.toFixed(2)),
    };
  });

  return (
    <div className="fintech-card h-full">
      <h3 className="text-sm font-semibold text-foreground mb-4">Reward Analytics</h3>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 30% 18%)" />
          <XAxis dataKey="step" stroke="hsl(215 20% 55%)" fontSize={11} />
          <YAxis stroke="hsl(215 20% 55%)" fontSize={11} />
          <Tooltip
            contentStyle={{
              background: 'hsl(222 41% 10%)',
              border: '1px solid hsl(222 30% 18%)',
              borderRadius: '8px',
              fontSize: 12,
              color: 'hsl(210 40% 92%)',
            }}
          />
          <Bar dataKey="reward" fill="hsl(187 94% 43%)" opacity={0.6} radius={[3, 3, 0, 0]} />
          <Line type="monotone" dataKey="cumulative" stroke="hsl(38 92% 50%)" strokeWidth={2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
