import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface AllocationPieProps {
  holdings: { ALPHA: number; BETA: number; GAMMA: number };
  prices: { ALPHA: number; BETA: number; GAMMA: number };
  cash: number;
}

const COLORS = ['hsl(152 69% 45%)', 'hsl(38 92% 50%)', 'hsl(262 83% 58%)', 'hsl(215 20% 55%)'];

export function AllocationPie({ holdings, prices, cash }: AllocationPieProps) {
  const data = [
    { name: 'ALPHA', value: parseFloat((holdings.ALPHA * prices.ALPHA).toFixed(0)) },
    { name: 'BETA', value: parseFloat((holdings.BETA * prices.BETA).toFixed(0)) },
    { name: 'GAMMA', value: parseFloat((holdings.GAMMA * prices.GAMMA).toFixed(0)) },
    { name: 'Cash', value: parseFloat(cash.toFixed(0)) },
  ];

  return (
    <div className="fintech-card h-full">
      <h3 className="text-sm font-semibold text-foreground mb-4">Asset Allocation</h3>
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={3} dataKey="value">
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i]} stroke="none" />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: 'hsl(222 41% 10%)',
              border: '1px solid hsl(222 30% 18%)',
              borderRadius: '8px',
              fontSize: 12,
              color: 'hsl(210 40% 92%)',
            }}
            formatter={(value: number) => [`$${value.toLocaleString()}`, '']}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
