import type { StepResult } from '@/lib/finlearn-engine';

interface ActionLogProps {
  history: StepResult[];
}

export function ActionLog({ history }: ActionLogProps) {
  const rows = [...history].reverse().slice(0, 15);

  return (
    <div className="fintech-card overflow-hidden">
      <h3 className="text-sm font-semibold text-foreground mb-4">Action Log</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-2 px-3 text-muted-foreground font-medium">Step</th>
              <th className="text-left py-2 px-3 text-muted-foreground font-medium">Action</th>
              <th className="text-right py-2 px-3 text-muted-foreground font-medium">Reward</th>
              <th className="text-right py-2 px-3 text-muted-foreground font-medium">Portfolio</th>
              <th className="text-left py-2 px-3 text-muted-foreground font-medium">AI Reasoning</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.step} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
                <td className="py-2 px-3 mono text-muted-foreground">{r.step}</td>
                <td className="py-2 px-3">
                  <span className={`inline-flex px-2 py-0.5 rounded text-xs font-medium ${
                    r.action.startsWith('BUY') ? 'bg-success/15 text-success' :
                    r.action.startsWith('SELL') ? 'bg-destructive/15 text-destructive' :
                    'bg-secondary text-secondary-foreground'
                  }`}>
                    {r.action}
                  </span>
                </td>
                <td className={`py-2 px-3 text-right mono font-medium ${r.reward >= 0 ? 'text-success' : 'text-destructive'}`}>
                  {r.reward >= 0 ? '+' : ''}{r.reward}
                </td>
                <td className="py-2 px-3 text-right mono text-foreground">${r.portfolioValue.toFixed(0)}</td>
                <td className="py-2 px-3 text-muted-foreground truncate max-w-[200px]">{r.explanation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
