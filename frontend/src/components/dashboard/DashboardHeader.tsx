import { TrendingUp, User } from 'lucide-react';

interface DashboardHeaderProps {
  regime: string;
  step: number;
  done: boolean;
}

export function DashboardHeader({ regime, step, done }: DashboardHeaderProps) {
  return (
    <header className="flex items-center justify-between py-4 px-6 border-b border-border">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-bold gradient-text">FinLearn Tutor</h1>
            <p className="text-xs text-muted-foreground">AI trading command center powered by explainable simulation</p>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <span className="badge-regime">{regime}</span>
        <span className="mono text-xs text-muted-foreground">
          Step {step}/50 {done && <span className="text-destructive ml-1">• DONE</span>}
        </span>
          <div className="flex items-center gap-2 pl-4 border-l border-border">
            <div className="w-7 h-7 rounded-full bg-secondary flex items-center justify-center">
              <User className="w-4 h-4 text-secondary-foreground" />
            </div>
            <div className="text-xs">
              <p className="font-medium text-foreground">Explainable AI</p>
              <p className="text-muted-foreground">Learning-first investing</p>
            </div>
          </div>
        </div>
    </header>
  );
}
