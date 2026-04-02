import { Play, RotateCcw, FastForward, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import type { Action } from '@/lib/finlearn-engine';

interface ControlPanelProps {
  actions: Action[];
  selectedAction: Action;
  onActionChange: (action: Action) => void;
  onStep: () => void;
  onReset: () => void;
  onRunEpisode: () => void;
  onReload: () => void;
  done: boolean;
  running: boolean;
}

export function ControlPanel({
  actions,
  selectedAction,
  onActionChange,
  onStep,
  onReset,
  onRunEpisode,
  onReload,
  done,
  running,
}: ControlPanelProps) {
  return (
    <div className="fintech-card">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground font-medium">Action:</span>
          <Select value={selectedAction} onValueChange={(v) => onActionChange(v as Action)}>
            <SelectTrigger className="w-[160px] h-9 text-xs bg-secondary border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {actions.map(a => (
                <SelectItem key={a} value={a} className="text-xs">{a}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button size="sm" onClick={onStep} disabled={done || running} className="gap-1.5 bg-primary text-primary-foreground hover:bg-primary/90">
          <Play className="w-3.5 h-3.5" /> Step
        </Button>
        <Button size="sm" variant="outline" onClick={onRunEpisode} disabled={done || running} className="gap-1.5 border-border text-foreground hover:bg-secondary">
          <FastForward className="w-3.5 h-3.5" /> {running ? 'Running...' : 'Run Episode'}
        </Button>
        <Button size="sm" variant="outline" onClick={onReset} className="gap-1.5 border-border text-foreground hover:bg-secondary">
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </Button>
        <Button size="sm" variant="outline" onClick={onReload} disabled={running} className="gap-1.5 border-border text-foreground hover:bg-secondary">
          <RefreshCw className="w-3.5 h-3.5" /> Reload Data
        </Button>
      </div>
    </div>
  );
}
