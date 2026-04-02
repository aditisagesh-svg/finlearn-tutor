import { useEffect, useMemo, useRef, useState } from 'react';
import { DashboardHeader } from '@/components/dashboard/DashboardHeader';
import { KpiCards } from '@/components/dashboard/KpiCards';
import { PortfolioChart } from '@/components/dashboard/PortfolioChart';
import { AllocationPie } from '@/components/dashboard/AllocationPie';
import { RewardChart } from '@/components/dashboard/RewardChart';
import { RiskPanel } from '@/components/dashboard/RiskPanel';
import { ActionLog } from '@/components/dashboard/ActionLog';
import { ControlPanel } from '@/components/dashboard/ControlPanel';
import { ACTIONS, fetchSimulation, type Action, type SimulationResult, type StepResult } from '@/lib/finlearn-engine';

const DEFAULT_MAX_STEPS = 20;
const DEFAULT_SEED = 42;

const Index = () => {
  const timerRef = useRef<number | null>(null);
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedAction, setSelectedAction] = useState<Action>('HOLD');
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSimulation = async () => {
    setLoading(true);
    setError(null);
    setRunning(false);

    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    try {
      const result = await fetchSimulation(DEFAULT_MAX_STEPS, DEFAULT_SEED);
      setSimulation(result);
      setCurrentIndex(result.steps.length > 0 ? 1 : 0);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to load simulation data.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadSimulation();
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!running || !simulation) {
      return;
    }

    if (currentIndex >= simulation.steps.length) {
      setRunning(false);
      return;
    }

    timerRef.current = window.setTimeout(() => {
      setCurrentIndex((value) => value + 1);
    }, 350);

    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, [running, currentIndex, simulation]);

  const visibleHistory = useMemo(() => {
    if (!simulation) {
      return [] as StepResult[];
    }
    return simulation.steps.slice(0, currentIndex);
  }, [simulation, currentIndex]);

  const state = visibleHistory[visibleHistory.length - 1] ?? null;
  const initialValue = simulation?.initialValue ?? 1000;

  const handleStep = () => {
    if (!simulation) {
      return;
    }
    setRunning(false);
    setCurrentIndex((value) => Math.min(value + 1, simulation.steps.length));
  };

  const handleReset = () => {
    setRunning(false);
    setCurrentIndex(simulation && simulation.steps.length > 0 ? 1 : 0);
  };

  const handleRunEpisode = () => {
    if (!simulation) {
      return;
    }
    setRunning(true);
  };

  const done = simulation ? currentIndex >= simulation.steps.length : false;

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader regime={state?.regime ?? 'Live Simulation'} step={state?.step ?? 0} done={done} />
      <main className="max-w-[1440px] mx-auto p-4 md:p-6 space-y-4">
        {error ? (
          <div className="fintech-card border border-destructive/40 text-sm text-destructive">
            Failed to load simulation data from the backend. Start the Python API and try again.
            <div className="mt-2 text-muted-foreground">{error}</div>
          </div>
        ) : null}

        <KpiCards
          portfolioValue={state?.portfolioValue ?? initialValue}
          cash={state?.cash ?? initialValue}
          initialValue={initialValue}
          learningScore={state?.learningScore ?? 0}
        />

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-4">
          <PortfolioChart history={visibleHistory} />
          <AllocationPie
            holdings={state?.holdings ?? { ALPHA: 0, BETA: 0, GAMMA: 0 }}
            prices={state?.prices ?? { ALPHA: 0, BETA: 0, GAMMA: 0 }}
            cash={state?.cash ?? initialValue}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <RewardChart history={visibleHistory} />
          <RiskPanel
            volatility={state?.volatility ?? 0}
            drawdown={state?.drawdown ?? 0}
            concentration={state?.concentration ?? 0}
          />
        </div>

        <ActionLog history={visibleHistory} />

        <ControlPanel
          actions={ACTIONS}
          selectedAction={selectedAction}
          onActionChange={setSelectedAction}
          onStep={handleStep}
          onReset={handleReset}
          onRunEpisode={handleRunEpisode}
          onReload={loadSimulation}
          done={done}
          running={running || loading}
        />
      </main>
    </div>
  );
};

export default Index;
