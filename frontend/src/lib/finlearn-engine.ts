export type Action =
  | 'BUY_ALPHA'
  | 'BUY_BETA'
  | 'BUY_GAMMA'
  | 'SELL_ALPHA'
  | 'SELL_BETA'
  | 'SELL_GAMMA'
  | 'HOLD'
  | 'REBALANCE'
  | 'REQUEST_HINT';

export interface AssetPrices {
  ALPHA: number;
  BETA: number;
  GAMMA: number;
}

export interface Portfolio {
  ALPHA: number;
  BETA: number;
  GAMMA: number;
}

export interface StepResult {
  step: number;
  action: Action;
  reward: number;
  explanation: string;
  concept: string;
  suggestion: string;
  portfolioValue: number;
  cash: number;
  prices: AssetPrices;
  holdings: Portfolio;
  done: boolean;
  regime: string;
  volatility: number;
  drawdown: number;
  concentration: number;
  learningScore: number;
}

export interface SimulationResult {
  initialValue: number;
  steps: StepResult[];
  overallScore: number;
}

declare global {
  interface Window {
    __FINLEARN_BOOTSTRAP__?: ApiResponse;
  }
}

interface ApiStep {
  step: number;
  action: string;
  reward: number;
  portfolio_value: number;
  cash_balance: number;
  holdings: Portfolio;
  prices: AssetPrices;
  volatility: Record<keyof Portfolio, number>;
  learning_score: number;
  reasoning: string;
  concept: string;
  suggestion: string;
  market_regime: string;
  max_drawdown: number;
  concentration_score: number;
  portfolio_volatility: number;
}

interface ApiResponse {
  initial_value: number;
  steps: ApiStep[];
  task_scores: {
    overall_score: number;
  };
  final_state: {
    market_regime?: string;
    max_drawdown?: number;
    concentration_score?: number;
  };
}

const DEFAULT_API_BASE = '';

function averageVolatility(volatility: Record<keyof Portfolio, number>): number {
  const values = Object.values(volatility);
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clean(text: string): string {
  return text
    .replace(/^📘\s*Concept:\s*/u, '')
    .replace(/^📘\s*/u, '')
    .replace(/^📌\s*Reasoning:\s*/u, '')
    .replace(/^📌\s*/u, '')
    .replace(/^➡\s*Suggestion:\s*/u, '')
    .replace(/^➡\s*/u, '')
    .trim();
}

export async function fetchSimulation(maxSteps = 20, seed = 42): Promise<SimulationResult> {
  if (window.__FINLEARN_BOOTSTRAP__ && maxSteps === 20 && seed === 42) {
    const bootstrap = window.__FINLEARN_BOOTSTRAP__;
    delete window.__FINLEARN_BOOTSTRAP__;
    return normalizeSimulation(bootstrap);
  }

  const apiBase = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE;
  const requestPath = `${apiBase}/api/simulation?max_steps=${maxSteps}&seed=${seed}`;
  let response: Response;

  try {
    response = await fetch(requestPath);
  } catch (error) {
    const fallbackPath = `http://127.0.0.1:7860/api/simulation?max_steps=${maxSteps}&seed=${seed}`;
    response = await fetch(fallbackPath).catch(() => {
      throw error;
    });
  }

  if (!response.ok) {
    throw new Error(`Simulation request failed with status ${response.status}`);
  }

  const data = (await response.json()) as ApiResponse;
  return normalizeSimulation(data);
}

function normalizeSimulation(data: ApiResponse): SimulationResult {
  let runningMax = data.initial_value;

  const normalizedSteps = data.steps.map((step, index) => {
    runningMax = Math.max(runningMax, step.portfolio_value);
    const drawdown = runningMax > 0 ? ((runningMax - step.portfolio_value) / runningMax) * 100 : 0;
    const currentHoldingsValue =
      step.holdings.ALPHA * step.prices.ALPHA +
      step.holdings.BETA * step.prices.BETA +
      step.holdings.GAMMA * step.prices.GAMMA;
    const largestPosition = Math.max(
      step.holdings.ALPHA * step.prices.ALPHA,
      step.holdings.BETA * step.prices.BETA,
      step.holdings.GAMMA * step.prices.GAMMA,
    );
    const concentration = currentHoldingsValue > 0 ? largestPosition / currentHoldingsValue : 0;

    return {
      step: step.step,
      action: step.action as Action,
      reward: step.reward,
      explanation: clean(step.reasoning),
      concept: clean(step.concept),
      suggestion: clean(step.suggestion),
      portfolioValue: step.portfolio_value,
      cash: step.cash_balance,
      prices: step.prices,
      holdings: step.holdings,
      done: index === data.steps.length - 1,
      regime: step.market_regime || data.final_state.market_regime || 'Live Simulation',
      volatility:
        step.portfolio_volatility !== undefined
          ? step.portfolio_volatility * 100
          : averageVolatility(step.volatility) * 100,
      drawdown: step.max_drawdown !== undefined ? step.max_drawdown * 100 : drawdown,
      concentration: step.concentration_score ?? concentration,
      learningScore: step.learning_score * 100,
    } satisfies StepResult;
  });

  return {
    initialValue: data.initial_value,
    steps: normalizedSteps,
    overallScore: data.task_scores.overall_score * 100,
  };
}

export const ACTIONS: Action[] = [
  'HOLD',
  'BUY_ALPHA',
  'BUY_BETA',
  'BUY_GAMMA',
  'SELL_ALPHA',
  'SELL_BETA',
  'SELL_GAMMA',
  'REBALANCE',
];
