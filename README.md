---
title: FinLearn Tutor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# FinLearn Tutor

FinLearn Tutor is an OpenEnv-compatible financial learning environment for evaluating agents on portfolio decision-making. It simulates a retail-investing workflow where an agent must manage cash, trade among synthetic assets, react to different market regimes, and balance returns against diversification and risk.

The environment is designed as a real-world financial learning simulation rather than a game. It supports profile-aware evaluation through deterministic investor profiles and exposes explainable feedback at each step.

## Why This Benchmark Matters

Most finance-themed agent demos reward one thing: the final portfolio value. Real decision systems are judged very differently. In practice, a strong financial agent must protect capital through drawdowns, adapt when market structure changes, avoid unnecessary churn, and justify its actions in a way that humans can inspect.

FinLearn Tutor is built to benchmark exactly that. It is not a toy "pick the winning stock" simulator. It is a deterministic training and evaluation environment for sequential financial decision-making, where intelligence is measured across the full trajectory of behavior.

## What Makes FinLearn Tutor Different

- Trajectory-aware scoring instead of terminal-only grading
- Deterministic market regimes and macro events for reproducible benchmarking
- Structured external signals that test reasoning without relying on brittle free-form NLP
- Investor-profile-aware incentives for conservative, balanced, and aggressive strategies
- Explainable observations with reasoning hints, risk level, market event context, and last-action feedback
- Lightweight runtime and OpenEnv-compatible API surface

## How Agent Intelligence Is Evaluated

FinLearn Tutor evaluates agents on the path they take, not just the ending they reach.

Each task score is a deterministic weighted combination of:

- Portfolio growth
- Maximum drawdown
- Volatility of returns
- Trade efficiency and overtrading control
- Regime adaptation quality

The benchmark exposes three refined tasks on the same environment:

- Capital Preservation: rewards downside control and consistent risk management
- Balanced Growth: rewards stable upside with disciplined diversification
- Aggressive Optimization: rewards higher upside while still penalizing reckless behavior

This makes the benchmark useful for evaluating whether an agent is actually behaving intelligently over time, rather than getting lucky on a single final state.

## Real-World Relevance

This benchmark maps well to real financial AI workflows:

- Training portfolio agents that must react to regime shifts
- Evaluating trading copilots before human review
- Comparing baseline policies against LLM-augmented decision agents
- Stress-testing risk-aware planning under crashes, inflation shocks, and rate hikes

For hackathon judges, the key point is simple: FinLearn Tutor demonstrates a credible evaluation framework for sequential financial reasoning, not just a market-themed interface.

## Environment Overview

- Domain: financial learning and portfolio management
- Episode length: 30 steps by default
- Initial cash: `$1000.00`
- Assets: `ALPHA`, `BETA`, `GAMMA`
- Trade size: fixed `$100` buy or sell increments
- Market regimes: `bull`, `bear`, `sideways`, `high_volatility`
- Market events: `interest_rate_hike`, `market_crash`, `tech_bubble`, `inflation_spike`
- External signals: structured deterministic market cues aligned with regime and event context
- User profiles: `conservative`, `balanced`, `aggressive`

The environment implements:

- Typed `Observation`, `Action`, and `Reward` models with Pydantic
- `reset()` to start a new episode
- `step(action)` to advance one environment step
- `state()` to return the current typed observation
- `openenv.yaml` metadata for validation

## Action Space

| ID | Action | Description |
|---|---|---|
| 0 | `HOLD` | Take no trade action |
| 1 | `BUY_ALPHA` | Buy `$100` of `ALPHA` |
| 2 | `BUY_BETA` | Buy `$100` of `BETA` |
| 3 | `BUY_GAMMA` | Buy `$100` of `GAMMA` |
| 4 | `SELL_ALPHA` | Sell up to `$100` of `ALPHA` |
| 5 | `SELL_BETA` | Sell up to `$100` of `BETA` |
| 6 | `SELL_GAMMA` | Sell up to `$100` of `GAMMA` |
| 7 | `REBALANCE` | Risk-aware rebalance using inverse-volatility targets |
| 8 | `REQUEST_HINT` | Request a tutor hint without trading |

## Observation Space

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `cash_balance` | `float` | Current available cash |
| `holdings` | `dict[str, int]` | Shares held for each stock |
| `prices` | `dict[str, float]` | Current stock prices |
| `trends` | `dict[str, float]` | Current directional market signal per stock |
| `volatility` | `dict[str, float]` | Current volatility estimate per stock |
| `portfolio_value` | `float` | Cash plus marked-to-market stock value |
| `step` | `int` | Current step number |
| `learning_score` | `float` | Running normalized learning score in `[0.0, 1.0]` |
| `risk_appetite` | `str` | Investor risk profile |
| `investment_horizon` | `str` | Investor horizon profile |
| `goal` | `str` | Investor objective |
| `investor_profile` | `str` | Deterministic user archetype used for reward shaping |
| `market_regime` | `str` | Active market regime |
| `market_event` | `str` | Active macro event affecting prices and risk |
| `external_signal` | `dict[str, str]` | Structured signal about sector or macro impact |
| `portfolio_volatility` | `float` | Weighted volatility estimate for held assets |
| `concentration_score` | `float` | Maximum single-asset concentration |
| `max_drawdown` | `float` | Running maximum drawdown in the episode |
| `reasoning_hint` | `str` | Deterministic explainability cue for the next decision |
| `risk_level` | `str` | Interpretable market risk state (`low`, `moderate`, `high`) |
| `last_action_feedback` | `str` | Deterministic explanation of the previous action's impact |

## Tasks

The project includes three deterministic trajectory-aware tasks with graders that return scores in `[0.0, 1.0]`.

| Task ID | Difficulty | Objective | Grader |
|---|---|---|---|
| `task1` | Easy | Capital Preservation | `grade_task1` |
| `task2` | Medium | Balanced Growth | `grade_task2` |
| `task3` | Hard | Aggressive Optimization | `grade_task3` |

Each task uses the same benchmark core but different weights across growth, risk control, stability, and decision quality.

## Reward Function

Reward is shaped across the full trajectory and includes:

- Portfolio growth reward
- Diversification bonus
- Concentration penalty
- Overtrading penalty
- Transaction cost penalty
- Profile-alignment penalty for mismatched risk behavior
- Higher penalties when aggressive actions are taken into high-risk regimes for conservative profiles

This provides intermediate learning signal instead of a purely terminal binary outcome.

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── README.md
├── frontend/
│   ├── public/
│   ├── src/
│   └── README.md
├── env/
│   ├── __init__.py
│   ├── environment.py
│   ├── feedback.py
│   ├── market.py
│   ├── metrics.py
│   ├── models.py
│   ├── rewards.py
│   └── tasks.py
└── server/
    ├── __init__.py
    └── app.py
```

## Frontend Placement

If you export a React or Lovable dashboard, place it inside `frontend/` instead of the repository root.

- Put app code such as `src/`, `components/`, `hooks/`, `lib/`, `pages/`, `App.tsx`, `main.tsx`, and `index.css` inside `frontend/`
- Put web assets such as `favicon.ico`, `robots.txt`, and `placeholder.svg` inside `frontend/public/` when the Vite app expects public assets
- Keep the existing Python files in the repository root so the simulation and server remain separate from the frontend build toolchain

This keeps the repo organized as:

- Python simulation and API at the root
- React frontend in `frontend/`

## Setup

### Local

```bash
pip install -r requirements.txt
python inference.py
```

`requirements.txt` intentionally installs the local project package, so dependency resolution comes from `pyproject.toml` and stays aligned with Docker builds and OpenEnv validation.

### Final Dashboard

The project now includes a separate React frontend in `frontend/` that uses the real simulation output from the Python backend.

Run the backend:

```bash
python -m server.app
```

Run the frontend in a second terminal:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

The frontend calls:

- `GET /api/simulation` for the final dashboard data
- `POST /reset` for the validator-compatible environment endpoint

### Docker

```bash
docker build -t finlearn-tutor .
docker run --rm -p 7860:7860 finlearn-tutor
```

## Inference Configuration

The baseline inference script is `inference.py` in the repository root.

It is a deterministic, lightweight benchmark baseline designed for reproducible validator runs. The script initializes an OpenAI-compatible client for hackathon compliance, while the actual action policy remains deterministic and low-compute.

Optional environment variables:

- `API_BASE_URL`
- `HF_TOKEN`
- `MODEL_NAME`
- `TASK_NAME`
- `BENCHMARK`
- `MAX_STEPS`
- `SEED`
- `SUCCESS_SCORE_THRESHOLD`

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_token_here"
export MODEL_NAME="deterministic-baseline-v2"
export MAX_STEPS="20"
export SEED="42"
python inference.py
```

The baseline passes `env.get_episode_summary()` into the graders, so task scores reflect trajectory-aware behavior rather than only the terminal state.

## Inference Output Format

The baseline emits strict validator-friendly logs:

```text
[START] task=task3_aggressive_optimization env=finlearn_tutor model=deterministic-baseline-v2
[STEP] step=1 action=1 reward=-0.15 done=false error=null
[STEP] step=2 action=1 reward=-0.14 done=false error=null
[END] success=true steps=20 score=0.584 rewards=-0.15,-0.14,...
```

## Benchmark Positioning

`inference.py` is intentionally a deterministic baseline agent. It is not presented as a production trading model. Its role is to provide a stable lower-bound benchmark for:

- regression testing
- leaderboard comparisons
- measuring whether more advanced agents genuinely improve trajectory quality

That separation makes the project stronger: the environment is the benchmark, and the baseline is the reproducible control policy.

## Baseline Scores

With the default deterministic baseline (`SEED=42`, `MAX_STEPS=20`), the environment produces reproducible trajectory-aware scores.

The exact values remain deterministic per seed and may evolve as the benchmark improves, but the reported outputs are always bounded in `[0.0, 1.0]` and derived from:

- growth over the episode
- maximum drawdown
- realized volatility
- trade efficiency
- regime adaptation quality

## API Check

The deployed Hugging Face Space exposes a validator-compatible reset endpoint:

```bash
curl -X POST https://aditisageshhh-finlearn-tutor.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Validation Checklist

Before submission, run:

```bash
python inference.py
docker build -t finlearn-tutor .
openenv validate
```

The current project is structured to satisfy:

- Hugging Face Space `/reset` availability
- Docker build validation
- OpenEnv validation
- baseline inference logging requirements
