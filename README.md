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

## Environment Overview

- Domain: financial learning and portfolio management
- Episode length: 30 steps by default
- Initial cash: `$1000.00`
- Assets: `ALPHA`, `BETA`, `GAMMA`
- Trade size: fixed `$100` buy or sell increments
- Market regimes: `bull`, `bear`, `sideways`, `crash`
- User profiles: deterministic combinations of risk appetite, investment horizon, and goal

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
| `market_regime` | `str` | Active market regime |
| `portfolio_volatility` | `float` | Weighted volatility estimate for held assets |
| `concentration_score` | `float` | Maximum single-asset concentration |
| `max_drawdown` | `float` | Running maximum drawdown in the episode |

## Tasks

The project includes three deterministic tasks with graders that return scores in `[0.0, 1.0]`.

| Task ID | Difficulty | Objective | Grader |
|---|---|---|---|
| `task1` | Easy | Preserve capital during uncertain market conditions | `grade_task1` |
| `task2` | Medium | Maintain a diversified portfolio under changing market regimes | `grade_task2` |
| `task3` | Hard | Maximize risk-adjusted returns while avoiding overexposure | `grade_task3` |

## Reward Function

Reward is shaped across the full trajectory and includes:

- Portfolio growth reward
- Diversification bonus
- Concentration penalty
- Overtrading penalty
- Transaction cost penalty
- Profile-alignment penalty for mismatched risk behavior

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

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional environment variables:

- `TASK_NAME`
- `BENCHMARK`
- `MAX_STEPS`
- `SEED`
- `SUCCESS_SCORE_THRESHOLD`

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

## Inference Output Format

The baseline emits strict validator-friendly logs:

```text
[START] task=task3_returns_with_low_risk env=finlearn_tutor model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=1 reward=-0.15 done=false error=null
[STEP] step=2 action=1 reward=-0.14 done=false error=null
[END] success=true steps=20 score=0.584 rewards=-0.15,-0.14,...
```

## Baseline Scores

With the default deterministic baseline (`SEED=42`, `MAX_STEPS=20`), the environment produces reproducible scores:

| Metric | Score |
|---|---|
| `task1_avoid_losses` | `1.0000` |
| `task2_diversification` | `0.4977` |
| `task3_returns_with_low_risk` | `0.2552` |
| `overall_score` | `0.5843` |

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
