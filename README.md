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

FinLearn Tutor is an OpenEnv-style financial learning environment for evaluating agents on portfolio decision-making. The environment simulates a small investment workflow where an agent must manage cash, select buy and sell actions, react to changing market trends, and balance growth against concentration risk.

The task is intended as a real-world financial learning simulation rather than a game. The agent is evaluated on capital preservation, diversification, and risk-adjusted returns across a fixed-horizon episode.

## Environment Overview

- Domain: financial learning and portfolio management
- Episode length: 30 steps by default
- Initial cash: `$1000.00`
- Assets: `ALPHA`, `BETA`, `GAMMA`
- Trade size: fixed `$100` buy or sell increments

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
| 7 | `REBALANCE` | Sell half of the most concentrated holding |
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

## Tasks

The project includes three deterministic tasks with graders that return scores in `[0.0, 1.0]`.

| Task ID | Difficulty | Objective | Grader |
|---|---|---|---|
| `task1` | Easy | Avoid losses relative to the starting portfolio value | `grade_task1` |
| `task2` | Medium | Maintain diversification across holdings | `grade_task2` |
| `task3` | Hard | Maximize returns while limiting concentration risk | `grade_task3` |

## Reward Function

Reward is shaped across the full trajectory and includes:

- Portfolio growth reward
- Diversification bonus
- Concentration penalty
- Overtrading penalty

This provides intermediate learning signal instead of a purely terminal binary outcome.

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── inference.py
├── openenv.yaml
├── requirements.txt
├── README.md
└── env/
    ├── __init__.py
    ├── environment.py
    ├── feedback.py
    ├── market.py
    ├── models.py
    ├── rewards.py
    └── tasks.py
```

## Setup

### Local

```bash
pip install -r requirements.txt
python inference.py
```

### Docker

```bash
docker build -t finlearn-tutor .
docker run --rm finlearn-tutor
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
[END] success=true steps=20 score=0.609 rewards=-0.15,-0.14,...
```

## Baseline Scores

With the default deterministic baseline (`SEED=42`, `MAX_STEPS=20`), the environment produces reproducible scores:

| Metric | Score |
|---|---|
| `task1_avoid_losses` | `1.0000` |
| `task2_diversification` | `0.4841` |
| `task3_returns_with_low_risk` | `0.3427` |
| `overall_score` | `0.6089` |

## Validation Checklist

Before submission, run:

```bash
python inference.py
docker build -t finlearn-tutor .
openenv validate
```

If deploying to Hugging Face Spaces, ensure the Space is live and responds to the validator checks required by the hackathon.
