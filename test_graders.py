import unittest

from env.tasks import grade_task1, grade_task2, grade_task3


def make_state(portfolio_value: float = 1200.0):
    return {"portfolio_value": portfolio_value}


def make_positive_trajectory():
    return {
        "portfolio_history": [1000.0, 1050.0, 1100.0, 1150.0, 1200.0],
        "action_history": [1, 1, 1, 1],
        "step_records": [
            {"action_id": 1, "regime": "bull", "risk_level": "low", "best_trend": 0.01},
            {"action_id": 1, "regime": "bull", "risk_level": "low", "best_trend": 0.01},
            {"action_id": 1, "regime": "bull", "risk_level": "low", "best_trend": 0.01},
            {"action_id": 1, "regime": "bull", "risk_level": "low", "best_trend": 0.01},
        ],
    }


class GraderTests(unittest.TestCase):
    def test_all_zero_returns_valid_score(self) -> None:
        for grader in (grade_task1, grade_task2, grade_task3):
            score = grader(make_state(portfolio_value=0.0), trajectory={"portfolio_history": [1000.0, 0.0]})
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_perfect_metrics_returns_near_one(self) -> None:
        for grader in (grade_task1, grade_task2, grade_task3):
            score = grader(make_state(), trajectory=make_positive_trajectory())
            self.assertGreaterEqual(score, 0.8)
            self.assertLessEqual(score, 1.0)

    def test_bankrupt_portfolio(self) -> None:
        for grader in (grade_task1, grade_task2, grade_task3):
            score = grader(make_state(portfolio_value=-1.0), trajectory={"portfolio_history": [1000.0, -1.0]})
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
