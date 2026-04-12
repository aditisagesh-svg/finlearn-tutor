import unittest

from fastapi.testclient import TestClient

from server.app import app


class IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_tasks(self) -> None:
        response = self.client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        tasks = response.json()["tasks"]
        self.assertGreaterEqual(len(tasks), 3)
        task_ids = {task["task_id"] for task in tasks}
        self.assertTrue({"task1", "task2", "task3"}.issubset(task_ids))

    def test_reset_run_flow(self) -> None:
        response = self.client.post("/reset", json={"task_id": "task1"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_id"], "task1")
        self.assertIn("observation", payload)
        self.assertIn("info", payload)
        self.assertIn("config", payload)

        response = self.client.post("/run", json={"task_id": "task1", "action": 0})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("observation", payload)
        self.assertIn("reward", payload)
        self.assertIn("done", payload)
        self.assertIsInstance(payload["reward"], float)
        self.assertIsInstance(payload["done"], bool)

    def test_score_in_range(self) -> None:
        self.client.post("/reset", json={"task_id": "task1", "max_steps": 1})
        self.client.post("/run", json={"task_id": "task1", "action": 0})
        response = self.client.post("/grade", json={"task_id": "task1"})
        self.assertEqual(response.status_code, 200)
        score = response.json()["score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_all_actions_no_crash(self) -> None:
        for action in range(9):
            self.client.post("/reset", json={"task_id": "task1"})
            response = self.client.post("/run", json={"task_id": "task1", "action": action})
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
