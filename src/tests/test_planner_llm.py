"""Central OR-Tools planner and the Ollama LLM coordinator (with a fake client)."""
import json
import unittest

from farm.constants import CENTRAL_VRP, LLM, NOVEL
from farm.llm import OllamaCoordinator, build_prompt, validate
from farm.model import FarmModel


def run(model, steps):
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    return model


AREAS = [{"area": 3, "trees": 4, "urgency": 0.9}, {"area": 7, "trees": 9, "urgency": 0.1}]
PICKERS = [
    {"picker": 1, "battery": 90, "free_storage": 10, "queued": 0, "travel": {3: 5, 7: 20}},
    {"picker": 2, "battery": 40, "free_storage": 5, "queued": 1, "travel": {3: 12, 7: 4}},
]


class FakeClient:
    """Stands in for ollama.Client; returns a fixed answer or raises."""

    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        assert kwargs["format"]["required"] == ["assignments"]
        assert kwargs["options"]["temperature"] == 0
        return {"message": {"content": json.dumps(self.answer)}}


class TestPromptAndValidation(unittest.TestCase):
    def test_prompt_lists_areas_pickers_and_travel(self):
        prompt = build_prompt(AREAS, PICKERS)
        for text in ("area 3", "area 7", "picker 1", "picker 2", "free storage 5", "area 7: 4 steps"):
            self.assertIn(text, prompt)

    def test_validate_keeps_good_and_drops_bad_assignments(self):
        answer = {"assignments": [
            {"area": 3, "picker": 1},
            {"area": 3, "picker": 2},       # duplicate area
            {"area": 7, "picker": 2},       # 9 trees > 5 free storage
            {"area": 99, "picker": 1},      # unknown area
            {"area": "x", "picker": 1},     # not a number
        ]}
        result, invalid = validate(answer, AREAS, PICKERS)
        self.assertEqual(result, {3: 1})
        self.assertEqual(invalid, 4)

    def test_answers_are_cached(self):
        client = FakeClient({"assignments": [{"area": 3, "picker": 1}]})
        llm = OllamaCoordinator(client=client)
        self.assertEqual(llm.assign(AREAS, PICKERS), {3: 1})
        self.assertEqual(llm.assign(AREAS, PICKERS), {3: 1})
        self.assertEqual(client.calls, 1)

    def test_failure_returns_none(self):
        llm = OllamaCoordinator(client=FakeClient(error=ConnectionError("no server")))
        self.assertIsNone(llm.assign(AREAS, PICKERS))
        self.assertEqual(llm.failures, 1)


class TestLLMMode(unittest.TestCase):
    def test_llm_assignments_become_contracts(self):
        class NearestFake:
            """Assign every area to the picker with the shortest travel (a sane LLM)."""

            def chat(self, messages, **kwargs):
                prompt = messages[1]["content"]
                areas = [int(line.split()[2].rstrip(":")) for line in prompt.splitlines()
                         if line.startswith("- area")]
                first = int(next(line for line in prompt.splitlines() if line.startswith("- picker")).split()[2].rstrip(":"))
                return {"message": {"content": json.dumps(
                    {"assignments": [{"area": a, "picker": first} for a in areas]})}}

        model = run(FarmModel(mode=NOVEL, allocation=LLM, seed=5, llm_client=NearestFake()), 300)
        self.assertGreater(model.llm.calls, 0)
        self.assertGreater(model.contracts_awarded, 0)
        self.assertGreater(model.picked, 0)

    def test_unavailable_llm_falls_back_to_heuristic(self):
        model = run(FarmModel(mode=NOVEL, allocation=LLM, seed=5,
                              llm_client=FakeClient(error=TimeoutError())), 200)
        self.assertGreater(model.llm_fallbacks, 0)
        self.assertGreater(model.contracts_awarded, 0)
        self.assertGreater(model.picked, 0)


class TestCentralPlanner(unittest.TestCase):
    def test_planner_routes_pickers(self):
        model = run(FarmModel(mode=NOVEL, allocation=CENTRAL_VRP, seed=5), 300)
        self.assertGreater(model.planner.solves, 0)
        self.assertGreater(model.contracts_awarded, 0)
        self.assertGreater(model.picked, 0)
        self.assertTrue(all(p.is_active for p in model.pickers))

    def test_routed_tasks_are_visited_in_order(self):
        model = run(FarmModel(mode=NOVEL, allocation=CENTRAL_VRP, seed=5), 120)
        picker = next((p for p in model.pickers if any(t.get("order") is not None for t in p.tasks.values())), None)
        if picker is None:
            self.skipTest("no picker holds routed tasks at this moment")
        first = min(picker.tasks, key=lambda pos: picker.tasks[pos]["order"])
        self.assertEqual(picker.choose_task(), first)

    def test_routes_respect_capacity(self):
        model = FarmModel(mode=NOVEL, allocation=CENTRAL_VRP, seed=5, initial_ripe_patches=20)
        planner = model.planner
        for d in model.drones:          # pretend the drones reported everything
            for pid, trees in model.patches.items():
                if any(t.is_ripe for t in trees):
                    planner.known[pid] = d.task_content(pid, [t for t in trees if t.is_ripe])
        routes = planner.solve(planner.open_areas(), model.pickers)
        for picker, stops in routes:
            self.assertLessEqual(sum(len(r) for _, r, _ in stops), picker.spare_capacity)


if __name__ == "__main__":
    unittest.main()
