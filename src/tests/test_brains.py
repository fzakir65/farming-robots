"""BDI (AgentSpeak) and learned (RL) picker brains."""
import unittest

import numpy as np

from farm.constants import BASIC, BDI, CAUTIOUS, DIRECT, EXTENDED, NOVEL, PICKER_BATTERY, RL
from farm.model import FarmModel
from farm.rl import K, N_ACTIONS, OBS_DIM, RLBrain, candidates, greedy_policy, make_aec_env, make_gym_env, observe
from tests.helpers import first_tree, lane_cell_next_to_tree, make_model, place, ripen


def run(model, steps):
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    return model


def give_task(picker, pos, urgency=0.5):
    picker.learn_task({"trees": [pos], "urgency": urgency}, task_id=None, manager=None)


class TestBDIPicker(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=EXTENDED, picker_brain=BDI)
        self.picker = self.model.pickers[0]

    def test_unloads_then_charges_at_base(self):
        self.picker.load = 3
        self.picker.battery = 50
        self.assertEqual(self.picker.make_decision(), "unload")
        self.picker.unload()
        self.assertEqual(self.picker.make_decision(), "charge")

    def test_waits_at_base_without_tasks(self):
        self.assertEqual(self.picker.make_decision(), "wait_at_base")

    def test_commits_to_the_best_task(self):
        tree = first_tree(self.model)
        ripen(self.model, tree)
        give_task(self.picker, tree)
        self.assertEqual(self.picker.make_decision(), "go_to_chosen_task")
        self.assertEqual(self.picker.chosen_task, tree)
        self.assertIn("committed", {lit.functor for lit in self._beliefs()})

    def test_drops_commitment_when_crop_is_gone(self):
        tree = first_tree(self.model)
        ripen(self.model, tree)
        give_task(self.picker, tree)
        self.picker.make_decision()
        self.picker.tasks.clear()                 # crop picked by someone else
        self.picker.make_decision()
        self.assertNotIn("committed", {lit.functor for lit in self._beliefs()})

    def test_picks_what_is_in_reach(self):
        tree = first_tree(self.model)
        ripen(self.model, tree)
        place(self.model, self.picker, lane_cell_next_to_tree(self.model, tree))
        self.assertEqual(self.picker.make_decision(), "pick")

    def test_returns_when_battery_is_low(self):
        place(self.model, self.picker, (10, 15))
        self.picker.battery = 5
        self.assertEqual(self.picker.make_decision(), "return_to_base")

    def _beliefs(self):
        return [b for group in self.picker.brain.agent.beliefs.values() for b in group]

    def test_bdi_runs_in_every_mode(self):
        for mode in (BASIC, EXTENDED, NOVEL):
            with self.subTest(mode=mode):
                model = run(FarmModel(mode=mode, comm_mode=DIRECT, picker_brain=BDI, seed=4), 300)
                self.assertGreater(model.picked, 0)
                self.assertTrue(all(p.is_active for p in model.pickers))

    def test_cautious_agents_reconsider(self):
        model = run(FarmModel(mode=EXTENDED, picker_brain=BDI, bdi_commitment=CAUTIOUS, seed=4), 400)
        self.assertGreater(sum(p.brain.reconsiderations for p in model.pickers), 0)


class TestLearnedPicker(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=EXTENDED, picker_brain=RL, rl_policy=greedy_policy)
        self.picker = self.model.pickers[0]

    def test_observation_and_mask(self):
        trees = sorted(self.model.trees_by_pos)[:3]
        for t in trees:
            ripen(self.model, t)
            give_task(self.picker, t)
        cands = candidates(self.picker)
        obs, mask = observe(self.picker, cands)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.all((obs >= 0) & (obs <= 1)))
        self.assertEqual(mask.tolist(), [True] * min(3, K) + [False] * (K - min(3, K)) + [True])

    def test_policy_choice_is_followed(self):
        tree = first_tree(self.model)
        ripen(self.model, tree)
        give_task(self.picker, tree)
        self.assertEqual(self.picker.make_decision(), "go_to_chosen_task")
        self.assertEqual(self.picker.chosen_task, tree)

    def test_policy_can_decline_and_stay_docked(self):
        self.picker.brain = RLBrain(lambda picker, obs, mask: N_ACTIONS - 1)
        tree = first_tree(self.model)
        ripen(self.model, tree)
        give_task(self.picker, tree)
        self.assertEqual(self.picker.make_decision(), "wait_at_base")

    def test_safety_rules_override_the_policy(self):
        place(self.model, self.picker, (10, 15))
        self.picker.battery = 5
        give_task(self.picker, first_tree(self.model))
        self.assertEqual(self.picker.make_decision(), "return_to_base")
        self.assertEqual(self.picker.brain.decisions, 0)

    def test_learned_brain_runs(self):
        model = run(FarmModel(mode=EXTENDED, picker_brain=RL, seed=4), 300)
        self.assertGreater(model.picked, 0)
        self.assertTrue(all(p.is_active for p in model.pickers))
        self.assertGreater(sum(p.brain.decisions for p in model.pickers), 0)


class TestRLEnvironments(unittest.TestCase):
    def test_pettingzoo_api(self):
        from pettingzoo.test import api_test

        env = make_aec_env(n_pickers=3, max_steps=120)
        api_test(env, num_cycles=100)
        env.close()

    def test_gymnasium_api(self):
        from stable_baselines3.common.env_checker import check_env

        env = make_gym_env(n_pickers=3, max_steps=120)
        check_env(env)
        obs, _ = env.reset(seed=1)
        total, done = 0.0, False
        while not done:
            mask = env.action_masks()
            obs, reward, terminated, truncated, _ = env.step(int(np.flatnonzero(mask)[0]))
            total += reward
            done = terminated or truncated
        env.close()
        self.assertIsInstance(total, float)

    def test_episode_reward_is_picked_minus_wasted(self):
        env = make_gym_env(n_pickers=3, max_steps=150)
        env.reset(seed=3)
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(0 if env.action_masks()[0] else K)
            total += reward
            done = terminated or truncated
        model = env.sim.model
        self.assertEqual(total, model.picked - model.wasted)
        env.close()

    def test_battery_feature_reflects_picker(self):
        picker = self.model_picker()
        picker.battery = PICKER_BATTERY / 2
        obs, _ = observe(picker, [])
        self.assertAlmostEqual(obs[K * 5], 0.5)

    def model_picker(self):
        return make_model(mode=EXTENDED).pickers[0]


if __name__ == "__main__":
    unittest.main()
