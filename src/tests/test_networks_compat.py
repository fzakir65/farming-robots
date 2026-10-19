"""Message transport (lossy radio, real XMPP) and Mesa 2/3 compatibility."""
import importlib.util
import os
import unittest

from farm import messages as msg
from farm.compat import MESA3, RandomScheduler
from farm.constants import EXTENDED, LOSSY, NOVEL, XMPP
from farm.model import FarmModel
from tests.helpers import make_model


def run(model, steps):
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    return model


class TestLossyRadio(unittest.TestCase):
    def test_everything_lost_when_drop_prob_is_one(self):
        model = make_model(network=LOSSY, drop_prob=1.0, msg_delay=0)
        a, b = model.pickers[0].unique_id, model.pickers[1].unique_id
        model.bus.send(msg.INFORM_CROP, a, b, {})
        self.assertEqual(model.bus.receive(b), [])
        self.assertEqual(model.bus.dropped, 1)
        self.assertEqual(model.bus.sent, 1)              # it was still transmitted

    def test_delay_holds_messages_back(self):
        model = make_model(network=LOSSY, drop_prob=0.0, msg_delay=2)
        a, b = model.pickers[0].unique_id, model.pickers[1].unique_id
        model.bus.send(msg.INFORM_CROP, a, b, {"x": 1})
        self.assertEqual(model.bus.receive(b), [])
        model.schedule.steps += 2
        self.assertEqual(len(model.bus.receive(b)), 1)

    def test_coordination_survives_message_loss(self):
        model = run(FarmModel(mode=NOVEL, network=LOSSY, drop_prob=0.3, seed=5), 400)
        self.assertGreater(model.bus.dropped, 0)
        self.assertGreater(model.picked, 0)
        self.assertTrue(all(p.is_active for p in model.pickers))


@unittest.skipUnless(importlib.util.find_spec("spade"), "SPADE is not installed")
@unittest.skipIf(os.environ.get("FARM_SKIP_XMPP"), "FARM_SKIP_XMPP is set")
class TestXmpp(unittest.TestCase):
    def test_messages_travel_over_xmpp(self):
        model = FarmModel(mode=NOVEL, network=XMPP, seed=3, max_steps=60)
        try:
            run(model, 60)
            self.assertGreater(model.bus.copies_sent, 0)
            self.assertEqual(model.bus.copies_received, model.bus.copies_sent)
            self.assertGreater(model.bus.mean_latency_ms, 0)
        finally:
            model.close()

    def test_same_outcome_as_ideal_network_when_nothing_is_lost(self):
        ideal = run(FarmModel(mode=EXTENDED, seed=8, max_steps=40), 40)
        model = FarmModel(mode=EXTENDED, network=XMPP, seed=8, max_steps=40)
        try:
            run(model, 40)
            self.assertEqual(model.bus.sent, ideal.bus.sent)
        finally:
            model.close()


class TestCompatibility(unittest.TestCase):
    def test_scheduler_steps_every_robot_once(self):
        model = make_model()
        calls = []
        for r in model.schedule.agents:
            r.step = (lambda rid: (lambda: calls.append(rid)))(r.unique_id)
        model.schedule.step()
        self.assertEqual(sorted(calls), sorted(r.unique_id for r in model.schedule.agents))
        self.assertEqual(model.schedule.steps, 1)

    def test_scheduler_is_our_own(self):
        self.assertIsInstance(make_model().schedule, RandomScheduler)

    def test_ids_are_ours_on_every_mesa_version(self):
        model = make_model()
        ids = [a.unique_id for a in model.schedule.agents]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_data_collection_works(self):
        model = run(make_model(mode=EXTENDED), 5)
        if MESA3:
            frame = model.datacollector.get_agenttype_vars_dataframe(type(model.pickers[0]))
        else:
            frame = model.datacollector.get_agent_vars_dataframe()
        self.assertGreater(len(frame), 0)


if __name__ == "__main__":
    unittest.main()
