"""Small compatibility layer so the same model runs on Mesa 2.x and Mesa 3.x.

The module's worksheets use Mesa 2.1 (``mesa.time`` schedulers, ``ModularServer``,
``Agent(unique_id, model)``). Mesa 3 removed or changed those APIs:

* agents no longer take a ``unique_id`` (Mesa numbers them itself);
* ``Model.next_id()`` and the ``mesa.time`` schedulers are deprecated or removed;
* the browser UI moved from ``ModularServer`` to Solara (``SolaraViz``).

The farm keeps its own ids and its own random-activation scheduler, so the
behaviour is identical on both versions. Only the user interface differs:
``run.py`` (Mesa 2) and ``app_solara.py`` (Mesa 3).
"""
import mesa

MESA_VERSION = tuple(int(p) for p in mesa.__version__.split(".")[:2])
MESA3 = MESA_VERSION >= (3, 0)


class FarmAgent(mesa.Agent):
    """Agent base class that accepts ``(unique_id, model)`` on every Mesa version."""

    def __init__(self, unique_id, model):
        if MESA3:
            super().__init__(model)
            self.unique_id = unique_id      # keep our ids so runs match across versions
        else:
            super().__init__(unique_id, model)


class RandomScheduler:
    """
    Random activation, as ``mesa.time.RandomActivation`` in Mesa 2: every step,
    each scheduled agent's ``step()`` is called once, in a new random order.

    Only robots are scheduled. Static agents (trees, river, bases) are driven by the
    environment instead.
    """

    def __init__(self, model):
        self.model = model
        self.steps = 0
        self._agents = []

    @property
    def agents(self):
        return list(self._agents)

    def add(self, agent):
        self._agents.append(agent)

    def remove(self, agent):
        self._agents.remove(agent)

    def get_agent_count(self):
        return len(self._agents)

    def step(self):
        order = list(self._agents)
        self.model.random.shuffle(order)
        for agent in order:
            agent.step()
        self.steps += 1
