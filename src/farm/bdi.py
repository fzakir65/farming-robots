"""BDI picker: the robot's practical reasoning is an AgentSpeak program (asl/picker.asl).

Each step:
1. **Belief revision.** The robot's percepts become belief literals. Only the
   *changes* are posted to the interpreter, as ``+belief`` / ``-belief`` events, so
   plans can react to them (e.g. ``-task(X, Y)`` drops a commitment).
2. **Deliberation and means-ends reasoning.** The goal ``!tick`` is posted and the
   interpreter selects the first applicable plan. It runs until the plan performs one
   action with the internal action ``.act(...)``.
3. **Execution.** The action name goes back to the Mesa agent, which runs it like the
   rule-based robot does.

The commitment strategy (Kinny & Georgeff, 1991) is a belief. A *bold* agent keeps
its intention until it is achieved or impossible. A *cautious* agent reconsiders
every step and switches when a clearly better crop is known.
"""
import os

import agentspeak
import agentspeak.runtime
import agentspeak.stdlib
from agentspeak import Literal

from .constants import BASIC, CAUTIOUS, EXPLORE_PATIENCE, PICKER_BATTERY

ASL_PATH = os.path.join(os.path.dirname(__file__), "asl", "picker.asl")
SWITCH_MARGIN = 5.0         # a cautious robot switches if the new crop scores this much better
MAX_REASONING_STEPS = 200   # safety net: interpreter steps allowed per simulation step

actions = agentspeak.Actions(agentspeak.stdlib.actions)


def _brain(agent):
    return agent.farm_brain


@actions.add(".act", 1)
def _act(agent, term, intention):
    _brain(agent).action = (str(agentspeak.grounded(term.args[0], intention.scope)), ())
    yield


@actions.add(".act", 3)
def _act3(agent, term, intention):
    name, x, y = (agentspeak.grounded(a, intention.scope) for a in term.args)
    _brain(agent).action = (str(name), (int(x), int(y)))
    yield


@actions.add(".best_task", 2)
def _best_task(agent, term, intention):
    pos = _brain(agent).picker.choose_task()
    if pos is None:
        return
    if agentspeak.unify(term.args[0], pos[0], intention.scope, intention.stack) and \
            agentspeak.unify(term.args[1], pos[1], intention.scope, intention.stack):
        yield


@actions.add(".better_task", 2)
def _better_task(agent, term, intention):
    x, y = (int(agentspeak.grounded(a, intention.scope)) for a in term.args)
    picker = _brain(agent).picker
    best = picker.choose_task()
    if best is not None and best != (x, y) and (x, y) in picker.tasks:
        if picker.task_score(best) + SWITCH_MARGIN < picker.task_score((x, y)):
            yield


class BDIBrain:
    """Runs the AgentSpeak interpreter for one picker."""

    _env = agentspeak.runtime.Environment()

    def __init__(self, picker, commitment):
        self.picker = picker
        self.commitment = commitment
        with open(ASL_PATH, encoding="utf-8") as source:
            self.agent = self._env.build_agent(source, actions, name=f"picker{picker.unique_id}")
        self.agent.farm_brain = self
        self.current = set()
        self.action = None
        self.reasoning_steps = 0
        self.reconsiderations = 0

    # 1. belief revision ------------------------------------------------- #
    def percepts(self):
        p = self.picker
        beliefs = set()
        flags = {
            "at_base": p.at_base,
            "carrying": p.load > 0,
            "full": p.is_full,
            "needs_charge": p.battery < PICKER_BATTERY,
            "ripe_in_reach": bool(p.ripe_in_reach()),
            "ripe_in_sight": p.model.mode == BASIC and bool(p.ripe_in_sight()),
            "has_path": bool(p.path),
            "returning": p.state == "returning",
            "bored": p.fruitless_steps >= EXPLORE_PATIENCE,
            "cautious": self.commitment == CAUTIOUS,
        }
        if not p.at_base:
            flags["must_return"] = p.should_return()
        beliefs |= {Literal(name) for name, on in flags.items() if on}
        beliefs.add(Literal("mode", (Literal("basic" if p.model.mode == BASIC else "coordinated"),)))
        beliefs |= {Literal("task", (x, y)) for (x, y) in p.tasks}
        return beliefs

    def revise(self):
        new = self.percepts()
        removal, addition = agentspeak.Trigger.removal, agentspeak.Trigger.addition
        belief = agentspeak.GoalType.belief
        for lit in self.current - new:
            self.agent.call(removal, belief, lit, agentspeak.runtime.Intention())
        for lit in new - self.current:
            self.agent.call(addition, belief, lit, agentspeak.runtime.Intention())
        self.current = new

    # 2-3. deliberate and act ---------------------------------------------- #
    def decide(self, picker):
        self.revise()
        self.action = None
        self.agent.call(agentspeak.Trigger.addition, agentspeak.GoalType.achievement,
                        Literal("tick"), agentspeak.runtime.Intention(), delayed=True)
        for _ in range(MAX_REASONING_STEPS):
            self.reasoning_steps += 1
            if not self.agent.step() or self.action is not None:
                break
        if self.action is None:
            return picker.rule_based_decision()
        name, args = self.action
        if name == "go_to_task":
            if self.picker.target is not None and self.picker.target != args:
                self.reconsiderations += 1
            picker.chosen_task = args
            return "go_to_chosen_task"
        return name
