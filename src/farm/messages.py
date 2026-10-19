"""Agent communication: FIPA-style messages and a simple mailbox-based message bus.

Messages are delivered asynchronously: a message sent in one step is read the next
time the receiver checks its mailbox (normally at the start of its own step).
"""
from dataclasses import dataclass, field

# Performatives (the "intent" of a message), loosely following FIPA-ACL
INFORM_CROP = "inform-crop"   # extended: drone -> picker(s), "ripe crops are here"
CFP = "cfp"                   # novel: drone -> all pickers, call for proposals
PROPOSE = "propose"           # novel: picker -> drone, bid with an estimated cost
REFUSE = "refuse"             # novel: picker -> drone, cannot take the task
ACCEPT = "accept-proposal"    # novel: drone -> winning picker
REJECT = "reject-proposal"    # novel: drone -> losing pickers
INFORM_DONE = "inform-done"   # novel: picker -> drone, task finished
CLOCK = "cfp-price"           # novel, English auction: drone -> pickers, current clock price
INFORM_SECTOR = "inform-sector"  # drones' sector game: drone -> drones, "I will search here"


@dataclass
class Message:
    performative: str
    sender: int
    receiver: int | None          # None means broadcast
    content: dict = field(default_factory=dict)
    time: int = 0


class MessageBus:
    """Mailboxes for every registered agent, plus counters used as KPIs."""

    def __init__(self, model):
        self.model = model
        self.mailboxes = {}
        self.sent = 0
        self.sent_by_type = {}

    def register(self, agent_id):
        self.mailboxes.setdefault(agent_id, [])

    def send(self, performative, sender, receiver, content=None):
        """Send a message to one agent."""
        msg = Message(performative, sender, receiver, content or {}, self.model.schedule.steps)
        self.mailboxes[receiver].append(msg)
        self._count(performative)
        return msg

    def broadcast(self, performative, sender, receivers, content=None):
        """Send the same message to every agent in ``receivers``.

        It counts as one transmission, as a radio broadcast would.
        """
        for r in receivers:
            msg = Message(performative, sender, None, content or {}, self.model.schedule.steps)
            self.mailboxes[r].append(msg)
        self._count(performative)

    def receive(self, agent_id):
        """Return and clear every message waiting for ``agent_id``."""
        msgs = self.mailboxes.get(agent_id, [])
        self.mailboxes[agent_id] = []
        return msgs

    def count(self, performative, n=1):
        """Count transmissions that happen outside ``send`` (e.g. auction clock rounds)."""
        self.sent += n
        self.sent_by_type[performative] = self.sent_by_type.get(performative, 0) + n

    def _count(self, performative):
        self.count(performative)

    def end_step(self):
        """Called by the model after every step (used by networked buses)."""

    def close(self):
        """Release resources (used by networked buses)."""


class LossyBus(MessageBus):
    """
    A simulated radio link: each copy of a message is lost with probability
    ``drop_prob`` and otherwise arrives ``delay`` steps after it was sent.

    Robots have to cope on their own. Drones re-announce crops that stay ripe, and
    an auction that receives no bids simply runs again later.
    """

    def __init__(self, model, drop_prob=0.1, delay=1):
        super().__init__(model)
        self.drop_prob = drop_prob
        self.delay = delay
        self.dropped = 0
        self.in_flight = []          # (deliver_at, receiver, message)

    def _post(self, receiver, message):
        if self.model.random.random() < self.drop_prob:
            self.dropped += 1
            return
        self.in_flight.append((message.time + self.delay, receiver, message))

    def send(self, performative, sender, receiver, content=None):
        msg = Message(performative, sender, receiver, content or {}, self.model.schedule.steps)
        self._post(receiver, msg)
        self._count(performative)
        return msg

    def broadcast(self, performative, sender, receivers, content=None):
        for r in receivers:
            self._post(r, Message(performative, sender, None, content or {}, self.model.schedule.steps))
        self._count(performative)

    def receive(self, agent_id):
        now = self.model.schedule.steps
        arrived = [m for t, r, m in self.in_flight if r == agent_id and t <= now]
        if arrived:
            self.in_flight = [x for x in self.in_flight if not (x[1] == agent_id and x[0] <= now)]
        return arrived
