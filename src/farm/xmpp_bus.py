"""A message bus that sends every robot message over a real XMPP network with SPADE.

Each robot gets a SPADE agent (``farm<run>_<id>@localhost``). Messages are FIPA-style:
the performative travels as SPADE metadata and the content as a JSON body. They go
through the XMPP server (SPADE's built-in ``pyjabber`` server, started on demand), not
SPADE's in-process shortcut, so transport latency is real and measured.

The SPADE agents live in a separate **network process** (``network_main``). Mesa 2
imports Tornado, and with Tornado loaded SPADE's Windows event loop (winloop) stalls,
so keeping the two apart is the robust option. The network process is started as
``python -m farm.xmpp_bus`` (so it never imports Mesa) and talks to the simulation
over an authenticated local socket (``multiprocessing.connection``):

    simulation --(register / send / flush / stop)--> network process --XMPP--> server
    simulation <--(deliveries, acknowledgements)--- network process <--XMPP--

The simulation itself stays step-based. A message sent in step *t* is delivered once
the network has carried it; ``end_step`` waits (up to a timeout) for in-flight
messages, so a receiver reads it in step *t + 1* at the latest.
"""
import atexit
import itertools
import json
import os
import secrets
import socket
import subprocess
import sys
import time

from .constants import XMPP_PORT
from .messages import Message, MessageBus

CONNECT_ATTEMPTS = 4
CONNECT_TIMEOUT = 10.0
TLS_SETTLE = 0.05      # seconds to wait after STARTTLS before restarting the stream
_server_process = None
_run_ids = itertools.count(1)


def _port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def ensure_server(port=XMPP_PORT, timeout=60.0):
    """Start SPADE's XMPP server (``spade run``) unless one is already listening."""
    global _server_process
    if _port_open(port):
        return
    exe = os.path.join(os.path.dirname(sys.executable), "spade.exe" if os.name == "nt" else "spade")
    _server_process = subprocess.Popen(
        [exe, "run", "--host", "localhost", "--memory", "--client_port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_server)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            time.sleep(1.0)             # the port opens slightly before logins work
            return
        time.sleep(0.2)
    raise RuntimeError("the SPADE XMPP server did not start")


def stop_server():
    global _server_process
    if _server_process is not None:
        _server_process.terminate()
        _server_process = None


# ---------------------------------------------------------------------- #
# The network process: one SPADE agent per robot
# ---------------------------------------------------------------------- #
def network_main(conn, port, tag):
    """Entry point of the network process (no Mesa, no Tornado imported here)."""
    import asyncio

    if os.environ.get("FARM_XMPP_LOG"):            # debugging aid: client-side XMPP log
        import logging

        logging.basicConfig(filename=os.environ["FARM_XMPP_LOG"], level=logging.DEBUG,
                            format="%(relativeCreated)d %(name)s %(message)s")
    import spade
    from spade.agent import Agent
    from spade.message import Message as SpadeMessage

    stats = {"sent": 0, "received": 0}

    class RobotLink(Agent):
        """The network side of one robot: it only sends and forwards messages."""

        async def _async_connect(self):
            # "localhost" resolves to ::1 first, but the server listens on IPv4 only;
            # on Windows each refused IPv6 attempt costs ~2 s. The XMPP domain stays
            # "localhost"; only the socket goes to 127.0.0.1.
            connect = self.client.connect

            def connect_ipv4(host=None, port=port, **kwargs):
                return connect(host="127.0.0.1", port=port, **kwargs)

            self.client.connect = connect_ipv4

            # pyjabber race: if the stream restart that follows STARTTLS reaches the
            # server before it has finished its side of the TLS upgrade, the server
            # drops it and the login hangs. Restart the stream a moment later instead.
            client = self.client
            made = client.connection_made

            def connection_made(transport, send_event=True):
                if send_event:
                    return made(transport, send_event)
                client.loop.call_later(TLS_SETTLE, made, transport, False)

            client.connection_made = connection_made
            await super()._async_connect()

        def dispatch(self, msg):
            body = json.loads(msg.body)
            latency = (time.time() - body["sent_at"]) * 1000
            stats["received"] += 1
            conn.send(("deliver", body["target"], msg.get_metadata("performative"), body, latency))
            return []

    links = {}

    errors = []

    async def connect(agent_id):
        # In-band registration and the first login race on a busy server: the login
        # can arrive before the new account is stored. Retrying the same account works.
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            link = RobotLink(f"farm{tag}_{agent_id}@localhost", "farm", port=port)
            try:
                await asyncio.wait_for(link.start(auto_register=True), CONNECT_TIMEOUT)
                return link
            except BaseException as exc:          # noqa: B036 - report and retry
                errors.append(f"attempt {attempt}: {exc!r}")
                try:
                    await asyncio.wait_for(link.stop(), 5)
                except Exception:
                    pass
        return None

    async def main():
        loop = asyncio.get_running_loop()
        while True:
            cmd = await loop.run_in_executor(None, conn.recv)
            kind = cmd[0]
            if kind == "register":
                links[cmd[1]] = await connect(cmd[1])
                conn.send(("registered", cmd[1], links[cmd[1]] is not None, list(errors)))
                errors.clear()
            elif kind == "send":
                _, performative, sender, target, body = cmd
                link = links.get(sender) or next(iter(links.values()))
                m = SpadeMessage(to=str(links[target].jid), body=body)
                m.set_metadata("performative", performative)
                m.sender = str(link.jid)
                m.prepare(link.client).send()          # through the XMPP server
                stats["sent"] += 1
            elif kind == "flush":
                deadline = time.time() + cmd[1]
                while stats["received"] < stats["sent"] and time.time() < deadline:
                    await asyncio.sleep(0.001)
                conn.send(("flushed", stats["sent"], stats["received"]))
            elif kind == "stop":
                await asyncio.gather(*(lk.stop() for lk in links.values() if lk), return_exceptions=True)
                conn.send(("stopped",))
                return

    spade.run(main())


# ---------------------------------------------------------------------- #
# The bus used by the model
# ---------------------------------------------------------------------- #
class XmppBus(MessageBus):
    """Same interface and counters as ``MessageBus``, but transported over XMPP."""

    def __init__(self, model, port=XMPP_PORT, flush_timeout=2.0):
        super().__init__(model)
        self.port = port
        self.flush_timeout = flush_timeout
        self.latencies_ms = []
        self.copies_sent = 0
        self.copies_received = 0
        self.connect_retries = 0
        self.mailboxes = {}
        ensure_server(port)
        from multiprocessing.connection import Listener

        key = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        tag = f"{os.getpid()}x{next(_run_ids)}"
        src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
        self.process = subprocess.Popen(
            [sys.executable, "-m", "farm.xmpp_bus", str(listener.address[1]), key.hex(), str(port), tag],
            env=env, cwd=src,
        )
        self.conn = listener.accept()
        listener.close()

    def _wait_for(self, kind, timeout):
        """Read events until one of type ``kind`` arrives, filing deliveries on the way."""
        deadline = time.time() + timeout
        while True:
            if not self.conn.poll(max(0.01, deadline - time.time())):
                raise TimeoutError(f"no {kind!r} from the network process")
            event = self.conn.recv()
            if event[0] == "deliver":
                self._file(event)
            elif event[0] == kind:
                return event

    def _file(self, event):
        _, target, performative, body, latency = event
        self.copies_received += 1
        self.latencies_ms.append(latency)
        self.mailboxes.setdefault(target, []).append(
            Message(performative, body["sender"], body["receiver"], body["content"], body["time"])
        )

    def register(self, agent_id):
        self.mailboxes.setdefault(agent_id, [])
        self.conn.send(("register", agent_id))
        _, _, ok, errors = self._wait_for("registered", CONNECT_ATTEMPTS * CONNECT_TIMEOUT + 60)
        self.connect_retries += len(errors)
        if not ok:
            raise RuntimeError(f"robot {agent_id} could not connect to the XMPP server: {errors}")

    def _post(self, performative, sender, receiver, target, content):
        body = json.dumps({
            "sender": sender,
            "receiver": receiver,
            "target": target,
            "content": content or {},
            "time": self.model.schedule.steps,
            "sent_at": time.time(),
        })
        self.copies_sent += 1
        self.conn.send(("send", performative, sender, target, body))

    def send(self, performative, sender, receiver, content=None):
        self._post(performative, sender, receiver, receiver, content)
        self._count(performative)
        return Message(performative, sender, receiver, content or {}, self.model.schedule.steps)

    def broadcast(self, performative, sender, receivers, content=None):
        for r in receivers:
            self._post(performative, sender, None, r, content)
        self._count(performative)

    def receive(self, agent_id):
        while self.conn.poll():                     # file whatever has arrived
            event = self.conn.recv()
            if event[0] == "deliver":
                self._file(event)
        msgs = self.mailboxes.get(agent_id, [])
        self.mailboxes[agent_id] = []
        return msgs

    def end_step(self):
        """Wait until the network has delivered everything sent so far."""
        self.conn.send(("flush", self.flush_timeout))
        self._wait_for("flushed", self.flush_timeout + 10)

    @property
    def mean_latency_ms(self):
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.conn.send(("stop",))
                self._wait_for("stopped", 15)
            except (TimeoutError, OSError, EOFError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.process = None


if __name__ == "__main__":
    from multiprocessing.connection import Client

    _port, _key, _xmpp_port, _tag = sys.argv[1:5]
    network_main(Client(("127.0.0.1", int(_port)), authkey=bytes.fromhex(_key)), int(_xmpp_port), _tag)
