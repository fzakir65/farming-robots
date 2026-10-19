"""LLM coordinator: a drone asks a local language model (Ollama) to allocate crop areas.

The drone collects new ripe areas for a few steps, then sends one prompt describing
the areas (urgency, number of trees) and the pickers (battery, free storage, queued
work, travel steps to every area). The model must answer in JSON that follows a
schema (Ollama's structured outputs). Every answer is **validated**: unknown ids,
duplicate areas and storage overflows are removed. If the call fails or times out,
the drone falls back to the nearest-free-picker heuristic and the failure is
counted, so the KPIs show what the LLM really contributed.

Everything runs locally (default model ``qwen2.5:3b``); nothing leaves the machine.
Answers are cached by prompt, so re-running a seed gives the same result.
"""
import hashlib
import json
import time

from .constants import LLM_MODEL, LLM_TIMEOUT

SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"area": {"type": "integer"}, "picker": {"type": "integer"}},
                "required": ["area", "picker"],
            },
        }
    },
    "required": ["assignments"],
}

SYSTEM = (
    "You coordinate strawberry-picking robots on a farm. Assign each ripe crop area to at "
    "most one picker robot. Prefer short travel, and give urgent areas (urgency near 1 is "
    "about to rot) to pickers that can reach them soonest. A picker's total trees must not "
    "exceed its free storage. Pickers with low battery should get few or no areas. "
    "Answer only with JSON matching the schema."
)


def build_prompt(areas, pickers):
    """``areas``: [{"area", "trees", "urgency"}]; ``pickers``: [{"picker", "battery",
    "free_storage", "queued", "travel": {area: steps}}]"""
    lines = ["Ripe crop areas:"]
    for a in areas:
        lines.append(f"- area {a['area']}: {a['trees']} trees, urgency {a['urgency']:.2f}")
    lines.append("Picker robots:")
    for p in pickers:
        travel = ", ".join(f"area {k}: {v} steps" for k, v in p["travel"].items())
        lines.append(
            f"- picker {p['picker']}: battery {p['battery']:.0f}%, free storage {p['free_storage']}, "
            f"queued tasks {p['queued']}; travel {travel}"
        )
    lines.append('Return {"assignments": [{"area": <id>, "picker": <id>}, ...]}.')
    return "\n".join(lines)


def validate(answer, areas, pickers):
    """Keep only assignments that refer to real ids and respect free storage."""
    trees = {a["area"]: a["trees"] for a in areas}
    free = {p["picker"]: p["free_storage"] for p in pickers}
    result, invalid = {}, 0
    for item in answer.get("assignments", []):
        try:
            area, picker = int(item["area"]), int(item["picker"])
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if area not in trees or picker not in free or area in result or free[picker] < trees[area]:
            invalid += 1
            continue
        free[picker] -= trees[area]
        result[area] = picker
    return result, invalid


class OllamaCoordinator:
    def __init__(self, model_name=LLM_MODEL, timeout=LLM_TIMEOUT, client=None):
        self.model_name = model_name
        self.timeout = timeout
        self._client = client
        self.cache = {}
        self.calls = 0
        self.failures = 0
        self.invalid = 0
        self.seconds = 0.0

    @property
    def client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(timeout=self.timeout)
        return self._client

    def assign(self, areas, pickers):
        """Returns {area: picker} or None if the model could not be used."""
        prompt = build_prompt(areas, pickers)
        key = hashlib.sha256(f"{self.model_name}\n{prompt}".encode()).hexdigest()
        if key not in self.cache:
            self.calls += 1
            t0 = time.perf_counter()
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    format=SCHEMA,
                    options={"temperature": 0, "seed": 0},
                )
                content = response["message"]["content"]
                self.cache[key] = json.loads(content)
            except Exception:
                self.failures += 1
                return None
            finally:
                self.seconds += time.perf_counter() - t0
        result, invalid = validate(self.cache[key], areas, pickers)
        self.invalid += invalid
        return result
