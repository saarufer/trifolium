"""Structured observability for the agent — every decision is recorded and
explainable. A good agent can show its reasoning, not just its output."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Trace:
    """Append-only structured trace of the agent's run. Each event is one JSON
    line; a human-readable summary is also streamed to stdout."""
    path: Path | None = None
    t0: float = field(default_factory=time.time)
    events: list = field(default_factory=list)

    def emit(self, kind: str, **fields):
        ev = {"t": round(time.time() - self.t0, 2), "kind": kind, **fields}
        self.events.append(ev)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(ev) + "\n")
        # human line
        msg = " ".join(f"{k}={v}" for k, v in fields.items()
                       if k not in ("smiles_list",))
        print(f"[{ev['t']:7.1f}s] {kind:<10} {msg}", file=sys.stdout, flush=True)
        return ev

    # convenience semantic events ------------------------------------------
    def perceive(self, **obs):
        return self.emit("perceive", **obs)

    def decide(self, action, reason, brain):
        return self.emit("decide", action=action, brain=brain, reason=reason)

    def act(self, action, **result):
        return self.emit("act", action=action, **result)

    def reflect(self, note, **fields):
        return self.emit("reflect", note=note, **fields)

    def champion(self, smiles, vina, sa, routable, **extra):
        return self.emit("champion", smiles=smiles, vina=round(vina, 3),
                         sa=round(sa, 2), routable=routable, **extra)

    def final(self, **fields):
        return self.emit("final", **fields)
