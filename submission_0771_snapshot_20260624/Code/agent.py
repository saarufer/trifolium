"""The autonomous agent: an explicit-state perceive -> decide -> act -> reflect
loop that designs a deep-binding, synthesizable, route-feasible molecule.

This is the orchestrator that ties together:
  brain.py       -> the policy that chooses the next action
  generate.py    -> the molecule operators each action uses
  docking.py     -> Vina scoring (deeper = better)
  objective.py   -> multi-objective evaluation + Pareto selection
  observability  -> a full structured trace of every decision
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import generate
import objective
from observability import Trace


@dataclass
class AgentState:
    pop: list = field(default_factory=list)        # list[objective.Eval], the population
    seen: set = field(default_factory=set)         # canonical SMILES already evaluated
    best: object = None                            # objective.Eval champion so far
    stagnation: int = 0                            # rounds since best improved
    round: int = 0
    champion_consensus_done: bool = False
    t0: float = field(default_factory=time.time)
    budget_s: float = 1200.0

    def budget_frac(self):
        return min(1.0, (time.time() - self.t0) / self.budget_s)

    def diversity(self):
        # fraction of unique Bemis-Murcko scaffolds in the population (0..1)
        if not self.pop:
            return 0.0
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffs = set()
        for e in self.pop[:50]:
            m = Chem.MolFromSmiles(e.smiles)
            if m:
                try:
                    scaffs.add(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
                except Exception:
                    pass
        return round(len(scaffs) / max(1, len(self.pop[:50])), 3)

    def observe(self):
        return {
            "round": self.round,
            "pop_size": len(self.pop),
            "best_vina": round(self.best.vina, 2) if self.best else None,
            "best_sa": round(self.best.sa, 2) if self.best else None,
            "best_routable": self.best.routable if self.best else None,
            "stagnation": self.stagnation,
            "diversity": self.diversity(),
            "champion_consensus_done": self.champion_consensus_done,
            "budget_frac": round(self.budget_frac(), 3),
        }


class Agent:
    def __init__(self, backend, brain, seeds, trace: Trace,
                 budget_s=1200.0, batch=8, rng_seed=12345,
                 strategist=None, strategy_every=5, target_desc="the protein target",
                 max_rounds=None):
        self.backend = backend
        self.max_rounds = max_rounds       # hard round cap (None = time-budget only)
        self.brain = brain
        self.trace = trace
        self.rng = random.Random(rng_seed)
        self.state = AgentState(budget_s=budget_s)
        self.batch = batch
        self._seed_smiles = seeds
        self.strategist = strategist          # optional LLM diagnose+design
        self.strategy_every = strategy_every
        self.target_desc = target_desc
        self._best_history = []               # best vina per round (for the strategist)

    # ── scoring: dock (real) or fall back to a deterministic surrogate ─────
    def _score(self, smiles):
        v = self.backend.dock(smiles) if self.backend else None
        if v is None:
            v = _surrogate_vina(smiles)     # offline proxy so the loop still runs
        return v

    def _admit(self, smiles):
        """Evaluate a candidate and add it to the population if new + valid."""
        if not smiles:
            return None
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        canon = Chem.MolToSmiles(m)
        if canon in self.state.seen:
            return None
        self.state.seen.add(canon)
        ev = objective.evaluate(canon, self._score(canon))
        self.state.pop.append(ev)
        return ev

    # ── the loop ──────────────────────────────────────────────────────────
    def run(self):
        # seed the population (never-empty principle)
        for s in self._seed_smiles:
            self._admit(s)
        self._update_best(seeded=True)

        while True:
            self.state.round += 1
            self._best_history.append(round(self.state.best.vina, 2) if self.state.best else None)

            # ── STRATEGIST (LLM): every N rounds, diagnose + design new candidates.
            # This is the LLM's job — read structures, reason chemically, inject
            # directed molecules the rule/GA loop could never mutate into.
            if (self.strategist is not None and
                    self.state.round % self.strategy_every == 1 and
                    self.state.budget_frac() < 0.9):
                self._consult_strategist()

            obs = self.state.observe()
            self.trace.perceive(**obs)

            d = self.brain.decide(obs)
            self.trace.decide(d.action, d.reason, d.brain)

            if d.action == "STOP":
                break
            self._act(d.action)
            self._update_best()

            if self.state.budget_frac() >= 1.0:
                self.trace.reflect("budget exhausted — stopping")
                break
            if self.max_rounds is not None and self.state.round >= self.max_rounds:
                self.trace.reflect("round cap reached — stopping", rounds=self.state.round)
                break

        champ = objective.select_champion(self.state.pop)
        self.trace.final(
            smiles=champ.smiles if champ else None,
            vina=round(champ.vina, 3) if champ else None,
            sa=round(champ.sa, 2) if champ else None,
            routable=champ.routable if champ else None,
            reaction=champ.reaction if champ else None,
            rounds=self.state.round,
            evaluated=len(self.state.seen),
        )
        return champ

    # ── strategist consultation (LLM diagnose + design) ──────────────────
    def _consult_strategist(self):
        pool = sorted([e for e in self.state.pop if e.valid], key=lambda e: e.vina)
        top = [(e.smiles, e.vina, e.sa) for e in pool[:5]]
        if not top:
            return
        advice = self.strategist.advise(self.target_desc, top, self._best_history)
        self.trace.emit("strategist", diagnosis=advice["diagnosis"],
                        focus=advice["focus"], n_designs=len(advice["designs"]))
        injected = 0
        for smi in advice["designs"]:
            # validate + dock + admit the LLM's design (LLM can emit junk; the
            # objective/docking gate keeps only the real, scorable molecules)
            if self._admit(smi) is not None:
                injected += 1
        if injected:
            self.trace.reflect("strategist designs injected", injected=injected,
                               focus=advice["focus"])
            self._update_best()

    # ── actions ─────────────────────────────────────────────────────────
    def _act(self, action):
        if action == "CONSENSUS":
            self._consensus()
            return
        parents = self._top_parents(k=6)
        made = 0
        t_act = time.time()
        for _ in range(self.batch):
            # watchdog: never let one action batch run away (a pathological
            # molecule can stall BRICS/embedding). Bail out of the batch.
            if time.time() - t_act > 20.0 or self.state.budget_frac() >= 1.0:
                self.trace.reflect("act batch watchdog tripped", action=action)
                break
            child = None
            try:
                if action == "EXPLOIT":
                    child = generate.mutate(self.rng.choice(parents), self.rng)
                elif action == "ANNEAL":
                    child = generate.annulate(self.rng.choice(parents), self.rng)
                elif action == "EXPLORE":
                    a, b = self.rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
                    child = generate.cross(a, b, self.rng)
            except Exception:
                child = None
            if child and len(child) <= 200 and self._admit(child) is not None:
                made += 1
        self.trace.act(action, made=made, pop=len(self.state.pop))

    def _consensus(self):
        """Re-dock the current champion several times; if the pose is unstable
        (high std) the deep score is likely a false positive -> penalize it."""
        champ = self.state.best
        if champ is None or self.backend is None:
            self.state.champion_consensus_done = True
            return
        mean, std = self.backend.dock_consensus(champ.smiles, n=3)
        self.state.champion_consensus_done = True
        if mean is None:
            return
        note = f"consensus mean={mean:.2f} std={std:.2f}"
        if std is not None and std > 1.5:
            # unstable -> distrust: replace the champion's vina with the (worse) mean
            for e in self.state.pop:
                if e.smiles == champ.smiles:
                    e.vina = mean
            self.state.best = None  # force re-selection from the corrected pool
            self.trace.reflect("champion rejected as unstable false positive", **{"detail": note})
        else:
            self.trace.reflect("champion confirmed stable", **{"detail": note})

    # ── helpers ────────────────────────────────────────────────────────
    def _top_parents(self, k=6):
        pool = [e for e in self.state.pop if e.valid]
        pool.sort(key=lambda e: e.vina)            # deepest first
        parents = [e.smiles for e in pool[:k]]
        return parents or self._seed_smiles

    def _update_best(self, seeded=False):
        champ = objective.select_champion(self.state.pop)
        prev = self.state.best
        if champ is None:
            return
        improved = (prev is None) or (champ.vina < prev.vina - 1e-6) or \
                   (champ.smiles != prev.smiles and champ.scalar() > prev.scalar() + 1e-6)
        self.state.best = champ
        if improved:
            self.state.stagnation = 0
            self.state.champion_consensus_done = False
            if not seeded:
                self.trace.champion(champ.smiles, champ.vina, champ.sa,
                                    champ.routable, reaction=champ.reaction)
        else:
            self.state.stagnation += 1


def _surrogate_vina(smiles):
    """Deterministic offline proxy for Vina when docking is unavailable, so the
    full agent loop is demonstrable without Docker. NOT a real binding score —
    it rewards aromatic ring count + size (the validated weak real signals) and
    penalizes greasiness, roughly tracking how Vina deepens with polycyclic
    systems. Replace with real docking for actual results."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0.0
    import math
    arom = rdMolDescriptors.CalcNumAromaticRings(m)
    rings = rdMolDescriptors.CalcNumRings(m)
    mw = Descriptors.MolWt(m)
    logp = Descriptors.MolLogP(m)
    # Real Vina PLATEAUS with size — use saturating (log/sqrt) terms, not linear,
    # so the surrogate also plateaus (~-13 to -14) instead of improving forever.
    score = -(5.0 + 2.5 * math.log1p(arom) + 1.2 * math.sqrt(rings)
              + 2.0 * (1.0 - math.exp(-mw / 350.0)))
    score += 0.5 * max(0.0, logp - 5.0)        # too greasy -> worse
    score += 0.4 * max(0.0, mw - 550) / 100.0  # oversize -> worse (Vina box limits)
    return round(score, 2)
