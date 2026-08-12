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

import os
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
                 max_rounds=None, act_watchdog_s=300.0):
        self.backend = backend
        self.max_rounds = max_rounds       # hard round cap (None = time-budget only)
        self.act_watchdog_s = act_watchdog_s   # per-batch safety cap; large = let batches finish
        # parallel docking: dock a batch concurrently (vina is the bottleneck and is
        # independent per molecule). Default to (cores-1); each vina pinned to 1 core.
        self.dock_workers = int(os.environ.get("DOCK_WORKERS", "0")) or max(1, (os.cpu_count() or 2) - 1)
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
        # strategist memory + circuit-breaker (avoid the idle-spin: same top-5 ->
        # same diagnosis repeated, designs all rejected, API burned for nothing)
        self._strat_diag_history = []         # past diagnoses (so LLM doesn't repeat itself)
        self._strat_last_reject = (0, 0)      # (rejected_flexible, not_improved) last call
        self._strat_best_at_call = None       # champion vina at the previous strategist call
        self._strat_dry_calls = 0             # consecutive REPEATING strategist calls (looping)
        self._strat_disabled = False          # circuit-breaker: stop consulting once tripped
        self.strat_dry_limit = 3              # disable strategist after this many looping calls
        self.strat_repeat_sim = 0.6           # diagnosis Jaccard >= this => "repeating itself"
        # Boötes — early start-quality: oversample first injection + restart a bad start
        self._strat_calls = 0                 # how many times strategist has been consulted
        self._strat_force_restart = False     # set by the early-restart check below
        self._strat_restarts = 0              # how many early restarts have fired
        self.strat_oversample = 18            # ① first LLM call requests this many DIVERSE designs
        # Eridanus — RULE-OPEN, LLM-REFINE: rules run alone until warmed up, THEN LLM.
        self.strat_warmup_rounds = 15         # no strategist before this round …
        self.strat_warmup_vina = -12.0        # … unless rules already reached this depth (start early)
        self.strat_reflect_every = 3          # Fornax: every 3rd LLM call is a strong-reflection call

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

    def _admit_many(self, smiles_list):
        """Admit a batch with PARALLEL docking — the dock (vina subprocess) is the
        bottleneck and is independent per molecule, so we run up to `dock_workers`
        at once (each vina pinned to 1 core via VINA_CPU). Canonicalize+dedup serially
        (cheap), dock the fresh ones concurrently, then admit results serially (state
        mutation has no contention). Returns the list of admitted Eval objects."""
        from rdkit import Chem
        fresh = []
        for smi in smiles_list:
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            canon = Chem.MolToSmiles(m)
            if canon in self.state.seen:
                continue
            self.state.seen.add(canon)      # reserve now so duplicates in the batch dedup too
            fresh.append(canon)
        if not fresh:
            return []
        if self.dock_workers <= 1 or self.backend is None:
            scores = [self._score(s) for s in fresh]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.dock_workers) as ex:
                scores = list(ex.map(self._score, fresh))
        admitted = []
        for canon, v in zip(fresh, scores):
            ev = objective.evaluate(canon, v)
            self.state.pop.append(ev)
            admitted.append(ev)
        return admitted

    # ── the loop ──────────────────────────────────────────────────────────
    def run(self):
        # seed the population (never-empty principle)
        for s in self._seed_smiles:
            self._admit(s)
        self._update_best(seeded=True)

        while True:
            self.state.round += 1
            self._best_history.append(round(self.state.best.vina, 2) if self.state.best else None)

            # ── STRATEGIST (LLM) — RULE-OPEN, LLM-REFINE (Eridanus). The old "LLM opens"
            # order had a ~50% crash rate (Virgo −9.76, Draco −9.38 stuck): LLM early
            # sampling has huge variance, and the rule loop can only deepen AROUND the
            # start LLM gives — a bad start dooms the run. Rules are DETERMINISTIC and
            # reliably reach −11~−12 (Aquarius), so let RULES open: no strategist until
            # the population has warmed up (round >= warmup OR champion already deep).
            # Then LLM refines a GOOD population — its variance now only affects "extra
            # gain", never "crash". This puts LLM where its value is highest (strategic
            # scaffold/linker redesign on a solid base) and its variance hurts least.
            warmed = (self.state.round >= self.strat_warmup_rounds
                      or (self.state.best is not None
                          and self.state.best.vina <= self.strat_warmup_vina))
            if (self.strategist is not None and not self._strat_disabled and warmed
                    and self.state.round % self.strategy_every == 1
                    and self.state.budget_frac() < 0.9):
                self._consult_strategist()

            obs = self.state.observe()
            self.trace.perceive(**obs)

            d = self.brain.decide(obs)
            self.trace.decide(d.action, d.reason, d.brain)

            if d.action == "STOP":
                break
            # PRE-ACT budget guard: don't START a new (parallel) dock batch once the
            # per-target budget is spent — a batch can take ~minutes, so stop here to
            # avoid overrunning the budget on the final round.
            if self.state.budget_frac() >= 1.0:
                self.trace.reflect("budget exhausted (pre-act) — stopping")
                break
            self._act(d.action)
            self._update_best()

            if self.state.budget_frac() >= 1.0:
                self.trace.reflect("budget exhausted — stopping")
                break
            if self.max_rounds is not None and self.state.round >= self.max_rounds:
                self.trace.reflect("round cap reached — stopping", rounds=self.state.round)
                break

        # FINAL HIGH-EXH CONFIRMATION: the molecule we actually submit must be verified
        # at high exhaustiveness, not left on its fast exh8 score. Re-dock the selected
        # champion at HIGH_EXH; if it regresses (was a false positive), correct its score
        # and re-select — repeat a few times until the top survivor is high-exh-stable.
        if self.backend is not None:
            from docking import HIGH_EXH
            for _ in range(4):
                champ = objective.select_champion(self.state.pop)
                if champ is None:
                    break
                hi = self.backend.dock(champ.smiles, exh=HIGH_EXH)
                if hi is None:
                    break
                delta = hi - champ.vina
                for e in self.state.pop:
                    if e.smiles == champ.smiles:
                        e.vina = hi
                self.trace.reflect("final high-exh confirm",
                                   **{"detail": f"exh8={champ.vina:.2f} exh{HIGH_EXH}={hi:.2f} d={delta:+.2f}"})
                if delta <= 1.0:
                    break   # confirmed stable -> this is our champion
                # else regressed: loop re-selects (a deeper confirmed mol may now win)
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
        best_before = self.state.best.vina if self.state.best else None
        # MEMORY + FAILURE FEEDBACK: tell the LLM what it already tried and why its
        # last batch failed, so it stops repeating the same diagnosis on a frozen
        # top-5. Without this the strategist idle-spins once the score plateaus.
        memo = {
            "past_diagnoses": self._strat_diag_history[-3:],
            "last_designs_rejected_flexible": self._strat_last_reject[0],
            "last_designs_no_improvement": self._strat_last_reject[1],
            "note": ("Your previous designs did NOT beat the current best. Do NOT repeat "
                     "the directions above — propose a genuinely different scaffold/linker."
                     if self._strat_diag_history else ""),
        }
        # ① OVERSAMPLE on the first call (and on a restart): the first injection is
        # one-shot and high-variance — a single mediocre batch dooms the whole run
        # (Virgo: −9.76 vs Pisces −16.49, same seed, diff only in LLM sampling). So
        # ask for many DIVERSE designs, dock all, keep the deepest. ③ diverse=True.
        first = (self._strat_calls == 0) or self._strat_force_restart
        n_want = self.strat_oversample if first else 6
        # INTERVAL STRONG-REFLECTION (Fornax): most calls explore (temp 0.4, diverse);
        # every Nth call (and whenever the LLM started repeating itself last time) is a
        # low-temp REFLECTION call that forces "why are we stuck → attack that cause".
        # Explore保上限, reflect破停滞 —— 分工, 不互相牺牲. Never reflect on the first call.
        reflect = (not first) and (
            (self._strat_calls + 1) % self.strat_reflect_every == 0
            or self._strat_dry_calls >= 1)
        advice = self.strategist.advise(self.target_desc, top, self._best_history, memo,
                                        n_want=n_want, diverse=first, reflect=reflect)
        self._strat_calls += 1
        self._strat_force_restart = False
        self.trace.emit("strategist", diagnosis=advice["diagnosis"],
                        focus=advice["focus"], n_designs=len(advice["designs"]),
                        oversample=first, n_want=n_want, reflect=reflect)
        self._strat_diag_history.append(advice["diagnosis"][:160])
        rejected_flex = 0
        # RIGIDITY GATE (before docking): reject FLEXIBLE designs (rot > cap) up front,
        # then dock the rest IN PARALLEL via _admit_many (the 18-design oversample batch
        # is the heaviest dock load — parallelism matters most here).
        passable = []
        for smi in advice["designs"]:
            if not objective.rigidity_compliant(smi):
                rejected_flex += 1
            else:
                passable.append(smi)
        admitted = self._admit_many(passable)
        injected = len(admitted)
        injected_log = [(ev.smiles, round(ev.vina, 2)) for ev in admitted]  # ④ reproducibility
        if first and injected_log:
            # ④ persist the exact first-injection set + scores (Anthropic API has no
            # strict seed; this lets a good start be inspected/reproduced).
            self.trace.emit("strategist_injection", call=self._strat_calls,
                            molecules=injected_log)
        if injected or rejected_flex:
            self._update_best()
        # CIRCUIT-BREAKER (Scorpio): trip on REPETITION, not on non-improvement.
        # The Libra(G2) breaker stopped after K calls with no champion gain — but the
        # LLM's payoff is LAGGED (G100's breakthroughs came at t=886/t=2085 after long
        # plateaus), so a non-improvement breaker kills the LLM right before its most
        # valuable move. The real waste is the LLM REPEATING ITSELF (same diagnosis on a
        # frozen top-5). So we count consecutive near-duplicate diagnoses instead: as long
        # as the LLM proposes genuinely new directions we let it run; we cut it only once
        # it starts looping. (improvement is still logged but does NOT drive the breaker.)
        best_after = self.state.best.vina if self.state.best else None
        improved = (best_before is not None and best_after is not None
                    and best_after < best_before - 1e-6)
        not_improved = injected if not improved else 0
        self._strat_last_reject = (rejected_flex, max(0, not_improved))
        sim = self._diag_similarity(advice["diagnosis"])   # vs the PREVIOUS diagnosis
        repeating = sim >= self.strat_repeat_sim           # LLM is looping
        if improved:
            self._strat_dry_calls = 0                      # real progress resets the counter
        elif repeating:
            self._strat_dry_calls += 1                     # looping AND no gain -> count it
            if self._strat_dry_calls >= self.strat_dry_limit:
                self._strat_disabled = True
        else:
            self._strat_dry_calls = 0                      # new direction (even if no gain) -> let it run
        self.trace.reflect("strategist designs injected", injected=injected,
                           rejected_flexible=rejected_flex, focus=advice["focus"],
                           improved=improved, diag_similarity=round(sim, 2),
                           repeating=repeating, dry_calls=self._strat_dry_calls,
                           strategist_disabled=self._strat_disabled)

    def _diag_similarity(self, diagnosis: str) -> float:
        """Jaccard word-overlap between this diagnosis and the previous one (0..1).
        High => the LLM is repeating itself (looping on a frozen top-5). No deps."""
        if len(self._strat_diag_history) < 2:
            return 0.0
        prev, cur = self._strat_diag_history[-2], self._strat_diag_history[-1]
        a = set(prev.lower().split())
        b = set(cur.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # ── actions ─────────────────────────────────────────────────────────
    def _act(self, action):
        if action == "CONSENSUS":
            self._consensus()
            return
        parents = self._top_parents(k=6)
        # GENERATE all children first (cheap, serial), then DOCK them in parallel.
        # Generation is fast; docking is the bottleneck — batching the docks lets
        # _admit_many run them concurrently across cores instead of one-at-a-time.
        children = []
        for _ in range(self.batch):
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
            if child and len(child) <= 200:
                children.append(child)
        made = len(self._admit_many(children))
        self.trace.act(action, made=made, pop=len(self.state.pop))

    def _consensus(self):
        """Confirm the champion with a HIGH-EXHAUSTIVENESS re-dock (two-stage exh).
        The pool was scored at EXH=8 (fast). A genuinely deep hit stays deep at HIGH_EXH;
        a false positive (lucky single pose at exh8) regresses. We overwrite the
        champion's vina with the high-exh score — that becomes its TRUSTED value, so a
        regressed false positive sinks in the pool and a confirmed hit is locked in.
        Cheaper + more discriminative than n× re-docking at the same exhaustiveness."""
        champ = self.state.best
        if champ is None or self.backend is None:
            self.state.champion_consensus_done = True
            return
        from docking import HIGH_EXH
        hi = self.backend.dock(champ.smiles, exh=HIGH_EXH)
        self.state.champion_consensus_done = True
        if hi is None:
            return
        delta = hi - champ.vina   # positive => regressed (less negative) => false positive
        note = f"exh8={champ.vina:.2f} -> exh{HIGH_EXH}={hi:.2f} (delta={delta:+.2f})"
        # Overwrite the champion's trusted score with the high-exh value, then re-select:
        # if it regressed, a deeper truly-confirmed molecule may now win.
        for e in self.state.pop:
            if e.smiles == champ.smiles:
                e.vina = hi
        self.state.best = None  # force re-selection from the corrected pool
        if delta > 1.0:
            self.trace.reflect("champion regressed at high exh (false positive corrected)",
                               **{"detail": note})
        else:
            self.trace.reflect("champion confirmed at high exh", **{"detail": note})

    # ── helpers ────────────────────────────────────────────────────────
    def _top_parents(self, k=6):
        # Select parents by Vina with a FLEXIBILITY penalty (objective.selection_score),
        # NOT raw Vina and NOT ligand efficiency. Raw Vina lets floppy molecules
        # free-ride on a false-positive deep score; ligand efficiency wrongly punished
        # large RIGID polycycles (A's winning annulation route). The flex penalty keeps
        # rigid deep binders winning while floppy false-positives sink. Size is allowed.
        pool = [e for e in self.state.pop if e.valid]
        pool.sort(key=objective.selection_score)     # most-negative (deep+rigid) first
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
