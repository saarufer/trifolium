"""Offline molecule generation operators — no LLM, no network.

Three operator classes, used by the agent's strategies:
  * EXPLOIT  -> _mutate:    add/swap a substituent (local refinement)
  * EXPLORE  -> _cross:     BRICS crossover of two parents (jump scaffold)
  * ANNEAL   -> _annulate:  fuse a new ring onto an aromatic edge (build polycyclic
                            systems -> deeper Vina; substituent-only mutation
                            plateaus ~-14, annulation reaches ~-16)

Seeds come from a local ChEMBL index if present (real known binders for the
target), else from a built-in set of drug-like scaffolds — so the agent always
has somewhere to start (never-empty principle).

Algorithms ported and cleaned from the earlier task2 evolve skill.
"""
from __future__ import annotations

import itertools
import os
import random

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


_SUBSTITUENTS = ["F", "Cl", "C", "O", "N", "C#N", "C(F)(F)F", "OC", "S",
                 "C(=O)N", "S(=O)(=O)N", "c1ccccc1", "c1ccncc1", "N1CCOCC1"]

# ring-annulation onto two adjacent aromatic CH (benzo / pyrido / saturated / pyrimido)
_ANNULATE_SMARTS = [
    "[cH:1][cH:2]>>[c:1]1cccc[c:2]1",
    "[cH:1][cH:2]>>[c:1]1cccn[c:2]1",
    "[cH:1][cH:2]>>[c:1]1CCCC[c:2]1",
    "[cH:1][cH:2]>>[c:1]1ncncc[c:2]1",
]

# built-in fallback seeds: diverse drug-like scaffolds (used when no ChEMBL index)
_BUILTIN_SEEDS = [
    "O=C(Nc1ccccc1)c1ccccc1",        # benzanilide
    "c1ccc2[nH]ccc2c1",              # indole
    "c1ccc2ncccc2c1",               # quinoline
    "O=c1[nH]c2ccccc2n1",            # benzimidazolone
    "c1ccc(-c2ccncc2)cc1",          # phenylpyridine
    "O=C(O)c1ccc(N)cc1",            # PABA
    "c1ccc(S(=O)(=O)Nc2ccccc2)cc1",  # sulfonanilide
    "C1CCC(N2CCNCC2)CC1",            # cyclohexyl-piperazine
]

# DEEP-SCAFFOLD seeds (idea-4 aggressive): real champions from our runs that are
# VERIFIED deep + embed-stable + routable on TYK2. They are big fused/sulfonamide-biaryl
# cores that tend to dock deep in many pockets. Per-target docking keeps only the ones
# that score deep ON THAT TARGET — so T1 starts from a −15 core; if a core doesn't suit
# T2/T3 it simply isn't picked (the BRICS-diverse pool covers those). Aggressive: we bet
# "a deep big scaffold start beats a shallow small start" — exactly the platform's reward.
_DEEP_SCAFFOLDS = [
    "NS(=O)(=O)c1cc(-c2cccc(-c3cc4ccccc4c4cc5ccccc5nc34)c2)cc(-c2cccc3cccnc23)n1",  # Aquarius -15.32 sulfonamide-biaryl
    "O=C(Nc1cccc(-c2cc(Cl)c(Cl)c3cc(Cl)c(F)nc23)c1)c1cc(O)c(Cl)cc1-c1cccc(-c2ccccc2)c1",  # Indus -13.92 halogenated
    "Nc1cc2c3ccc4cccc5cc(-c6ccccc6)n(c2cc1C(=O)NC(=O)c1ccc2c(ccc6cccnc62)c1)c3c45",  # Fornax -15.66 azaacridine-imide
]


def _diverse_brics_pool(seeds, rng, n_want=20):
    """GENERATE a structurally diverse seed pool by BRICS-recombining the base seeds —
    NO hand-picked scaffolds (those carry the author's bias toward kinase-like planar
    rings, which would NOT generalize to non-kinase targets). The algorithm recombines
    fragments into novel cores; we then keep only valid, routable, sane-size molecules and
    DEDUP by Murcko scaffold to maximize structural diversity. Each target later docks this
    whole pool and starts from the deepest *for that target* — so a kinase picks planar
    cores, a protease picks polar/charged ones, etc. Diversity = generalization."""
    import route as route_mod
    pool, scaffolds = [], set()
    try:
        frags = []
        for s in seeds:
            m = Chem.MolFromSmiles(s)
            if m:
                frags += list(BRICS.BRICSDecompose(m))
        fmols = [Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f)]
        fmols = fmols[:20]
        if len(fmols) >= 2:
            built = itertools.islice(BRICS.BRICSBuild(fmols), 200)
            for ch in built:
                try:
                    Chem.SanitizeMol(ch)
                    smi = _canon(Chem.MolToSmiles(ch))
                except Exception:
                    continue
                if not smi or len(smi) > 200:
                    continue
                m = Chem.MolFromSmiles(smi)
                if m is None or not (10 <= m.GetNumHeavyAtoms() <= 40):
                    continue
                if not route_mod.route_for(smi)["routable"]:   # must be synthesizable
                    continue
                try:
                    from rdkit.Chem.Scaffolds import MurckoScaffold
                    sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
                except Exception:
                    sc = smi
                if sc in scaffolds:                            # dedup by scaffold = diversity
                    continue
                scaffolds.add(sc)
                pool.append(smi)
                if len(pool) >= n_want:
                    break
    except Exception:
        pass
    return pool


def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def mutate(smi, rng):
    """EXPLOIT: one valid edit — add a substituent on an aromatic CH, or swap a
    halogen. Returns canonical SMILES or None."""
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    op = rng.random()
    try:
        if op < 0.65:
            sub = rng.choice(_SUBSTITUENTS)
            rxn = AllChem.ReactionFromSmarts(f"[cH:1]>>[c:1]{sub}")
            prods = rxn.RunReactants((m,))
            if prods:
                p = prods[rng.randrange(len(prods))][0]
                Chem.SanitizeMol(p)
                return _canon(Chem.MolToSmiles(p))
        else:
            for a, b in [("F", "Cl"), ("Cl", "F"), ("F", "C"), ("Cl", "C#N")]:
                if a in smi:
                    return _canon(smi.replace(a, b, 1))
    except Exception:
        return None
    return None


def annulate(smi, rng, mw_cap=650.0):
    """ANNEAL: fuse a new ring onto adjacent aromatic CH — builds polycyclic
    systems for deeper Vina, MW-capped so it grows but does not explode."""
    m = Chem.MolFromSmiles(smi)
    if not m or Descriptors.MolWt(m) > mw_cap:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts(_ANNULATE_SMARTS[rng.randrange(len(_ANNULATE_SMARTS))])
        prods = list(rxn.RunReactants((m,)))
        rng.shuffle(prods)
        for p in prods:
            try:
                pm = p[0]; Chem.SanitizeMol(pm)
                return _canon(Chem.MolToSmiles(pm))
            except Exception:
                continue
    except Exception:
        return None
    return None


def cross(s1, s2, rng):
    """EXPLORE: BRICS crossover — break both parents, recombine fragments.
    Guarded: BRICSDecompose/BRICSBuild can blow up on large polycyclic inputs,
    so we cap parent size and the number of fragments/built molecules."""
    try:
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if not (m1 and m2):
            return None
        # guard: skip oversized parents (BRICS gets combinatorially expensive)
        if m1.GetNumHeavyAtoms() > 45 or m2.GetNumHeavyAtoms() > 45:
            return None
        frags = list(BRICS.BRICSDecompose(m1)) + list(BRICS.BRICSDecompose(m2))
        fmols = [Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f)]
        fmols = fmols[:12]                          # cap fragment pool size
        if len(fmols) < 2:
            return None
        built = list(itertools.islice(BRICS.BRICSBuild(fmols), 10))
        if built:
            ch = built[rng.randrange(len(built))]
            Chem.SanitizeMol(ch)
            out = _canon(Chem.MolToSmiles(ch))
            return out if out and len(out) <= 200 else None
    except Exception:
        return None
    return None


def load_seeds(uniprot: str | None, index_dir: str | None, n: int = 12) -> list[str]:
    """Return seed SMILES: real known binders from a local ChEMBL index if
    available for this target, else the built-in drug-like scaffolds."""
    seeds = []
    if uniprot and index_dir:
        try:
            import sqlite3
            from pathlib import Path
            idx = Path(index_dir)
            db = next((p for p in [idx / "chembl_actives.db", idx / "chembl.db"]
                       if p.exists()), None)
            if db is not None:
                con = sqlite3.connect(str(db))
                rows = con.execute(
                    "SELECT canonical_smiles FROM actives WHERE uniprot_id=? "
                    "ORDER BY ic50_nm ASC LIMIT ?", (uniprot, n)).fetchall()
                seeds = [r[0] for r in rows if r and r[0]]
                con.close()
        except Exception:
            seeds = []
    seeds = [c for c in (_canon(s) for s in seeds) if c]
    if not seeds:
        seeds = [c for c in (_canon(s) for s in _BUILTIN_SEEDS) if c]
    return seeds[:n]


def pick_deep_seeds(backend, base_seeds, rng=None, n_keep=6, budget_s=None):
    """Idea-4: DEEP-SEED START. Build a pool (deep scaffolds + builtins + BRICS-recombined),
    dock on THIS target, start from the n_keep deepest *here*. Pure Vina. TIME-CAPPED: dock
    the DEEP scaffolds first (they matter most), then the rest, and STOP if budget_s elapses
    — so seed-picking can never eat the whole target budget. Degrades to base seeds if no
    backend. budget_s default from env SEED_PICK_BUDGET_SEC (300s)."""
    import time as _time, random as _random
    rng = rng or _random.Random(12345)
    deadline = _time.time() + (budget_s if budget_s is not None
                               else float(os.environ.get("SEED_PICK_BUDGET_SEC", "300")))
    base = [c for c in (_canon(s) for s in base_seeds) if c]
    deep = [c for c in (_canon(s) for s in _DEEP_SCAFFOLDS) if c]
    # ORDER: deep scaffolds first (highest value), then builtins, then BRICS extras.
    pool = list(dict.fromkeys(deep + base + _diverse_brics_pool(base_seeds, rng, n_want=20)))
    if backend is None or not getattr(backend, "ok", False):
        return (deep + base)[:n_keep] or [c for c in (_canon(s) for s in _BUILTIN_SEEDS) if c]
    seed_exh = int(os.environ.get("SEED_SCOUT_EXH", "4"))
    scored = []
    for smi in pool:
        if _time.time() > deadline and scored:             # time cap: stop, use what we have
            break
        v = backend.dock(smi, exh=seed_exh)
        if v is not None:
            scored.append((v, smi))
    if not scored:
        return (deep + base)[:n_keep]
    scored.sort()                                          # most-negative (deepest) first
    return [smi for _, smi in scored[:n_keep]]


def operator_for(action: str):
    """Map an agent action to the operator class it should use."""
    return {"EXPLOIT": "mutate", "EXPLORE": "cross", "ANNEAL": "annulate"}.get(action)
