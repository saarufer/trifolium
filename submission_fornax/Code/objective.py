"""Multi-objective evaluation: the agent does NOT greedily chase Vina alone.

Goals (in priority order):
  1. Vina as deep (most negative) as possible        -> primary axis
  2. SA reasonable (synthesizable)                   -> soft gate
  3. Route reasonable (a named disconnection exists) -> soft gate

Design choices (from EXPERIENCE.md):
  * Chasing Vina alone yields greasy, high-SA, unsynthesizable monsters and Vina
    false positives. So SA and route are constraints, not ignored.
  * SA soft gate: SA<4 preferred; SA>=4 accepted only if Vina is stronger by at
    least SA_VINA_OFFSET kcal — the trade that offsets the SA score penalty.
  * We keep a PARETO FRONT over (vina, sa) rather than collapsing to one scalar,
    so the agent can reason about trade-offs and the final pick is defensible.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

import route as route_mod

SA_SOFT = 4.0            # SA below this is "reasonable"
SA_VINA_OFFSET = 2.3     # SA>=4 accepted only if Vina stronger by this many kcal
MW_CAP = 650.0           # above this, docking is slow and molecules get silly
GREASY_AROM_RING_MIN = 0 # (kept for reference; greasiness penalised in score)


def sa_score(smiles: str) -> float:
    m = Chem.MolFromSmiles(smiles)
    return sascorer.calculateScore(m) if m else 99.0


ROT_CAP = 6              # rotatable-bond cap (rigidity gate; chemistry, not speed)
FLEX_PENALTY = 0.30      # kcal added to selection score per rotatable bond


def rigidity_compliant(smiles: str) -> bool:
    """RIGIDITY gate for strategist designs: reject FLEXIBLE molecules (rotatable
    bonds > ROT_CAP), NOT large ones. Chemistry, not engineering convenience:
      * Vina systematically OVER-scores flexible ligands (finds an entropically
        impossible pose in the box) -> deep Vina on a floppy molecule is a likely
        false positive. Rigid molecules' deep scores are more trustworthy.
      * Conformational restriction lowers the entropy penalty on binding and
        improves oral absorption (Veber: rotatable bonds is the key descriptor).
    Large+polycyclic+RIGID (high rings, low rot) is GOOD — that is exactly the
    rule-loop's annulation route to deep, real binders. Size is not the problem;
    flexibility is. Invalid SMILES -> rejected."""
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return False
    return Descriptors.NumRotatableBonds(m) <= ROT_CAP


def selection_score(e) -> float:
    """Parent-selection key (most-negative = best): Vina with a FLEXIBILITY
    penalty. A deep score earned by a rigid molecule beats an equally deep score
    from a floppy one, because the floppy one's depth is likely a Vina false
    positive (entropy not paid). This does NOT penalize size — a large rigid
    polycycle keeps its deep score, so the annulation route to deep binders is
    preserved (unlike ligand-efficiency, which wrongly penalized A's winners)."""
    m = Chem.MolFromSmiles(e.smiles)
    rot = Descriptors.NumRotatableBonds(m) if m else 20
    return e.vina + FLEX_PENALTY * rot


def _greasy(smiles: str) -> bool:
    """A long aliphatic chain with no rings tends to be a Vina false positive."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return True
    n_arom = Chem.rdMolDescriptors.CalcNumAromaticRings(m)
    n_rot = Descriptors.NumRotatableBonds(m)
    return n_arom == 0 and n_rot >= 8


@dataclass
class Eval:
    smiles: str
    vina: float          # most-negative is best (e.g. -12.4)
    sa: float
    mw: float
    routable: bool
    reaction: str | None
    route: str | None
    valid: bool
    greasy: bool

    def acceptable(self) -> bool:
        """Passes the soft gates: valid, not greasy-false-positive, MW sane, and
        SA reasonable OR compensated by a strong-enough Vina."""
        if not self.valid or self.greasy or self.mw > MW_CAP:
            return False
        if self.sa < SA_SOFT:
            return True
        # SA is high — accept only if Vina is deep enough to offset it
        return self.vina <= -(11.0 + SA_VINA_OFFSET)  # heuristic deep-enough bar

    def scalar(self) -> float:
        """A single comparable score (higher=better) used only for tie-breaking /
        logging. Vina dominates; SA and route give small nudges. The agent's
        actual selection uses the Pareto front, not this scalar."""
        s = -self.vina                       # deeper vina -> larger
        s -= 0.25 * max(0.0, self.sa - SA_SOFT)   # SA penalty above the soft gate
        s += 0.30 if self.routable else 0.0       # route bonus
        return s


def evaluate(smiles: str, vina: float) -> Eval:
    m = Chem.MolFromSmiles(smiles)
    valid = m is not None
    mw = Descriptors.MolWt(m) if valid else 9999.0
    sa = sascorer.calculateScore(m) if valid else 99.0
    r = route_mod.route_for(smiles) if valid else {"routable": False, "reaction": None, "route": None}
    return Eval(smiles=smiles, vina=vina, sa=sa, mw=mw,
                routable=r["routable"], reaction=r["reaction"], route=r["route"],
                valid=valid, greasy=_greasy(smiles) if valid else True)


def pareto_front(evals: list[Eval]) -> list[Eval]:
    """Non-dominated set over (vina deeper better, sa lower better). Only
    acceptable molecules are considered."""
    pool = [e for e in evals if e.acceptable()]
    front = []
    for e in pool:
        dominated = False
        for o in pool:
            if o is e:
                continue
            # o dominates e if o is >= on both objectives and > on at least one
            if (o.vina <= e.vina and o.sa <= e.sa) and (o.vina < e.vina or o.sa < e.sa):
                dominated = True
                break
        if not dominated:
            front.append(e)
    return front


def select_champion(evals: list[Eval]) -> Eval | None:
    """Pick the final molecule. Routable molecules are preferred — so we compute
    the Pareto front WITHIN the routable set FIRST. (Computing the front over the
    whole pool first is a bug: an unsynthesizable PAH false-positive — e.g. a
    coronene with very deep Vina + low SA — dominates and EVICTS the real
    synthesizable champion from the front, then drops out itself for being
    non-routable, leaving a shallower fallback. Filter routable, THEN Pareto.)"""
    routable = [e for e in evals if e.routable]
    front = pareto_front(routable)
    if front:
        return min(front, key=lambda e: e.vina)  # deepest vina on the routable front
    # no routable winner — fall back to the Pareto front over everything
    front = pareto_front(evals)
    if front:
        return min(front, key=lambda e: e.vina)
    valid = [e for e in evals if e.valid]
    return min(valid, key=lambda e: e.vina) if valid else None
