"""Offline synthesis-route generation via named-reaction retrosynthesis.

Given a product SMILES, find a single retrosynthetic disconnection at a common,
reliable bond (amide, ester, biaryl/Suzuki, sulfonamide, ether) and return the
two reactants. A molecule is "routable" if a *named* (non-generic) disconnection
exists — that is the proxy for "readily synthesizable" used as a route gate.

Ported and cleaned from the earlier task2 retro_route skill; no network needed.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# A retro rule disconnects a bond and replaces each side with a real functional
# group via an RDKit reaction SMARTS (product >> reactant1.reactant2). Using
# reaction SMARTS (not manual atom surgery) keeps the chemistry correct.
#   amide:   R-C(=O)-N(R')R''  ->  R-C(=O)-OH  +  H-N(R')R''   (amide coupling)
#   ester:   R-C(=O)-O-R'      ->  R-C(=O)-OH  +  HO-R'        (esterification)
#   sulfon:  R-S(=O)2-N(R')R'' ->  R-S(=O)2-Cl +  H-N(R')R''   (sulfonamide)
#   ether:   Ar-O-C            ->  Ar-OH       +  Cl-C         (Williamson)
_RULES = [
    ("amide_coupling",
     "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[O].[N:3]"),
    ("esterification",
     "[C:1](=[O:2])[O:3][#6:4]>>[C:1](=[O:2])[O].[O:3][#6:4]"),
    ("sulfonamide",
     "[S:1](=[O:2])(=[O:3])[N:4]>>[S:1](=[O:2])(=[O:3])[Cl].[N:4]"),
    ("ether",
     "[c:1][O:2][#6:3]>>[c:1][O:2].[Cl][#6:3]"),
]

from rdkit.Chem import AllChem


def _apply(rxn_smarts, mol):
    """Run a retro reaction; return two reactant SMILES (sorted) or None."""
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        for prods in rxn.RunReactants((mol,)):
            parts = []
            ok = True
            for p in prods:
                try:
                    Chem.SanitizeMol(p)
                    parts.append(Chem.MolToSmiles(p))
                except Exception:
                    ok = False
                    break
            if ok and len(parts) == 2 and all(parts):
                return tuple(sorted(parts))
    except Exception:
        return None
    return None


def retro_route(smiles: str):
    """Return (route_str, reaction_name) or None.
    route_str = 'reactant1.reactant2>>product'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for name, rxn_smarts in _RULES:
        pair = _apply(rxn_smarts, mol)
        if pair and smiles not in pair:
            return f"{pair[0]}.{pair[1]}>>{smiles}", name
    return None


def is_routable(smiles: str) -> bool:
    """True if a named (non-generic) disconnection exists = readily synthesizable."""
    return retro_route(smiles) is not None


def route_for(smiles: str):
    """Return a dict describing the best route, or a single-step 'purchase' note
    if no disconnection is found (never empty — always returns *something* valid)."""
    r = retro_route(smiles)
    if r is None:
        return {"routable": False, "reaction": None, "route": None,
                "note": "no common named disconnection found; treat as building block"}
    route_str, name = r
    return {"routable": True, "reaction": name, "route": route_str,
            "note": f"one-step {name} from two simpler reactants"}
