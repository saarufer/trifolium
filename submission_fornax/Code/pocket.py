"""Offline binding-pocket box resolver (no network).

Defense-in-depth, all offline:
  1. UniProt ID:  extract the protein sequence from the PDB, identify the UniProt
     via a baked k-mer index (protein_id.py + uniprot_kmer_index.pkl).
  2. offline pocket DB:  if the UniProt is in the baked pocket DB (pocket_db.pkl,
     cleaned v2), align its homologous holo references onto the target and take
     the consensus ligand centroid — the validated ~0.9 A method.
  3. bound-ligand centroid:  if the target PDB itself has a real bound ligand
     (holo), use its centroid (skipping crystallographic junk: ions, buffers,
     sugars, modified residues — the Thrombin=NAG trap).
  4. geometric cavity:  for apo / DB-miss targets, find the largest BURIED cavity
     by a grid scan (purely geometric, needs no ID or network).
  5. protein centroid:  last-resort fallback.

Returns a box dict {cx,cy,cz,sx,sy,sz} plus the source label for the log.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# reuse the validated junk list + offline-DB resolver from the ported skill
from pocket_finder import _LIGAND_JUNK, _resolve_offline_entry

_AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
        "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V"}

_POCKET_DB = None
_KMER_READY = False


def _load_pocket_db():
    global _POCKET_DB
    if _POCKET_DB is None:
        import pickle
        for p in (Path("/app/Data/pocket_db.pkl"),
                  Path(__file__).resolve().parent.parent / "Data" / "pocket_db.pkl"):
            if p.exists():
                try:
                    _POCKET_DB = pickle.load(open(p, "rb"))
                except Exception:
                    _POCKET_DB = {}
                break
        if _POCKET_DB is None:
            _POCKET_DB = {}
    return _POCKET_DB


def _sequences(pdb_path: Path):
    """{chain: one-letter seq} from CA atoms."""
    seq = {}
    for ln in pdb_path.read_text(errors="ignore").splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            ch = ln[21]
            res = ln[17:20].strip()
            seq.setdefault(ch, []).append(_AA3.get(res, "X"))
    return {k: "".join(v) for k, v in seq.items()}


def _identify_uniprot(pdb_path: Path) -> str:
    """Offline UniProt identification via the baked k-mer index. '' on miss OR on
    LOW CONFIDENCE. A weak k-mer match (few shared k-mers, or top-1 barely ahead of
    top-2) means we are NOT sure which protein this is — trusting it would put the
    docking box on the WRONG pocket and tank that target's binding (the classic
    'wrong-pocket → bottom-of-leaderboard' failure). When unsure, return '' and let
    resolve_box fall through to the more reliable bound-ligand / geometric methods."""
    try:
        import protein_id
        protein_id._INDEX_CANDIDATES = [
            Path("/app/Data/uniprot_kmer_index.pkl"),
            Path(__file__).resolve().parent.parent / "Data" / "uniprot_kmer_index.pkl"]
        protein_id._INDEX_CACHE = None
        seqs = _sequences(pdb_path)
        if not seqs:
            return ""
        # need top-2 to judge how decisively top-1 wins
        hits = protein_id.identify_protein(seqs, top_n=2)
        if not (hits and hits[0].get("uniprot_id")):
            return ""
        top = hits[0]
        score = top.get("score", 0)
        ratio = top.get("ratio_vs_second", 0)  # top1/top2; large = decisive, ~1 = ambiguous
        # Confidence gate: require a real number of shared k-mers AND a clear lead.
        # A correct match shares many k-mers and dominates top-2 (TYK2: score>>1000,
        # ratio large). Tunable via env for safety.
        import os
        min_score = int(os.environ.get("UNIPROT_MIN_SCORE", "30"))
        min_ratio = float(os.environ.get("UNIPROT_MIN_RATIO", "1.3"))
        if score >= min_score and ratio >= min_ratio:
            return top["uniprot_id"]
    except Exception:
        pass
    return ""


def _atoms(pdb_path: Path):
    """Return (het_centroid_atoms, protein_atoms) as Nx3 arrays. HETATM junk
    (ions/buffers/sugars/modres) is excluded so the box never lands on a
    crystallographic artifact."""
    het, prot = [], []
    for ln in pdb_path.read_text(errors="ignore").splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            resn = ln[17:20].strip()
            try:
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            if ln.startswith("HETATM") and resn.upper() not in _LIGAND_JUNK:
                het.append((x, y, z))
            elif ln.startswith("ATOM"):
                prot.append((x, y, z))
    return np.array(het) if het else np.zeros((0, 3)), \
           np.array(prot) if prot else np.zeros((0, 3))


def _geometric_cavity(prot: np.ndarray):
    """Largest buried cavity by a coarse grid scan. A grid point is a cavity
    candidate if it is empty (no protein atom within ~2.5 A) yet enclosed (many
    protein atoms within ~8 A in several directions). Returns the centroid of the
    most-enclosed empty region, or None."""
    if len(prot) < 50:
        return None
    lo = prot.min(axis=0) + 3.0
    hi = prot.max(axis=0) - 3.0
    if np.any(hi <= lo):
        return None
    step = 2.0
    gx = np.arange(lo[0], hi[0], step)
    gy = np.arange(lo[1], hi[1], step)
    gz = np.arange(lo[2], hi[2], step)
    # cap grid size for runtime
    if len(gx) * len(gy) * len(gz) > 60000:
        step = 2.5
        gx = np.arange(lo[0], hi[0], step)
        gy = np.arange(lo[1], hi[1], step)
        gz = np.arange(lo[2], hi[2], step)
    best, best_score = None, -1
    P = prot
    for x in gx:
        for y in gy:
            for z in gz:
                d2 = ((P - (x, y, z)) ** 2).sum(axis=1)
                near = (d2 < 2.5 ** 2).sum()
                if near > 0:                       # occupied by protein -> not a cavity
                    continue
                shell = (d2 < 8.0 ** 2).sum()      # enclosure = atoms in 8 A shell
                if shell > best_score:
                    best_score = shell
                    best = (float(x), float(y), float(z))
    # require real burial (an exposed empty grid point has few shell atoms)
    return best if best_score >= 40 else None


def resolve_box(pdb_path: str):
    """Return (box_dict, source). box_dict = {cx,cy,cz,sx,sy,sz}."""
    pdb = Path(pdb_path)
    het, prot = _atoms(pdb)

    # 1+2. UniProt -> offline pocket DB (the strongest, validated ~0.9 A)
    uni = _identify_uniprot(pdb)
    if uni:
        db = _load_pocket_db()
        if uni in db:
            try:
                c = _resolve_offline_entry(db[uni], str(pdb))
                if c:
                    return _box(c, 20.0), f"offline_db:{uni}"
            except Exception:
                pass

    # 3. bound ligand centroid (holo, junk-filtered)
    if len(het) >= 4:
        c = tuple(het.mean(axis=0))
        return _box(c, 20.0), "ligand_centroid"

    # 4. geometric cavity (apo / DB miss)
    c = _geometric_cavity(prot)
    if c:
        return _box(c, 22.0), "geometric_cavity"

    # 5. protein centroid (last resort)
    if len(prot):
        return _box(tuple(prot.mean(axis=0)), 26.0), "protein_centroid"
    return _box((0.0, 0.0, 0.0), 26.0), "none"


def candidate_boxes(pdb_path: str):
    """Return ALL plausible docking boxes (not just the first that hits), as a list of
    (box, source). The caller can then dock probe molecules in each and KEEP THE ONE THAT
    PRODUCES THE DEEPEST VINA — letting Vina itself vote on the pocket, instead of betting
    that UniProt/ligand identification picked the right cavity. This directly attacks the
    'wrong box on an unknown target -> deep Vina at the wrong place' failure (T2/T3).
    Aligned with the platform's pure-Vina objective: we pick the box where the score is
    deepest, which is exactly what the platform rewards."""
    pdb = Path(pdb_path)
    het, prot = _atoms(pdb)
    out = []
    # UniProt -> offline DB pocket (strongest if ID is confident)
    uni = _identify_uniprot(pdb)
    if uni:
        db = _load_pocket_db()
        if uni in db:
            try:
                c = _resolve_offline_entry(db[uni], str(pdb))
                if c:
                    out.append((_box(c, 20.0), f"offline_db:{uni}"))
            except Exception:
                pass
    # bound ligand centroid (holo)
    if len(het) >= 4:
        out.append((_box(tuple(het.mean(axis=0)), 20.0), "ligand_centroid"))
    # geometric cavity (apo / DB miss)
    c = _geometric_cavity(prot)
    if c:
        out.append((_box(c, 22.0), "geometric_cavity"))
    # protein centroid (last resort)
    if len(prot):
        out.append((_box(tuple(prot.mean(axis=0)), 26.0), "protein_centroid"))
    if not out:
        out.append((_box((0.0, 0.0, 0.0), 26.0), "none"))
    return out


def _box(center, size):
    return dict(cx=float(center[0]), cy=float(center[1]), cz=float(center[2]),
                sx=size, sy=size, sz=size)
