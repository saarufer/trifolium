"""Skill: pocket_finder — locate the real binding pocket of a target structure.

Validated 2026-06-16: the geometric centroid is the wrong box (Vina↔official
binding corr ~0.22). The right pocket lifts it to ~0.7. Blind docking arbitrary
actives is noisy (~7 A off). The robust method is to align HOMOLOGOUS holo
structures (with bound ligands) onto the target and take the consensus ligand
centroid — validated to 0.9 A on TYK2.

Resolution order (defense in depth):
  1. offline pocket DB  (baked {uniprot -> pocket center}, robust, no network)
  2. ECS consensus      (RCSB sequence search + fetch + Kabsch align + cluster)
  3. blind docking      (dock probe actives in a whole-protein box; noisy)
  4. geometric centroid (last resort)

All network goes through our ECS data gateway (ANTHROPIC_BASE_URL/data?url=...);
every step degrades gracefully so a target/run never crashes.
"""
from __future__ import annotations
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# Crystallographic artifacts — NEVER the active-site binder we want. Picking one as the
# pocket reference puts the box on a glycosylation site / surface / Gla domain (the Thrombin=NAG
# bug). Covers ions, buffers/solvents, sugars, and post-translationally modified residues.
_LIGAND_JUNK = {
    # ions / buffers / solvents
    "HOH", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "NI", "CU", "CD",
    "SO4", "PO4", "NO3", "GOL", "EDO", "PEG", "PG4", "PGE", "ACT", "DMS", "P6G", "7PE",
    "TRS", "MES", "EPE", "FMT", "CIT", "IOD", "BR", "MPD", "BME", "IMD", "1PE", "2PE", "12P",
    "ACY", "CO3", "NH4", "FLC", "BCT", "TLA", "MLI", "UNX", "UNL", "OH", "MOH", "TAR", "NHE",
    # glycosylation sugars (surface, not active site)
    "NAG", "MAN", "BMA", "GLC", "GAL", "FUC", "NDG", "BGC", "FUL", "A2G", "SIA", "XYS",
    "RIB", "GLA", "NGA", "MAL", "LMT", "SGN", "GCU",
    # post-translationally modified residues (part of the protein chain, not a ligand)
    "MSE", "PTR", "SEP", "TPO", "CSO", "KCX", "LLP", "CME", "OCS", "CSD", "CAS", "MLY",
    "M3L", "ALY", "CGU", "PCA", "MHO", "HYP", "SEC", "CSX",
}
_AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
        'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
        'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
_MIN_LIGAND_ATOMS = 12       # below this a HETATM is a fragment/ion, not a drug
_CLUSTER_RADIUS = 6.0        # A; ligand centroids within this of each other agree
_MIN_ALIGN_RESIDUES = 30     # need this many common CA to trust a superposition


def _ecs_base() -> str:
    return os.environ.get("ANTHROPIC_BASE_URL", "http://43.98.197.239").rstrip("/")


def _ecs_fetch(url: str, method: str = "GET", body: str | None = None,
               timeout: int = 25) -> str | None:
    """Fetch a whitelisted URL through the ECS /data gateway. None on failure.

    Uses stdlib urllib (NOT requests) — the runtime image installs only offline wheels
    and has no requests, so any requests import here would make the whole ECS pocket path
    silently fail → geometric-centroid box (bad). The target URL is passed url-encoded as
    the ?url= value so its own '?&=' don't corrupt the gateway query."""
    import urllib.request
    import urllib.parse
    gw = f"{_ecs_base()}/data?url={urllib.parse.quote(url, safe='')}"
    try:
        data = body.encode("utf-8") if body else None
        headers = {"Content-Type": "application/json"} if method == "POST" else {}
        req = urllib.request.Request(gw, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if getattr(r, "status", 200) == 200:
                return r.read().decode("utf-8", "replace")
    except Exception as e:
        log.debug("[pocket] ECS fetch failed: %s", str(e)[:80])
    return None


def _rcsb_sequence_search(seq: str, identity: float = 0.5, rows: int = 8) -> list[str]:
    """RCSB sequence search via ECS → list of PDB IDs (homologous structures)."""
    if not seq or len(seq) < 30:
        return []
    q = {"query": {"type": "terminal", "service": "sequence",
                   "parameters": {"evalue_cutoff": 1, "identity_cutoff": identity,
                                  "sequence_type": "protein", "value": seq}},
         "return_type": "polymer_entity",
         "request_options": {"paginate": {"start": 0, "rows": rows}}}
    txt = _ecs_fetch("https://search.rcsb.org/rcsbsearch/v2/query",
                     method="POST", body=json.dumps(q), timeout=40)
    if not txt:
        return []
    try:
        d = json.loads(txt)
        return [x["identifier"].split("_")[0] for x in d.get("result_set", [])]
    except Exception:
        return []


def _parse_pdb_text(txt: str):
    """Return (ca_by_chain_resnum, ligand_atoms_by_chain_ligname)."""
    ca = defaultdict(dict)          # chain -> resnum -> (x,y,z)
    ligs = defaultdict(list)        # (chain, ligname) -> [(x,y,z), ...]
    for line in txt.splitlines():
        rec = line[:6].strip()
        if rec == "ATOM" and line[12:16].strip() == "CA":
            try:
                ca[line[21]][int(line[22:26])] = (
                    float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except (ValueError, IndexError):
                pass
        elif rec == "HETATM":
            name = line[17:20].strip()
            if name in _LIGAND_JUNK:
                continue
            try:
                ligs[(line[21], name)].append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except (ValueError, IndexError):
                pass
    return ca, ligs


def _kabsch(P, Q):
    """Rotation R + translations so that R@(P-Pm) ~ (Q-Qm). Returns (R, Pm, Qm, rmsd)."""
    import numpy as np
    P = np.asarray(P); Q = np.asarray(Q)
    Pm = P.mean(0); Qm = Q.mean(0)
    Pc = P - Pm; Qc = Q - Qm
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    rmsd = float(np.sqrt((((R @ Pc.T).T - Qc) ** 2).sum(1)).mean())
    return R, Pm, Qm, rmsd


def _align_ligand_centroid(ref_ca, lig_atoms, target_ca):
    """Superpose ref onto target on common CA, transform the ligand centroid."""
    import numpy as np
    common = sorted(set(ref_ca) & set(target_ca))
    if len(common) < _MIN_ALIGN_RESIDUES:
        return None
    P = [ref_ca[r] for r in common]
    Q = [target_ca[r] for r in common]
    R, Pm, Qm, rmsd = _kabsch(P, Q)
    if rmsd > 5.0:                  # poor superposition → distrust
        return None
    lc = np.asarray(lig_atoms).mean(0)
    return tuple(float(x) for x in (R @ (lc - Pm) + Qm))


def _parse_pdb_named(txt: str):
    """Like _parse_pdb_text but CA carries the residue NAME:
    returns (ca_named {chain: {resnum: (resname, (x,y,z))}}, ligs {(chain,name): [xyz]})."""
    ca = defaultdict(dict)
    ligs = defaultdict(list)
    for line in txt.splitlines():
        rec = line[:6].strip()
        if rec == "ATOM" and line[12:16].strip() == "CA":
            try:
                ca[line[21]][int(line[22:26])] = (
                    line[17:20].strip(),
                    (float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except (ValueError, IndexError):
                pass
        elif rec == "HETATM":
            name = line[17:20].strip()
            if name in _LIGAND_JUNK:
                continue
            try:
                ligs[(line[21], name)].append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except (ValueError, IndexError):
                pass
    return ca, ligs


def _best_offset(tgt_named: dict, ref_named: dict) -> tuple[int, int]:
    """Constant resnum offset k (ref resnum = tgt resnum + k) maximizing residue-NAME
    matches. Robust to renumbering AND wrong-chain (a non-homolog chain matches poorly).
    Returns (k, n_matches)."""
    tgt_rn = sorted(tgt_named)
    ref_rn = sorted(ref_named)
    if not tgt_rn or not ref_rn:
        return 0, 0
    best_k, best_n = 0, 0
    for k in range(ref_rn[0] - tgt_rn[-1], ref_rn[-1] - tgt_rn[0] + 1):
        n = sum(1 for r in tgt_rn
                if (r + k) in ref_named and tgt_named[r][0] == ref_named[r + k][0])
        if n > best_n:
            best_n, best_k = n, k
    return best_k, best_n


def _offset_align_centroid(ref_named: dict, tgt_named: dict, lig_atoms: list) -> tuple | None:
    """Superpose ref chain onto target via the best resname-offset, transform the ligand
    centroid. Requires the ref chain to be a real sequence homolog (>=30 name matches),
    which rejects wrong chains/contaminants and survives renumbering."""
    import numpy as np
    k, nmatch = _best_offset(tgt_named, ref_named)
    if nmatch < _MIN_ALIGN_RESIDUES:
        return None
    pairs = [(tgt_named[r][1], ref_named[r + k][1])
             for r in tgt_named
             if (r + k) in ref_named and tgt_named[r][0] == ref_named[r + k][0]]
    if len(pairs) < _MIN_ALIGN_RESIDUES:
        return None
    Q = [p[0] for p in pairs]   # target frame
    P = [p[1] for p in pairs]   # ref frame
    R, Pm, Qm, rmsd = _kabsch(P, Q)
    if rmsd > 5.0:
        return None
    lc = np.asarray(lig_atoms).mean(0)
    return tuple(float(x) for x in (R @ (lc - Pm) + Qm))


def _consensus(points: list[tuple]) -> tuple | None:
    """Largest cluster (within _CLUSTER_RADIUS) of pocket points → its mean."""
    if not points:
        return None
    best_idx, best_members = -1, []
    for i, p in enumerate(points):
        members = [q for q in points if math.dist(p, q) < _CLUSTER_RADIUS]
        if len(members) > len(best_members):
            best_members = members
    n = len(best_members)
    return (sum(m[0] for m in best_members) / n,
            sum(m[1] for m in best_members) / n,
            sum(m[2] for m in best_members) / n)


def find_pocket_consensus(target_named: dict, target_seq: str,
                          max_structures: int = 6) -> tuple | None:
    """ECS path: sequence search → fetch holo → align ligands → consensus pocket.

    target_named: {resnum -> (resname, (x,y,z))} CA of the target structure (one chain).
    Each holo ligand is aligned via the chain it sits in, using a resname-OFFSET
    superposition — this survives renumbered targets AND rejects wrong/contaminant
    chains (a kinase chain won't sequence-match a nuclear-receptor target).
    """
    pdb_ids = _rcsb_sequence_search(target_seq, identity=0.5, rows=max_structures + 6)
    if not pdb_ids:
        log.info("[pocket] RCSB sequence search returned no homologs")
        return None
    centroids = []
    for pid in pdb_ids:
        if len(centroids) >= max_structures:
            break
        txt = _ecs_fetch(f"https://files.rcsb.org/download/{pid}.pdb")
        if not txt:
            continue
        ca_named, ligs = _parse_pdb_named(txt)
        if not ligs:
            continue
        # try drug-like ligands biggest-first; align via the ligand's OWN chain
        for (chain, name), atoms in sorted(ligs.items(), key=lambda kv: -len(kv[1])):
            if len(atoms) < _MIN_LIGAND_ATOMS:
                break
            if name.upper() in _LIGAND_JUNK:        # skip sugars/buffers/modres (Thrombin=NAG bug)
                continue
            ref_chain = ca_named.get(chain) or max(ca_named.values(), key=len, default={})
            c = _offset_align_centroid(ref_chain, target_named, atoms)
            if c:
                centroids.append(c)
                log.debug("[pocket] %s %s -> (%.1f,%.1f,%.1f)", pid, name, *c)
                break
    if not centroids:
        return None
    pc = _consensus(centroids)
    log.info("[pocket] ECS consensus from %d holo ligand(s): (%.1f,%.1f,%.1f)",
             len(centroids), *pc)
    return pc


def find_pocket(target_pdb_path: str, uniprot: str = "", target_seq: str = "",
                probe_smiles: list[str] | None = None,
                offline_db: dict | None = None) -> tuple[tuple | None, str]:
    """Orchestrate pocket finding with full fallback chain.

    Returns (center, source) where source ∈ {offline_db, ecs_consensus,
    blind_dock, geometric_centroid, none}. center is None only if everything
    failed AND the caller wants to skip docking.
    """
    # --- 1. offline pocket DB (baked, robust) ---
    if offline_db and uniprot and uniprot in offline_db:
        entry = offline_db[uniprot]
        # entry may be a ready center or {ref_ca, ligand_centroid} needing alignment
        c = _resolve_offline_entry(entry, target_pdb_path)
        if c:
            log.info("[pocket] offline DB hit for %s: (%.1f,%.1f,%.1f)", uniprot, *c)
            return c, "offline_db"

    # --- target CA(+resnames) + sequence, parsed consistently with the holos ---
    target_named, target_ca = {}, {}
    try:
        txt = Path(target_pdb_path).read_text()
        ca_named_all, _ = _parse_pdb_named(txt)
        if ca_named_all:
            chain = max(ca_named_all.values(), key=len)        # {resnum:(resname,xyz)}
            target_named = chain
            target_ca = {r: v[1] for r, v in chain.items()}    # {resnum:xyz}
            if not target_seq:
                target_seq = "".join(_AA3.get(v[0], "X") for _, v in sorted(chain.items()))
    except Exception as e:
        log.warning("[pocket] target parse failed: %s", e)

    # --- 2. ECS consensus alignment (renumber- and contaminant-robust) ---
    if target_named and target_seq:
        try:
            c = find_pocket_consensus(target_named, target_seq)
            if c:
                return c, "ecs_consensus"
        except Exception as e:
            log.warning("[pocket] ECS consensus error: %s", e)

    # --- 3. blind docking (noisy fallback) ---
    if probe_smiles:
        try:
            from tools.docking import find_pocket_center
            c = find_pocket_center(target_pdb_path, probe_smiles)
            if c:
                return c, "blind_dock"
        except Exception as e:
            log.warning("[pocket] blind dock error: %s", e)

    # --- 4. geometric centroid (last resort) ---
    if target_ca:
        xs = [v[0] for v in target_ca.values()]
        ys = [v[1] for v in target_ca.values()]
        zs = [v[2] for v in target_ca.values()]
        c = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        log.info("[pocket] falling back to geometric centroid (LOW confidence)")
        return c, "geometric_centroid"
    return None, "none"


def _resolve_offline_entry(entry, target_pdb_path: str) -> tuple | None:
    """Offline DB entry → pocket center in the target's frame.

    Entry forms:
      {"center": [x,y,z]}              → used directly (same-frame assumption)
      {"refs": [{ref_ca, ligand}, ...]} → align EACH ref→target, transform its
                                          ligand, cluster → consensus (robust).
    """
    try:
        if "center" in entry:
            return tuple(entry["center"])
        refs = entry.get("refs")
        if not refs:
            return None
        ca_named_all, _ = _parse_pdb_named(Path(target_pdb_path).read_text())
        if not ca_named_all:
            return None
        tgt_named = max(ca_named_all.values(), key=len)        # {resnum:(resname,xyz)}
        tca = {r: v[1] for r, v in tgt_named.items()}          # {resnum:xyz}
        pts = []
        for r in refs:
            sample = next(iter(r["ref_ca"].values()))
            if sample and isinstance(sample[0], str):          # NEW named format: [resname,[x,y,z]]
                ref_named = {int(k): (v[0], tuple(v[1])) for k, v in r["ref_ca"].items()}
                c = _offset_align_centroid(ref_named, tgt_named, [r["ligand"]])
            else:                                              # legacy format: [x,y,z]
                ref_ca = {int(k): tuple(v) for k, v in r["ref_ca"].items()}
                c = _align_ligand_centroid(ref_ca, [r["ligand"]], tca)
            if c:
                pts.append(c)
        if not pts:
            return None
        return _consensus(pts)
    except Exception as e:
        log.debug("[pocket] offline entry resolve failed: %s", e)
    return None
