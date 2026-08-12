"""Prep-time builder for the offline pocket DB (runs on YOUR machine, not Docker).

For each drug-target UniProt: find homologous holo structures (drug-like bound
ligand), store each holo's {named CA coords, ligand centroid}. At runtime,
pocket_finder aligns each ref onto the server structure (Kabsch) and takes the
consensus ligand centroid — validated ~1-3 A.

Speed: fetches DIRECTLY from RCSB/UniProt and processes targets in PARALLEL
threads (network-bound). The full 1819-target build takes ~1-2 h on 16 workers.

The list of target UniProt accessions is read from `target_uniprots.txt`
(shipped next to this script, one accession per line). Override with --uniprots.

Usage:
    python3 build_pocket_db.py --out ../submission_fornax/Data/pocket_db.pkl --workers 16
    python3 build_pocket_db.py --uniprots P29597,P24941 --out pocket_db.pkl

Dependency: this script imports the pocket-parsing helpers from a submission's
`pocket_finder.py`. Point --pocket-finder at one of the snapshot Code dirs (both
snapshots share the same pocket_finder). By default it looks for
`../submission_fornax/Code`.
"""
from __future__ import annotations
import argparse
import json
import logging
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("build_pocket_db")


def _load_pocket_finder(code_dir: Path):
    """Import the ligand/PDB helpers from a submission's flat pocket_finder.py."""
    sys.path.insert(0, str(code_dir))
    from pocket_finder import (  # noqa: E402
        _parse_pdb_named, _MIN_LIGAND_ATOMS, _AA3, _LIGAND_JUNK,
    )
    return _parse_pdb_named, _MIN_LIGAND_ATOMS, _AA3, _LIGAND_JUNK


def _fetch(url: str, post_body: str | None = None, timeout: int = 30) -> str | None:
    try:
        import requests
        if post_body is not None:
            r = requests.post(url, data=post_body,
                              headers={"Content-Type": "application/json"}, timeout=timeout)
        else:
            r = requests.get(url, timeout=timeout)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def _uniprot_sequence(acc: str) -> str:
    txt = _fetch(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta")
    if not txt:
        return ""
    return "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))


def _seq_search(seq: str, identity: float = 0.5, rows: int = 10) -> list[str]:
    if not seq or len(seq) < 30:
        return []
    q = {"query": {"type": "terminal", "service": "sequence",
                   "parameters": {"evalue_cutoff": 1, "identity_cutoff": identity,
                                  "sequence_type": "protein", "value": seq}},
         "return_type": "polymer_entity",
         "request_options": {"paginate": {"start": 0, "rows": rows}}}
    txt = _fetch("https://search.rcsb.org/rcsbsearch/v2/query",
                 post_body=json.dumps(q), timeout=40)
    if not txt:
        return []
    try:
        d = json.loads(txt)
        return [x["identifier"].split("_")[0] for x in d.get("result_set", [])]
    except Exception:
        return []


def _chain_seq(chain_named: dict, aa3: dict) -> str:
    return "".join(aa3.get(v[0], "X") for _, v in sorted(chain_named.items()))


def _is_homolog_chain(chain_named: dict, ref_seq: str, aa3: dict, thresh: float = 0.4) -> bool:
    """True if the chain is the SAME protein as `ref_seq` (by 6-mer overlap). Rejects
    contaminant chains (e.g. a kinase chain in a complex when building a GR entry)."""
    cs = _chain_seq(chain_named, aa3)
    if len(cs) < 30 or len(ref_seq) < 30:
        return False
    refk = {ref_seq[i:i + 6] for i in range(len(ref_seq) - 6)}
    cks = [cs[i:i + 6] for i in range(len(cs) - 6)]
    return bool(cks) and sum(1 for k in cks if k in refk) / len(cks) >= thresh


def build_entry(uniprot: str, helpers, max_structures: int = 6) -> dict | None:
    """Store up to `max_structures` holo references for a UniProt. Each ref keeps the
    NAMED CA of the homolog chain (resnum -> [resname, [x,y,z]]) + the ligand centroid.
    The ligand is picked ONLY from a chain that sequence-matches `uniprot` (no
    cross-chain ATP/ADP contamination), is drug-like (>= _MIN_LIGAND_ATOMS, not in
    _LIGAND_JUNK), and is BURIED (>=15 CA contacts within 12 A — a surface glycan/
    buffer is rejected). This is what prevents the Thrombin=NAG class of wrong pockets.
    """
    import numpy as np
    _parse_pdb_named, _MIN_LIGAND_ATOMS, _AA3, _LIGAND_JUNK = helpers
    seq = _uniprot_sequence(uniprot)
    if not seq:
        return None
    # Scan MANY structures (not just the top sequence hits): for well-studied targets the
    # top hits are often apo / glycosylated (NAG) / Gla-domain (CGU) — the inhibitor-bound
    # structures are further down. Scan deep, keep the first `max_structures` with a real
    # buried drug-like ligand.
    pdb_ids = _seq_search(seq, identity=0.5, rows=80)
    if not pdb_ids:
        return None
    refs = []
    for pid in pdb_ids:
        if len(refs) >= max_structures:
            break
        txt = _fetch(f"https://files.rcsb.org/download/{pid}.pdb")
        if not txt:
            continue
        ca_named, ligs = _parse_pdb_named(txt)
        if not ligs:
            continue
        homolog = {ch for ch, cn in ca_named.items() if _is_homolog_chain(cn, seq, _AA3)}
        cand = []
        for (ch, nm), a in ligs.items():
            if ch not in homolog or len(a) < _MIN_LIGAND_ATOMS or nm.upper() in _LIGAND_JUNK:
                continue
            cen = np.asarray(a).mean(0)
            contacts = sum(1 for v in ca_named[ch].values()
                           if np.linalg.norm(np.asarray(v[1]) - cen) < 12.0)
            if contacts < 15:                       # surface ligand → not a real pocket
                continue
            cand.append(((ch, nm), a))
        if not cand:
            continue
        (chain, name), atoms = max(cand, key=lambda kv: len(kv[1]))
        cn = ca_named[chain]
        if len(cn) < 30:
            continue
        refs.append({"pdb": pid, "ligand_name": name,
                     "ref_ca": {str(k): [v[0], [float(x) for x in v[1]]] for k, v in cn.items()},
                     "ligand": [float(x) for x in np.asarray(atoms).mean(0)]})
    if not refs:
        return None
    return {"refs": refs}


def _read_uniprot_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniprots", default="", help="comma-separated accessions (overrides the txt list)")
    ap.add_argument("--list", default=str(here / "target_uniprots.txt"),
                    help="file with one UniProt accession per line")
    ap.add_argument("--pocket-finder", default=str(here.parent / "submission_fornax" / "Code"),
                    help="dir containing a submission's pocket_finder.py")
    ap.add_argument("--out", default="pocket_db.pkl")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    helpers = _load_pocket_finder(Path(args.pocket_finder))

    if args.uniprots:
        accs = [a.strip() for a in args.uniprots.split(",") if a.strip()]
    else:
        accs = _read_uniprot_list(Path(args.list))
    accs = list(dict.fromkeys(accs))
    if not accs:
        log.error("No UniProt accessions — provide --uniprots or a non-empty --list")
        sys.exit(1)

    out = Path(args.out)
    db = pickle.load(open(out, "rb")) if out.exists() else {}
    todo = [a for a in accs if a not in db]
    log.info("Pocket DB: %d targets, %d already done, %d to build (%d workers)",
             len(accs), len(db), len(todo), args.workers)

    lock = threading.Lock()
    t0 = time.time()
    done = [0]

    def work(acc):
        try:
            return acc, build_entry(acc, helpers)
        except Exception as ex:
            log.debug("  %s: %s", acc, str(ex)[:60])
            return acc, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, a) for a in todo]
        for fut in as_completed(futures):
            acc, entry = fut.result()
            with lock:
                if entry:
                    db[acc] = entry
                done[0] += 1
                if done[0] % 25 == 0:
                    pickle.dump(db, open(out, "wb"))
                    rate = done[0] / max(time.time() - t0, 1)
                    eta = (len(todo) - done[0]) / max(rate, 0.01) / 60
                    log.info("  %d/%d done, %d hits, %.1f/s, ETA %.0f min",
                             done[0], len(todo), len(db), rate, eta)

    pickle.dump(db, open(out, "wb"))
    size_mb = out.stat().st_size / 1024 / 1024
    log.info("Wrote %d entries → %s (%.1f MB), %.0fs total",
             len(db), out, size_mb, time.time() - t0)


if __name__ == "__main__":
    main()
