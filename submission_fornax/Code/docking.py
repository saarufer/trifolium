"""In-image Vina docking backend (subprocess) — for the self-contained task2
submission image. Unlike the development backend, this does NOT use Docker: the
Linux Vina binary is baked into the image at /app/vina and called directly via
subprocess, because the platform runs this image inside a container where
Docker-in-Docker is not available.

Same public interface as the dev backend so agent.py is unchanged:
  backend = VinaBackend(receptor_pdb, work_dir)
  score   = backend.dock(smiles)
  backend.dock_consensus(smiles, n=3)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

EXH = int(os.environ.get("VINA_EXHAUSTIVENESS", "8"))
# Two-stage exhaustiveness: every round docks at EXH (fast, broad). The champion is
# RE-DOCKED at HIGH_EXH (slower, more thorough) to confirm its depth — a real deep hit
# stays deep at higher exhaustiveness; a false-positive (lucky single-pose at EXH=8)
# regresses. Cheaper + more discriminative than n-times consensus at the same EXH.
# HIGH_EXH scales with cores: a beefy server (>=8 cores) earns exh24; else exh16.
def _default_high_exh():
    try:
        return 24 if (os.cpu_count() or 1) >= 8 else 16
    except Exception:
        return 16
HIGH_EXH = int(os.environ.get("VINA_HIGH_EXHAUSTIVENESS", "0")) or _default_high_exh()

_VINA_CANDIDATES = [
    Path("/app/vina"),
    Path(os.environ.get("VINA_BIN", "")) if os.environ.get("VINA_BIN") else None,
    Path(__file__).resolve().parent.parent / "vina",
    Path("/usr/local/bin/vina"),
]


def _find_vina():
    for p in _VINA_CANDIDATES:
        if p and p.exists():
            return str(p)
    w = shutil.which("vina")
    return w


def _find_obabel():
    return shutil.which("obabel") or "/usr/bin/obabel"


class VinaBackend:
    def __init__(self, receptor_pdb: str, work_dir: str):
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.vina = _find_vina()
        self.obabel = _find_obabel()
        self.ok = self.vina is not None
        self.box = None
        self.receptor_pdbqt = None
        if self.ok:
            try:
                self._prepare_receptor(Path(receptor_pdb))
            except Exception:
                self.ok = False

    # ── receptor + pocket box ────────────────────────────────────────────
    def _prepare_receptor(self, pdb: Path):
        rec_pdb = self.work / "receptor.pdb"
        shutil.copy(pdb, rec_pdb)
        out = self.work / "receptor.pdbqt"
        subprocess.run([self.obabel, str(rec_pdb), "-xr", "-O", str(out)],
                       capture_output=True, timeout=120)
        self.receptor_pdbqt = str(out)
        # Single offline pocket resolution (multi-pocket was tested and abandoned —
        # decoy boxes won by Vina noise, would replace the true pocket).
        import pocket as pocket_mod
        self.box, self.box_source = pocket_mod.resolve_box(str(rec_pdb))

    # ── ligand prep + dock ───────────────────────────────────────────────
    def _ligand_pdbqt(self, smiles: str, tag: str):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = Chem.AddHs(m)
        # EMBED STABILITY GATE: a single fixed-seed embed can succeed by luck on a
        # strained conformer → Vina scores that bad geometry as a false-positive deep
        # hit. Require >=need of `tries` seeds to embed; else reject (anti false-positive).
        tries = int(os.environ.get("EMBED_TRIES", "3"))
        need = int(os.environ.get("EMBED_NEED", "2"))
        oks = 0
        for seed in (0xC0FFEE, 1, 42, 7, 100)[:tries]:
            if AllChem.EmbedMolecule(m, randomSeed=seed) == 0:
                oks += 1
                if oks >= need:
                    break
        if oks < need:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        sdf = self.work / f"lig_{tag}.sdf"
        Chem.SDWriter(str(sdf)).write(m)
        out = self.work / f"lig_{tag}.pdbqt"
        r = subprocess.run([self.obabel, str(sdf), "-O", str(out)],
                           capture_output=True, timeout=60)
        return str(out) if r.returncode == 0 and out.exists() else None

    def dock(self, smiles: str, exh: int = None):
        if not self.ok or self.box is None:
            return None
        e = EXH if exh is None else exh
        tag = uuid.uuid4().hex[:8]
        lig = self._ligand_pdbqt(smiles, tag)
        if lig is None:
            return None
        b = self.box
        out = self.work / f"out_{tag}.pdbqt"
        try:
            r = subprocess.run([
                self.vina, "--receptor", self.receptor_pdbqt, "--ligand", lig,
                "--center_x", str(b["cx"]), "--center_y", str(b["cy"]),
                "--center_z", str(b["cz"]),
                "--size_x", str(b["sx"]), "--size_y", str(b["sy"]),
                "--size_z", str(b["sz"]),
                "--exhaustiveness", str(e), "--cpu", os.environ.get("VINA_CPU", "1"),
                "--out", str(out),
            ], capture_output=True, text=True, timeout=240 if e <= 8 else 480)
            score = self._parse_score(r.stdout)
        except Exception:
            score = None
        for f in self.work.glob(f"*_{tag}.*"):
            try:
                f.unlink()
            except Exception:
                pass
        return score

    def dock_consensus(self, smiles: str, n: int = 3):
        vals = [v for v in (self.dock(smiles) for _ in range(n)) if v is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    @staticmethod
    def _parse_score(stdout: str):
        for line in (stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    return float(parts[1])
                except ValueError:
                    continue
        return None
