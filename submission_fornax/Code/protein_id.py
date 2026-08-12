"""Protein identification tool: k-mer index search."""
from __future__ import annotations
import pickle
import logging
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

_INDEX_CANDIDATES = [
    Path("/app/uniprot_kmer_index.pkl"),
    Path(__file__).parent.parent.parent / "Data" / "uniprot_kmer_index.pkl",
]

_INDEX_CACHE: dict | None = None


def _load_index() -> dict | None:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    for path in _INDEX_CANDIDATES:
        if path.exists():
            with open(path, "rb") as f:
                _INDEX_CACHE = pickle.load(f)
            log.info("[protein_id] Loaded k-mer index from %s", path)
            return _INDEX_CACHE
    log.warning("[protein_id] k-mer index not found")
    return None


def identify_protein(sequences: dict[str, str], top_n: int = 5) -> list[dict]:
    """
    Search k-mer index to identify protein from sequences.
    Returns top_n hits sorted by k-mer overlap score.
    Each hit: {uniprot_id, gene, description, score, ratio_vs_second}
    """
    idx = _load_index()
    if idx is None:
        return []

    k = idx["kmer_size"]
    kmer_to_ids = idx["kmer_to_ids"]
    entries = idx["entries"]

    counter: Counter = Counter()
    for seq in sequences.values():
        seen: set[str] = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in kmer_to_ids and kmer not in seen:
                for uid in kmer_to_ids[kmer]:
                    counter[uid] += 1
                seen.add(kmer)

    results = []
    top_list = counter.most_common(top_n + 1)
    for rank, (uid, score) in enumerate(top_list[:top_n]):
        second_score = top_list[1][1] if rank == 0 and len(top_list) > 1 else 0
        ratio = score / max(second_score, 1) if rank == 0 else 0
        e = entries.get(uid, {})
        results.append({
            "uniprot_id": uid,
            "gene": e.get("gene", ""),
            "description": e.get("desc", "")[:100],
            "score": score,
            "ratio_vs_second": round(ratio, 1),
        })

    if results:
        log.info(
            "[protein_id] Top hit: %s (%s) score=%d ratio=%.1fx",
            results[0]["uniprot_id"], results[0]["gene"],
            results[0]["score"], results[0]["ratio_vs_second"],
        )
    return results


# ── Claude tool definition ────────────────────────────────────────────────
TOOL_DEF = {
    "name": "identify_protein",
    "description": (
        "Search the UniProt k-mer index to identify the protein from PDB sequences. "
        "Returns ranked candidate proteins with UniProt IDs, gene names, and confidence scores. "
        "Use this early in the pipeline to determine the target protein identity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sequences": {
                "type": "object",
                "description": "Dict mapping chain ID to amino acid sequence (single-letter codes)",
                "additionalProperties": {"type": "string"},
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top hits to return (default 5)",
                "default": 5,
            },
        },
        "required": ["sequences"],
    },
}
