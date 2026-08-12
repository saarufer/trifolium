"""
Build a k-mer index from UniProt human Swiss-Prot FASTA for fast protein identification.

Usage:
    python build_uniprot_index.py [human_swissprot.fasta] [uniprot_kmer_index.pkl]

Output: uniprot_kmer_index.pkl containing:
    {
        "kmer_size": 7,
        "entries": {uniprot_id: {"gene": str, "desc": str, "seq_len": int}},
        "kmer_to_ids": {kmer_str: [uniprot_id, ...]}
    }
"""
from __future__ import annotations
import sys
import pickle
import re
from collections import defaultdict
from pathlib import Path

KMER_SIZE = 6
MIN_SEQ_LEN = 80  # skip fragments shorter than this


def parse_fasta(path: Path):
    """Yield (uniprot_id, gene_name, description, sequence) from UniProt FASTA."""
    uid, gene, desc, seq_parts = None, "", "", []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if uid and seq_parts:
                    yield uid, gene, desc, "".join(seq_parts)
                # >sp|P12345|GENE_HUMAN Description OS=...
                parts = line[1:].split("|")
                uid = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                rest = parts[2] if len(parts) >= 3 else ""
                gene_match = re.search(r'^(\S+)_HUMAN', rest)
                gene = gene_match.group(1) if gene_match else ""
                desc_match = re.match(r'[^\s]+\s+(.*?)\s+OS=', rest)
                desc = desc_match.group(1) if desc_match else rest[:80]
                seq_parts = []
            else:
                seq_parts.append(line)
    if uid and seq_parts:
        yield uid, gene, desc, "".join(seq_parts)


def build_index(fasta_path: Path, out_path: Path, kmer_size: int = KMER_SIZE):
    print(f"Parsing {fasta_path} ...")
    entries = {}
    kmer_to_ids: dict[str, list[str]] = defaultdict(list)

    count = 0
    for uid, gene, desc, seq in parse_fasta(fasta_path):
        if len(seq) < MIN_SEQ_LEN:
            continue
        entries[uid] = {"gene": gene, "desc": desc[:100], "seq_len": len(seq)}
        # Index every k-mer in the sequence
        seen = set()
        for i in range(len(seq) - kmer_size + 1):
            kmer = seq[i:i + kmer_size]
            if kmer not in seen:
                kmer_to_ids[kmer].append(uid)
                seen.add(kmer)
        count += 1
        if count % 5000 == 0:
            print(f"  {count} proteins indexed...")

    print(f"Indexed {count} proteins, {len(kmer_to_ids)} unique {kmer_size}-mers")

    # Remove k-mers that appear in >500 proteins (too common, not discriminative)
    before = len(kmer_to_ids)
    kmer_to_ids = {k: v for k, v in kmer_to_ids.items() if len(v) <= 50}
    print(f"Filtered {before - len(kmer_to_ids)} overly-common k-mers, {len(kmer_to_ids)} remaining")

    index = {
        "kmer_size": kmer_size,
        "entries": entries,
        "kmer_to_ids": dict(kmer_to_ids),
    }
    with open(out_path, "wb") as f:
        pickle.dump(index, f, protocol=4)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved index → {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    fasta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("human_swissprot.fasta")
    out   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("uniprot_kmer_index.pkl")
    build_index(fasta, out)
