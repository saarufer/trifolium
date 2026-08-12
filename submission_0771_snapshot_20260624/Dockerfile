# Task2 autonomous design agent — self-contained, NO LLM, offline.
# AutoDock Vina (Linux x86_64) is baked in and called via subprocess; Open Babel
# handles receptor/ligand PDBQT conversion. Reads 3 target PDBs, designs one
# deep-Vina + synthesizable molecule per target, writes result{1,2,3}.csv.
FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openbabel \
    && rm -rf /var/lib/apt/lists/*

# rdkit + numpy for chemistry/scoring. anthropic + httpx provide the OPTIONAL LLM
# strategy layer (this submission runs pure-rule, strategist=None — the key is kept
# only for interface completeness; the GA loop designs & evolves without it).
# anthropic 0.109.1 + httpx 0.28.1: the combo verified to coexist (0.39 broke on the
# removed 'proxies' kwarg).
RUN pip install --no-cache-dir rdkit==2024.3.2 numpy==1.26.4 anthropic==0.109.1 httpx==0.28.1

WORKDIR /app

# agent code
COPY Code/ /app/Code/

# Offline pocket assets (baked in, no network):
#   pocket_db.pkl          cleaned v2 pocket DB (UniProt -> homologous holo refs)
#   uniprot_kmer_index.pkl k-mer index for offline UniProt identification
COPY Data/ /app/Data/

# Vina binary (Linux x86_64, statically linked) baked into the image
COPY vina /app/vina

# Entry script MUST live at /app/run.sh — the platform convention (fixed name
# /app/run.sh; a previous /app/Code/run.sh entrypoint failed with exit 127).
COPY Code/run.sh /app/run.sh
RUN chmod +x /app/vina /app/run.sh

RUN mkdir -p /app/submission /app/work /saisresult

ENV SAIS_DATA_DIR=/saisdata
ENV SAIS_RESULT_DIR=/saisresult
ENV APP_DIR=/app
ENV TARGET_BUDGET_SEC=3600
ENV VINA_EXHAUSTIVENESS=8

# Optional LLM strategy layer via the Singapore ECS proxy. This submission runs
# pure-rule (strategist=None) and does NOT call the API; the key is injected at
# build time only to satisfy the "must include APIKEY" submission requirement:
#   docker buildx build --build-arg ANTHROPIC_API_KEY=$MY_KEY ...
ENV ANTHROPIC_BASE_URL=http://43.98.197.239
ARG ANTHROPIC_API_KEY
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

ENTRYPOINT ["/app/run.sh"]
