# Task2 autonomous design agent — Fornax: rule-open → LLM-refine + interval strong-reflection.
# AutoDock Vina (Linux x86_64) baked in, called via subprocess; Open Babel for PDBQT.
# Reads 3 target PDBs, designs one deep-Vina + synthesizable molecule per target.
# LLM strategist runs via the Singapore ECS proxy (ANTHROPIC_BASE_URL); degrades to
# pure rules if the key/proxy is unavailable (rules guarantee the floor).
FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openbabel \
    && rm -rf /var/lib/apt/lists/*

# rdkit + numpy for chemistry; anthropic SDK for the LLM strategist (Fornax).
# anthropic 0.109.1 + httpx 0.28.1: the combo verified in the historical CC2 LLM image.
# (anthropic 0.39 breaks against httpx>=0.28 — the 'proxies' kwarg was removed.)
RUN pip install --no-cache-dir rdkit==2024.3.2 numpy==1.26.4 anthropic==0.109.1 httpx==0.28.1

WORKDIR /app

# agent code
COPY Code/ /app/Code/

# Offline pocket assets (baked in, no network needed for pocket identification)
COPY Data/ /app/Data/

# Vina binary (Linux x86_64, statically linked) baked into the image
COPY vina /app/vina

# Entry script MUST live at /app/run.sh — platform convention (fixed name /app/run.sh).
COPY Code/run.sh /app/run.sh
RUN chmod +x /app/vina /app/run.sh

RUN mkdir -p /app/submission /app/work /saisresult

ENV SAIS_DATA_DIR=/saisdata
ENV SAIS_RESULT_DIR=/saisresult
ENV APP_DIR=/app
# 3 targets × ~73min each (= 3h39m) — leaves ~15min headroom under the 3h55m outer
# timeout for the final packaging, even if a late round runs slightly long.
ENV TARGET_BUDGET_SEC=4400
ENV VINA_EXHAUSTIVENESS=8
# Parallel docking: 1 core per vina; agent uses (cores-1) workers by default.
ENV VINA_CPU=1
# Embed-stability gate: reject difficult-to-embed Vina false positives.
ENV EMBED_TRIES=3
ENV EMBED_NEED=2
# AGGRESSIVE DEPTH CHASE (pure rule, going for max Vina depth):
#   N_RESTART  — independent evolutions per target; keep the globally deepest champion.
#   ANNEAL_K   — ring-fuse after only this many stalls (狂堆稠环冲深).
#   SEED_SCOUT_EXH / SEED_PICK_BUDGET_SEC — fast cheap seed-picking, time-capped.
#   TARGET_TAIL_SEC / MIN_RESTART_SEC — wall-clock guards so a target never overruns.
ENV N_RESTART=3
ENV ANNEAL_K=2
ENV SEED_SCOUT_EXH=4
ENV TARGET_TAIL_SEC=150
ENV MIN_RESTART_SEC=300

# LLM strategist via Singapore ECS proxy. API key injected at build time:
#   docker buildx build --build-arg ANTHROPIC_API_KEY=$MY_KEY ...
ENV ANTHROPIC_BASE_URL=http://43.98.197.239
ARG ANTHROPIC_API_KEY
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

ENTRYPOINT ["/app/run.sh"]
