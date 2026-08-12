#!/bin/bash
set -uo pipefail
echo "[run.sh] task2 autonomous design agent (no-LLM) starting — $(date -u)"

# Outer hard backstop so the process exits before the platform's hard limit even
# if a target's docking wedges. Per-target budget is 1h (3 targets => ~3h);
# default outer timeout ~3h55m.
RUN_TIMEOUT="${RUN_TIMEOUT_SEC:-14100}"

set +e
timeout --signal=TERM "${RUN_TIMEOUT}" python3 -u /app/Code/main.py > /app/run.log 2>&1
status=$?
set -e
echo "[run.sh] main.py exit status: ${status}"
tail -120 /app/run.log || true

# Safety net: if the run was killed before producing a zip, package whatever
# result CSVs exist (or fallbacks) so the platform always receives a submission.
if [ ! -s /saisresult/result.zip ]; then
  echo "[run.sh] result.zip missing — emergency deliver"
  python3 -u /app/Code/main.py --emergency-deliver >> /app/run.log 2>&1 || true
fi

[ -s /saisresult/result.zip ] && echo "[run.sh] result.zip OK" \
  || echo "[run.sh] WARNING: result.zip still missing"
echo "[run.sh] Done — $(date -u)"
