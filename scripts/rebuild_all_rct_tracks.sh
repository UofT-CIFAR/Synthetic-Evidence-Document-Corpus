#!/usr/bin/env bash
#
# Wipe SROIE + CORD + FindIt2 RCT outputs and rebuild from scratch (tier batches
# + clean controls). Export API keys in the environment before running, e.g.:
#   export OPENAI_API_KEY=... GOOGLE_API_KEY=... IDEOGRAM_API_KEY=...
#   export COMFYUI_URL=http://127.0.0.1:8188   # variant D
#
# Usage:
#   ./scripts/rebuild_all_rct_tracks.sh
#   DRY_RUN=1 ./scripts/rebuild_all_rct_tracks.sh    # print steps only
#   SKIP_PHASE0=1 ./scripts/rebuild_all_rct_tracks.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN="${DRY_RUN:-0}"
SKIP_PHASE0="${SKIP_PHASE0:-0}"

run() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

echo "==> Synthetic Evidence Corpus: full RCT rebuild (SROIE + CORD + FindIt2)"
echo "    project: $ROOT"

# --- backup manifest ---
STAMP="$(date +%Y%m%d%H%M%S)"
run cp "$ROOT/manifest/manifest.parquet" "$ROOT/manifest/manifest.parquet.bak.$STAMP"
echo "    manifest backup: manifest/manifest.parquet.bak.$STAMP"

# --- remove corpus outputs (brace expand outside quotes) ---
run rm -rf \
  "$ROOT/corpus/TRN/RCT"/TRN-RCT-T{1,2,3,4}-{A,B,C,D} \
  "$ROOT/corpus/TST/RCT"/TST-RCT-T{1,2,3,4}-{A,B,C,D} \
  "$ROOT/corpus/TRN/RCT"/TRN-RCT-T{1,2,3,4}-{A,B,C,D}-CORD \
  "$ROOT/corpus/TST/RCT"/TST-RCT-T{1,2,3,4}-{A,B,C,D}-CORD \
  "$ROOT/corpus/TRN/RCT"/TRN-RCT-T{1,2,3,4}-{A,B,C,D}-FIN \
  "$ROOT/corpus/TST/RCT"/TST-RCT-T{1,2,3,4}-{A,B,C,D}-FIN \
  "$ROOT/corpus/TRN/RCT/TRN-RCT-CLEAN" \
  "$ROOT/corpus/TST/RCT/TST-RCT-CLEAN" \
  "$ROOT/corpus/TRN/RCT/TRN-RCT-CLEAN-CORD" \
  "$ROOT/corpus/TST/RCT/TST-RCT-CLEAN-CORD" \
  "$ROOT/corpus/TRN/RCT/TRN-RCT-CLEAN-FIN" \
  "$ROOT/corpus/TST/RCT/TST-RCT-CLEAN-FIN" \
  "$ROOT/corpus/TRN/RCT/__clean__" \
  "$ROOT/corpus/TST/RCT/__clean__" \
  "$ROOT/corpus/TRN/RCT/__clean__CORD" \
  "$ROOT/corpus/TST/RCT/__clean__CORD" \
  "$ROOT/corpus/TRN/RCT/__clean__FIN" \
  "$ROOT/corpus/TST/RCT/__clean__FIN"

# --- remove per-batch logs / prompts / QA (Python: match batch_id stems) ---
if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] prune logs/prompts_log/qa for RCT batch files"
else
  python3 - <<'PY'
import re
import sys
from pathlib import Path

root = Path(".").resolve()
pat = re.compile(
    r"^(TRN|TST)-RCT-(T[1-4]-[ABCD]|CLEAN(-CORD|-FIN)?)(\.[a-z]+)?$",
    re.IGNORECASE,
)
for sub in ("logs", "prompts_log", "qa"):
    d = root / sub
    if not d.is_dir():
        continue
    for p in d.iterdir():
        if not p.is_file():
            continue
        if pat.search(p.name):
            p.unlink()
            print("removed", p)
PY
fi

# --- drop those batch_ids from manifest.parquet ---
if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] filter manifest/manifest.parquet"
else
  python3 - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sec.batch_registry import cord_batches, findit2_batches, sroie_batches

drop = {b.batch_id for b in sroie_batches()}
drop |= {b.batch_id for b in cord_batches()}
drop |= {b.batch_id for b in findit2_batches()}
drop |= {
    "TRN-RCT-CLEAN",
    "TST-RCT-CLEAN",
    "TRN-RCT-CLEAN-CORD",
    "TST-RCT-CLEAN-CORD",
    "TRN-RCT-CLEAN-FIN",
    "TST-RCT-CLEAN-FIN",
}
mp = Path("manifest/manifest.parquet")
t = pq.read_table(mp)
n0 = t.num_rows
kept = t.filter(pc.invert(pc.is_in(t["batch_id"], value_set=pa.array(sorted(drop)))))
pq.write_table(kept, mp)
print(f"manifest rows: {n0} -> {kept.num_rows} (dropped {n0 - kept.num_rows})")
PY
fi

# --- phase-0 pool sidecars (and SROIE style pools skipped: split only) ---
if [ "$SKIP_PHASE0" = "1" ]; then
  echo "==> skipping phase-0 (SKIP_PHASE0=1); using existing pool_split_*.yaml"
else
  echo "==> phase-0: pool splits (+ CORD/FindIt materialisation as needed)"
  run python3 -m scripts.phase0_setup --skip-pools --only-pool-split
  run python3 -m scripts.phase0_setup_cord
  run python3 -m scripts.phase0_setup_findit2
fi

# --- clean controls (spec counts: 250 TRN / 100 TST per source) ---
echo "==> clean controls"
run python3 -m scripts.build_clean_controls --pool TRN --n 250 --source sroie
run python3 -m scripts.build_clean_controls --pool TST --n 100 --source sroie
run python3 -m scripts.build_clean_controls --pool TRN --n 250 --source cord
run python3 -m scripts.build_clean_controls --pool TST --n 100 --source cord
run python3 -m scripts.build_clean_controls --pool TRN --n 250 --source findit
run python3 -m scripts.build_clean_controls --pool TST --n 100 --source findit

# --- all tier batches (32 each); sequential to reduce API / disk contention ---
echo "==> run_all: SROIE (32) -> CORD (32) -> FindIt2 (32)"
run python3 -m scripts.run_all_sroie
run python3 -m scripts.run_all_cord
run python3 -m scripts.run_all_findit2

echo "==> done. Next: python3 -m scripts.validate_manifest && python3 -m scripts.provenance_audit"
