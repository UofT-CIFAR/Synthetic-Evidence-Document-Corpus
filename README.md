# Synthetic Evidence Corpus — SROIE2019 Receipts

This project implements the Synthetic Evidence Corpus specification for the
Receipts (`RCT`) family, scoped to SROIE2019. It produces the 32 SROIE
batches `{TRN,TST}-RCT-T{1,2,3,4}-{A,B,C,D}`: 320 manipulated training items +
160 manipulated test items plus roughly 350 clean controls (with ~10%
re-saved through a tool path per spec §4.6).

## Layout

- `sec/`            : importable Python package (config, manifest, adapters,
                      tier edits, batch runner, QA).
- `configs/`        : `paths.yaml`, `tools.yaml`, ComfyUI workflow JSONs.
- `prompts/`        : Tier-3 / Tier-4 prompt templates.
- `assets/`         : Tier-2 and Tier-3 content banks.
- `style_pools/`    : produced in Phase 0; disjoint train/test signature pools.
- `scripts/`        : CLI entry points.
- `corpus/`         : output artifacts (`<pool>/<family>/<batch_id>/<id>.png`).
- `manifest/`       : `manifest.parquet` (one row per artifact).
- `logs/`, `prompts_log/`, `qa/`, `audit/` : per-batch logs and reports.

## Setup

```bash
cd CIFAR/SyntheticEvidenceCorpus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Populate the credentials for whichever variants you want to run:

```bash
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
export IDEOGRAM_API_KEY=...
export COMFYUI_URL=http://localhost:8188
```

Any adapter whose credentials are absent will raise
`AdapterCredentialError` when a batch tries to use it; the item then **fails**
in the batch runner (no local substitute image).

Style-pool Phase 0 can still use deterministic PIL reference strokes when no
ComfyUI adapter is available (see `populate_pools`); re-run pool generation with
an API when you need provider-generated refs.

## Running the pipeline

```bash
# One-time Phase 0 setup (source verification, pool split sidecar, Tier-2 style
# pools). Pass --skip-pools if you do not have Variant D (ComfyUI) yet and are
# happy with the deterministic PIL fallback references.
python -m scripts.phase0_setup --refs 6

# Clean controls (~250 train, 100 test)
python -m scripts.build_clean_controls --pool TRN --n 250
python -m scripts.build_clean_controls --pool TST --n 100

# A single batch
python -m scripts.run_batch TRN-RCT-T1-A

# All 32 SROIE batches, or a filtered slice
python -m scripts.run_all_sroie
python -m scripts.run_all_sroie --only-tier T1 --only-pool TRN
python -m scripts.run_all_sroie --dry-run

# Post-hoc validators
python -m scripts.validate_manifest
python -m scripts.provenance_audit
```

## Manifest schema

See `sec/schema.py` for the pyarrow schema. Highlights:

- `tier` is `0` for clean controls, `1..4` for manipulated items.
- `edit_regions` is a list of `{page,x,y,w,h,kind,old_text,new_text}`
  structs.
- `identity_seed`, `style_pool_index`, `letterhead_seed` are populated where
  the tier requires them so the ablation experiments in spec §4.7 can be
  re-run from the manifest alone.
- `provenance_marker` is a JSON string recording which markers were written;
  `sha256_pre_marker` + `sha256` let you verify re-save resilience.

## Variant adapters

- A — OpenAI `gpt-image-2` (image) + `gpt-4o` (text/vision) — `sec/adapters/gpt.py`
- B — Google `gemini-2.5-flash-image` + `gemini-2.5-pro` — `sec/adapters/gemini.py`
- C — Ideogram v3 Magic Fill / generate — `sec/adapters/ideogram.py`
- D — local ComfyUI with Flux.1-Fill (inpaint) + SD3 Medium (generate) — `sec/adapters/comfyui.py`
  - Workflows live at `configs/comfyui/flux_fill_inpaint.json` and
    `configs/comfyui/sd3_medium_generate.json`. The adapter performs string
    substitution on `<<PROMPT>>`, `<<SEED>>`, etc., then POSTs the result to
    ComfyUI's `/prompt` endpoint.

All four implement the protocol in `sec/adapters/base.py` so tier code is
variant-agnostic.

## Tier implementation notes

- **T1 (date / dollar)** uses the SROIE task2 JSON (`company`, `date`,
  `total`) as ground truth for the edit target, cross-referencing task1 OCR
  polygons to locate the bounding box precisely. **Dates** use **full-frame
  `adapter.inpaint` only** (same receipt, new date); there is no local fallback
  unless `tier1_date.use_local_burn_only: true`. **`image_edit.scope`** affects
  **dollar** (and T2/T3), not the date API shape. **Dollar** edits use
  ``adapter.inpaint`` only (full frame or patch); there is no local burn on
  failure.
  Sub-variants: consistent vs inconsistent dollar
  edits (half / half by item index).
- **T2 (signature / handwriting)** uses `image_edit.scope`: `full_image` calls
  `adapter.inpaint` on the full frame with a region hint; **failure stops the
  item** (no patch/composite fallback). `patch` uses `few_shot_image` + local
  composite of **API** glyphs only.
- **T3 (line-item insertion)** drafts a new receipt line via
  `text_complete` conditioned on a target from `assets/clause_targets.txt`,
  then runs `adapter.inpaint` (`full_image` or masked strip per `image_edit.scope`).
  **There is no local burn fallback** on vision failure.
- **T4 (whole-receipt fabrication)** calls `adapter.few_shot_image` with up to
  three SROIE anchor images plus `prompts/T4-RCT-image.md`; the API raster is
  the artifact (no JSON + Pillow path).

## Determinism

Every decision is seeded from `item_seed = batch_seed * 1000 + item_index`,
so any run of the pipeline yields the same artifact hashes (modulo the
provider-side nondeterminism of the large closed models, which the manifest
captures).

## What is NOT in this deliverable

- EML and DOC families (Enron, Avocado, UCSF, RVL-CDIP, DUDE). The manifest
  schema, adapter contract, batch registry, and QA harness are already
  family-agnostic; to add a family, implement a loader under
  `sec/sources/<family>.py` and the corresponding tier edits under
  `sec/edits/`.
- CORD (`SRC-CORD-TRN` / `SRC-CORD-TST`) — the same hook points apply.
- Find It Again! `external_labeled` calibration subset.
