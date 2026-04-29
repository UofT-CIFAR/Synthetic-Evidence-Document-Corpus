"""Single-batch runner for the 32 SROIE RCT batches.

Given a `BatchSpec`, this module:
- Resolves the Variant adapter from tools.yaml.
- Picks `items` source documents from the pool (deterministic by seed,
  filtered by task2 availability and OCR confidence).
- Dispatches to the correct tier-specific edit module.
- Writes artifacts with provenance markers, appends manifest rows, and logs
  per-item JSONL.
- Runs the QA harness and writes `qa/<batch_id>.md`.

Acceptance per spec §9.3 is gated at the CLI level in
`scripts/run_batch.py` by reading the QA report.
"""

from __future__ import annotations

import getpass
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import provenance
from .adapters.base import VariantAdapter, load_adapter
from .batch_registry import BatchSpec
from .clean_controls import _prov_cfg
from .config import Config
from .edits import t1_date_img, t1_dollar_img, t2_hw, t2_sig, t3_insert, t4_rct
from .edits.common import parse_amount
from .logging_utils import BatchLogger, new_logger
from .manifest import append_rows
from .ocr.tesseract import mean_confidence
from .pools import PoolSplit, sample_ids
from .provenance import ProvenanceConfig
from .seeding import item_seeds
from .sources.sroie import SROIELoader, SROIEItem
from .style_pools import StylePools


LOG = new_logger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


class BatchRunner:
    def __init__(
        self,
        config: Config,
        batch: BatchSpec,
        loader: Any,
        split: PoolSplit,
        *,
        style_pools: StylePools | None = None,
        prov_cfg: ProvenanceConfig | None = None,
        source_dataset: str | None = None,
        source_license: str | None = None,
    ) -> None:
        self.config = config
        self.batch = batch
        self.loader = loader
        self.split = split
        self.style_pools = style_pools
        self.prov_cfg = prov_cfg or _prov_cfg(config)
        self.adapter: VariantAdapter = load_adapter(batch.variant, config.tools)
        # Source-dataset tag for the manifest; fall back to loader-provided
        # constants so SROIE/CORD both work without caller boilerplate.
        self.source_dataset = (
            source_dataset
            or getattr(loader, "SOURCE_DATASET", None)
            or "SROIE2019"
        )
        self.source_license = (
            source_license
            or getattr(loader, "SOURCE_LICENSE", None)
            or "ICDAR 2019 SROIE task license"
        )
        self.logger = BatchLogger(
            logs_dir=config.logs_dir,
            prompts_log_dir=config.prompts_log_dir,
            batch_id=batch.batch_id,
        )
        config.ensure_runtime_dirs()
        self.out_dir = config.corpus_batch_dir(batch.pool, batch.family, batch.batch_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._items_cache: dict[str, SROIEItem] | None = None
        # Variant B (Gemini) must not report success when the pipeline silences API failure.
        self._forbid_adapter_fallback = batch.variant == "B"

    # -- helpers ----------------------------------------------------------

    def _ensure_items_cache(self) -> dict[str, SROIEItem]:
        if self._items_cache is None:
            self._items_cache = {item.doc_id: item for item in self.loader.iter_items(include_test=True)}
        return self._items_cache

    def _eligible_source_ids(self) -> list[str]:
        pool_ids = list(self.split.for_pool(self.batch.pool))
        cache = self._ensure_items_cache()
        eligible: list[str] = []
        min_conf = float(self.config.tools.get("ocr", {}).get("min_confidence", 0.85))
        for doc_id in pool_ids:
            item = cache.get(doc_id)
            if not item:
                continue
            if not item.has_task2():
                # Many tiers depend on task2 fields; skip items without them.
                if self.batch.tier in ("T1", "T3"):
                    continue
            if self.batch.tier in ("T1", "T3"):
                # T1 (dollar) and T3 (insert) both need a parseable total to
                # perform a meaningful edit. CORD has some rows with an empty
                # gt_parse.total.total_price; filter them out here so the
                # sampler never picks them.
                total_text = (item.task2_kv or {}).get("total", "")
                if parse_amount(total_text) is None:
                    continue
            summary = mean_confidence(item.image_path) if item.image_path.exists() else None
            if summary and summary.available and summary.mean_confidence < min_conf:
                continue
            eligible.append(doc_id)
        return eligible

    def _sample_source(self, item_index: int) -> SROIEItem | None:
        eligible = self._eligible_source_ids()
        if not eligible:
            return None
        picks = sample_ids(eligible, n=self.batch.items, seed=self.batch.seed ^ 0x50415353)
        if item_index >= len(picks):
            picks = picks + [eligible[(item_index + i) % len(eligible)] for i in range(self.batch.items)]
        doc_id = picks[item_index]
        return self._ensure_items_cache().get(doc_id)

    def _write(self, image: Image.Image, artifact_id: str) -> dict[str, Any]:
        out_path = self.out_dir / f"{artifact_id}.png"
        return provenance.write_image_with_provenance(image, out_path, cfg=self.prov_cfg)

    def _base_row(self, *, artifact_id: str, item_index: int, source: SROIEItem | None) -> dict:
        seeds = item_seeds(self.batch.seed, item_index)
        return {
            "artifact_id": artifact_id,
            "pool": self.batch.pool,
            "family": self.batch.family,
            "tier": self.batch.tier_int(),
            "batch_id": self.batch.batch_id,
            "variant": self.batch.variant,
            "tool_family": self.batch.tool_family,
            "tool_specific": self.batch.tool_specific,
            "source_artifact_id": source.doc_id if source else None,
            "source_dataset": self.source_dataset,
            "source_license": self.source_license,
            "prompt": None,
            "edit_regions": None,
            "identity_seed": None,
            "style_pool_index": None,
            "letterhead_seed": None,
            "intended_evidentiary_role": f"synthetic RCT {self.batch.tier} receipt",
            "provenance_marker": "",
            "sha256": "",
            "sha256_pre_marker": None,
            "item_index": item_index,
            "item_seed": seeds.item_seed,
            "file_path": "",
            "created_at": datetime.now(timezone.utc),
            "created_by": f"sec.batch_runner@{getpass.getuser()}",
            "notes": None,
        }

    # -- tier dispatch ----------------------------------------------------

    def run(self) -> dict[str, int]:
        handlers: dict[str, Callable[[int], tuple[Image.Image, dict, str, str] | None]] = {
            "T1": self._run_t1_item,
            "T2": self._run_t2_item,
            "T3": self._run_t3_item,
            "T4": self._run_t4_item,
        }
        handler = handlers[self.batch.tier]
        rows: list[dict] = []
        generated = failed = skipped = 0
        for i in range(self.batch.items):
            try:
                result = handler(i)
                if result is None:
                    skipped += 1
                    self.logger.log_item(
                        {"batch_id": self.batch.batch_id, "item_index": i, "status": "skipped"}
                    )
                    continue
                image, row_patch, prompt, raw = result
                artifact_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"sec:{self.batch.batch_id}:{i}")
                )
                marker = self._write(image, artifact_id)
                row = self._base_row(
                    artifact_id=artifact_id,
                    item_index=i,
                    source=row_patch.get("_source"),
                )
                row["prompt"] = prompt or None
                row["provenance_marker"] = marker["provenance_marker"]
                row["sha256"] = marker["sha256"]
                row["sha256_pre_marker"] = marker["sha256_pre_marker"]
                row["file_path"] = str(
                    (self.out_dir / f"{artifact_id}.png").relative_to(self.config.project_root)
                )
                for k, v in row_patch.items():
                    if k.startswith("_"):
                        continue
                    row[k] = v
                rows.append(row)
                generated += 1
                self.logger.log_item(
                    {
                        "batch_id": self.batch.batch_id,
                        "item_index": i,
                        "status": "ok",
                        "artifact_id": artifact_id,
                        "sha256_pre_marker": marker["sha256_pre_marker"],
                        "sha256": marker["sha256"],
                        "source_artifact_id": row.get("source_artifact_id"),
                        "tier": self.batch.tier,
                        "variant": self.batch.variant,
                    }
                )
                if raw and self.batch.tier in ("T3", "T4"):
                    self.logger.log_prompt(
                        {
                            "batch_id": self.batch.batch_id,
                            "item_index": i,
                            "artifact_id": artifact_id,
                            "prompt": prompt,
                            "response": raw,
                        }
                    )
            except Exception as e:  # noqa: BLE001
                failed += 1
                LOG.exception("Batch item failed: %s[%d]", self.batch.batch_id, i)
                self.logger.log_item(
                    {
                        "batch_id": self.batch.batch_id,
                        "item_index": i,
                        "status": "error",
                        "error": str(e),
                        "trace": traceback.format_exc(),
                    }
                )
        if rows:
            append_rows(self.config.manifest_path, rows)
        return {"generated": generated, "failed": failed, "skipped": skipped}

    # -- per-tier item handlers ------------------------------------------

    def _run_t1_item(self, item_index: int):
        source = self._sample_source(item_index)
        if source is None:
            return None
        # Split T1 50/50 between DATE and DOLLAR by item index so a batch
        # contains both sub-operations. If the source has no parseable date
        # (common on CORD), silently fall through to a DOLLAR edit so the
        # batch still fills out to `items`.
        if item_index % 2 == 0:
            try:
                res = t1_date_img.apply(
                    source,
                    adapter=self.adapter,
                    item_index=item_index,
                    seed=self.batch.seed * 1000 + item_index,
                    forbid_adapter_fallback=self._forbid_adapter_fallback,
                )
            except ValueError as e:
                LOG.info(
                    "T1 date skipped for %s (%s) -> falling back to dollar",
                    source.doc_id,
                    e,
                )
            else:
                patch = {
                    "_source": source,
                    "edit_regions": [
                        {
                            "page": 0,
                            "x": res.bbox[0],
                            "y": res.bbox[1],
                            "w": res.bbox[2],
                            "h": res.bbox[3],
                            "kind": "date",
                            "old_text": res.old_date,
                            "new_text": res.new_date,
                        }
                    ],
                    "notes": f"offset_days={res.offset_days}; {res.notes}",
                }
                return res.image, patch, res.prompt, ""
        res = t1_dollar_img.apply(
            source,
            adapter=self.adapter,
            item_index=item_index,
            seed=self.batch.seed * 1000 + item_index,
            forbid_adapter_fallback=self._forbid_adapter_fallback,
        )
        patch = {
            "_source": source,
            "edit_regions": [
                {
                    "page": 0,
                    "x": r.bbox[0],
                    "y": r.bbox[1],
                    "w": r.bbox[2],
                    "h": r.bbox[3],
                    "kind": r.kind,
                    "old_text": r.old_text,
                    "new_text": r.new_text,
                }
                for r in res.edited_regions
            ],
            "notes": f"sub_variant={res.sub_variant}; factor={res.factor}; {res.notes}",
        }
        return res.image, patch, res.prompt, ""

    def _run_t2_item(self, item_index: int):
        source = self._sample_source(item_index)
        if source is None:
            return None
        if self.style_pools is None:
            raise RuntimeError("Tier-2 batches require style_pools; call BatchRunner with style_pools=...")
        if item_index % 2 == 0:
            res = t2_sig.apply(
                source,
                adapter=self.adapter,
                pools=self.style_pools,
                pool=self.batch.pool,
                item_index=item_index,
                batch_seed_value=self.batch.seed,
                forbid_adapter_fallback=self._forbid_adapter_fallback,
            )
            patch = {
                "_source": source,
                "identity_seed": res.identity_seed,
                "style_pool_index": res.style_pool_index,
                "edit_regions": [
                    {
                        "page": 0,
                        "x": res.bbox[0],
                        "y": res.bbox[1],
                        "w": res.bbox[2],
                        "h": res.bbox[3],
                        "kind": "signature",
                        "old_text": "",
                        "new_text": res.signature_name,
                    }
                ],
                "notes": (
                    f"perturb rot={res.perturbation.rotation_deg:.2f} "
                    f"scale={res.perturbation.scale:.3f} shear={res.perturbation.shear_deg:.2f}; "
                    f"{res.notes}"
                ),
            }
            return res.image, patch, res.prompt, ""
        res = t2_hw.apply(
            source,
            adapter=self.adapter,
            pools=self.style_pools,
            pool=self.batch.pool,
            item_index=item_index,
            batch_seed_value=self.batch.seed,
            assets_dir=ASSETS_DIR,
            forbid_adapter_fallback=self._forbid_adapter_fallback,
        )
        patch = {
            "_source": source,
            "identity_seed": res.identity_seed,
            "style_pool_index": res.style_pool_index,
            "edit_regions": [
                {
                    "page": 0,
                    "x": res.bbox[0],
                    "y": res.bbox[1],
                    "w": res.bbox[2],
                    "h": res.bbox[3],
                    "kind": "handwritten",
                    "old_text": "",
                    "new_text": res.phrase,
                }
            ],
            "notes": f"phrase={res.phrase!r}; {res.notes}",
        }
        return res.image, patch, res.prompt, ""

    def _run_t3_item(self, item_index: int):
        source = self._sample_source(item_index)
        if source is None:
            return None
        res = t3_insert.apply(
            source,
            adapter=self.adapter,
            item_index=item_index,
            seed=self.batch.seed * 1000 + item_index,
            assets_dir=ASSETS_DIR,
            prompts_dir=PROMPTS_DIR,
            forbid_adapter_fallback=self._forbid_adapter_fallback,
        )
        patch = {
            "_source": source,
            "edit_regions": [
                {
                    "page": 0,
                    "x": res.bbox[0],
                    "y": res.bbox[1],
                    "w": res.bbox[2],
                    "h": res.bbox[3],
                    "kind": "line_item_insert",
                    "old_text": "",
                    "new_text": res.inserted_text,
                }
            ],
            "notes": f"target={res.target}; {res.notes}",
        }
        return res.image, patch, res.prompt, res.response_raw

    def _run_t4_item(self, item_index: int):
        res = t4_rct.apply(
            adapter=self.adapter,
            loader=self.loader,
            item_index=item_index,
            batch_seed_value=self.batch.seed,
            prompts_dir=PROMPTS_DIR,
            forbid_adapter_fallback=self._forbid_adapter_fallback,
        )
        patch = {
            "_source": None,
            "source_artifact_id": None,
            "identity_seed": res.identity_seed,
            "letterhead_seed": res.letterhead_seed,
            "edit_regions": None,
            "notes": (
                f"sub_variant={res.sub_variant}; anchors={','.join(res.anchor_ids)}; {res.notes}"
            ),
        }
        return res.image, patch, res.prompt, res.response_raw
