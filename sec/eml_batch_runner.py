"""Batch runner for EML (email) corpus batches — mirrors ``batch_runner.BatchRunner``.

Artifacts are PNG renders so provenance + QA match the RCT raster pipeline.
"""

from __future__ import annotations

import getpass
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import provenance
from .adapters.base import VariantAdapter, load_adapter
from .batch_registry import BatchSpec
from .clean_controls import _prov_cfg
from .config import Config
from .edits import eml_t1_date, eml_t2_hw, eml_t2_sig, eml_t4_thread
from .edits.eml_t3_reply import apply as eml_t3_apply
from .logging_utils import BatchLogger, new_logger
from .ocr.tesseract import document_confidence_and_word_count
from .manifest import append_rows, read_rows
from .pools import PoolSplit, sample_ids
from .provenance import ProvenanceConfig
from .seeding import item_seeds
from .sources.mail_base import EmailItem, load_email_bytes, parse_email_message, plain_body
from .style_pools import StylePools


LOG = new_logger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _image_edit_scope(tools: dict[str, Any] | None) -> str:
    if not tools:
        return "full_image"
    ie = tools.get("image_edit")
    if isinstance(ie, dict):
        s = str(ie.get("scope", "full_image")).strip().lower()
        if s in ("full_image", "patch"):
            return s
    return "full_image"


class EMLBatchRunner:
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
        exclude_source_ids: frozenset[str] | None = None,
    ) -> None:
        self.config = config
        self.batch = batch
        self.loader = loader
        self.split = split
        self.style_pools = style_pools
        self.prov_cfg = prov_cfg or _prov_cfg(config)
        self.adapter: VariantAdapter = load_adapter(batch.variant, config.tools)
        self.source_dataset = source_dataset or getattr(loader, "SOURCE_DATASET", None) or "EML"
        self.source_license = source_license or getattr(loader, "SOURCE_LICENSE", None) or ""
        self.logger = BatchLogger(
            logs_dir=config.logs_dir,
            prompts_log_dir=config.prompts_log_dir,
            batch_id=batch.batch_id,
        )
        self._exclude_source_ids = exclude_source_ids or frozenset()
        config.ensure_runtime_dirs()
        self.out_dir = config.corpus_batch_dir(batch.pool, batch.family, batch.batch_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._items_cache: dict[str, EmailItem] | None = None
        self._eligible_source_ids_cache: list[str] | None = None
        self._image_edit_scope = _image_edit_scope(config.tools)

    def _artifact_id(self, item_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sec:{self.batch.batch_id}:{item_index}"))

    def _manifest_rows_by_item_index(self) -> dict[int, dict]:
        path = self.config.manifest_path
        if not path.exists():
            return {}
        bid = self.batch.batch_id
        out: dict[int, dict] = {}
        for r in read_rows(path):
            if r.get("batch_id") != bid:
                continue
            idx = r.get("item_index")
            if idx is None:
                continue
            try:
                ix = int(idx)
            except (TypeError, ValueError):
                continue
            out[ix] = r
        return out

    def _ensure_items_cache(self) -> dict[str, EmailItem]:
        if self._items_cache is None:
            self._items_cache = {item.doc_id: item for item in self.loader.iter_items()}
        return self._items_cache

    def _eligible_source_ids(self) -> list[str]:
        if self._eligible_source_ids_cache is not None:
            return self._eligible_source_ids_cache
        pool_ids = list(self.split.for_pool(self.batch.pool))
        cache = self._ensure_items_cache()
        eligible: list[str] = []
        min_conf = float(self.config.tools.get("ocr", {}).get("min_confidence", 0.85))
        for doc_id in pool_ids:
            if doc_id in self._exclude_source_ids:
                continue
            item = cache.get(doc_id)
            if not item:
                continue
            if getattr(item, "modality", "rfc822") == "rvl_email_page":
                summary, wc = document_confidence_and_word_count(item.path)
                if summary and summary.available and summary.mean_confidence < min_conf:
                    continue
                if wc < 12:
                    continue
                eligible.append(doc_id)
                continue
            try:
                msg = parse_email_message(load_email_bytes(item.path))
            except Exception:
                continue
            if not msg.get("Date"):
                continue
            if len(plain_body(msg).split()) < 12:
                continue
            eligible.append(doc_id)
        self._eligible_source_ids_cache = eligible
        return eligible

    def _sample_source(self, item_index: int) -> EmailItem | None:
        eligible = self._eligible_source_ids()
        if not eligible:
            return None
        picks = sample_ids(eligible, n=self.batch.items, seed=self.batch.seed ^ 0x50415353)
        if item_index >= len(picks):
            picks = picks + [
                eligible[(item_index + i) % len(eligible)] for i in range(self.batch.items)
            ]
        doc_id = picks[item_index]
        return self._ensure_items_cache().get(doc_id)

    def _write(self, image: Image.Image, artifact_id: str) -> dict[str, Any]:
        out_path = self.out_dir / f"{artifact_id}.png"
        return provenance.write_image_with_provenance(image, out_path, cfg=self.prov_cfg)

    def _base_row(self, *, artifact_id: str, item_index: int, source: EmailItem | None) -> dict:
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
            "intended_evidentiary_role": f"synthetic EML {self.batch.tier} email render",
            "provenance_marker": "",
            "sha256": "",
            "sha256_pre_marker": None,
            "item_index": item_index,
            "item_seed": seeds.item_seed,
            "file_path": "",
            "created_at": datetime.now(timezone.utc),
            "created_by": f"sec.eml_batch_runner@{getpass.getuser()}",
            "notes": None,
        }

    def run(self, *, skip_existing: bool = False) -> dict[str, int]:
        handlers: dict[str, Callable[[int], tuple[Image.Image, dict, str, str] | None]] = {
            "T1": self._run_t1_item,
            "T2": self._run_t2_item,
            "T3": self._run_t3_item,
            "T4": self._run_t4_item,
        }
        handler = handlers[self.batch.tier]
        rows: list[dict] = []
        manifest_by_ix = self._manifest_rows_by_item_index() if skip_existing else {}
        generated = failed = skipped = skipped_existing = 0
        for i in range(self.batch.items):
            try:
                if skip_existing:
                    aid = self._artifact_id(i)
                    png = self.out_dir / f"{aid}.png"
                    prev = manifest_by_ix.get(i)
                    if png.exists() and prev is not None and str(prev.get("artifact_id", "")) == aid:
                        skipped_existing += 1
                        self.logger.log_item(
                            {
                                "batch_id": self.batch.batch_id,
                                "item_index": i,
                                "status": "skipped_existing",
                                "artifact_id": aid,
                            }
                        )
                        continue
                result = handler(i)
                if result is None:
                    skipped += 1
                    self.logger.log_item(
                        {"batch_id": self.batch.batch_id, "item_index": i, "status": "skipped"}
                    )
                    continue
                image, row_patch, prompt, raw = result
                artifact_id = self._artifact_id(i)
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
                LOG.exception("EML batch item failed: %s[%d]", self.batch.batch_id, i)
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
        return {
            "generated": generated,
            "failed": failed,
            "skipped": skipped,
            "skipped_existing": skipped_existing,
        }

    def _run_t1_item(self, item_index: int):
        source = self._sample_source(item_index)
        if source is None:
            return None
        res = eml_t1_date.apply(
            source,
            adapter=self.adapter,
            item_index=item_index,
            seed=self.batch.seed * 1000 + item_index,
            prompts_dir=PROMPTS_DIR,
        )
        if res is None:
            return None
        patch = {
            "_source": source,
            "edit_regions": [
                {
                    "page": 0,
                    "x": res.bbox[0],
                    "y": res.bbox[1],
                    "w": res.bbox[2],
                    "h": res.bbox[3],
                    "kind": "email_date_header",
                    "old_text": res.old_date,
                    "new_text": res.new_date,
                }
            ],
            "notes": (
                f"coherent_received={res.coherent_received}; "
                f"offset_policy=spec_xt1_eml; {res.notes}"
            ),
        }
        return res.image, patch, res.prompt, ""

    def _run_t2_item(self, item_index: int):
        source = self._sample_source(item_index)
        if source is None:
            return None
        if self.style_pools is None:
            raise RuntimeError("EML Tier-2 batches require style_pools")
        if item_index % 2 == 0:
            res = eml_t2_sig.apply(
                source,
                adapter=self.adapter,
                pools=self.style_pools,
                pool=self.batch.pool,
                item_index=item_index,
                batch_seed_value=self.batch.seed,
                image_edit_scope=self._image_edit_scope,
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
                "notes": res.notes,
            }
            return res.image, patch, res.prompt, ""
        res = eml_t2_hw.apply(
            source,
            adapter=self.adapter,
            pools=self.style_pools,
            pool=self.batch.pool,
            item_index=item_index,
            batch_seed_value=self.batch.seed,
            assets_dir=ASSETS_DIR,
            image_edit_scope=self._image_edit_scope,
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
            "notes": res.notes,
        }
        return res.image, patch, res.prompt, ""

    def _run_t3_item(self, item_index: int):
        source = self._sample_source(item_index)
        if source is None:
            return None
        res = eml_t3_apply(
            source,
            adapter=self.adapter,
            item_index=item_index,
            seed=self.batch.seed * 1000 + item_index,
            assets_dir=ASSETS_DIR,
            prompts_dir=PROMPTS_DIR,
            image_edit_scope=self._image_edit_scope,
        )
        patch = {
            "_source": source,
            "identity_seed": None,
            "edit_regions": [
                {
                    "page": 0,
                    "x": res.bbox[0],
                    "y": res.bbox[1],
                    "w": res.bbox[2],
                    "h": res.bbox[3],
                    "kind": "inserted_reply",
                    "old_text": "",
                    "new_text": res.inserted_text[:500],
                }
            ],
            "notes": (
                f"target={res.target}; impersonation_subject_id={res.impersonation_subject_id}; "
                f"{res.notes}"
            ),
        }
        return res.image, patch, res.prompt, res.response_raw

    def _run_t4_item(self, item_index: int):
        res = eml_t4_thread.apply(
            adapter=self.adapter,
            loader=self.loader,
            item_index=item_index,
            batch_seed_value=self.batch.seed,
            prompts_dir=PROMPTS_DIR,
            assets_dir=ASSETS_DIR,
        )
        patch = {
            "_source": None,
            "source_artifact_id": None,
            "identity_seed": res.identity_seed_a,
            "letterhead_seed": res.identity_seed_b,
            "edit_regions": None,
            "notes": (
                f"style_variant={res.style_variant}; topic={res.topic!r}; "
                f"anchors={','.join(res.anchor_ids)}; {res.notes}"
            ),
        }
        return res.image, patch, res.prompt, res.response_raw
