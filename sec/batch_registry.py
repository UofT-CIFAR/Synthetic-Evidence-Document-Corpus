"""Batch registry.

Enumerates every batch in the 96-cell corpus matrix and provides filters that
select the 32 SROIE `RCT` batches this deliverable actually produces.

Each batch record is a plain dataclass so it can be serialized into the batch
log or a YAML sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .seeding import POOLS, FAMILIES, TIERS, VARIANTS, batch_seed


@dataclass(frozen=True)
class BatchSpec:
    batch_id: str
    pool: str
    family: str
    tier: str
    variant: str
    items: int
    seed: int
    tool_family: str
    tool_specific: str
    source_datasets: tuple[str, ...]
    transformation: str

    def tier_int(self) -> int:
        return int(self.tier[1])

    def as_dict(self) -> dict:
        d = asdict(self)
        d["source_datasets"] = list(self.source_datasets)
        return d


_TRAIN_ITEMS = 20
_TEST_ITEMS = 10


def _tool_info(tier: str, variant: str) -> tuple[str, str]:
    if variant == "A":
        return "closed_llm", "openai:gpt-image-2+gpt-4o"
    if variant == "B":
        return "closed_llm", "google:gemini-2.5-flash-image"
    if variant == "C":
        return "closed_llm", "ideogram:v3"
    if variant == "D":
        if tier == "T4":
            return "open_llm", "comfyui:sd3-medium"
        return "open_llm", "comfyui:flux1-fill-dev"
    raise ValueError(variant)


_TRANSFORMATIONS = {
    ("RCT", "T1"): "Change date or dollar amount",
    ("RCT", "T2"): "Forge signature or add margin note",
    ("RCT", "T3"): "Insert line item or service fee",
    ("RCT", "T4"): "Fabricate whole receipt",
    ("EML", "T1"): "Change header date or timestamp",
    ("EML", "T2"): "Forge signature block or margin note in rendered email",
    ("EML", "T3"): "Insert a reply impersonating a real participant",
    ("EML", "T4"): "Fabricate whole email thread",
    ("DOC", "T1"): "Change date, dollar amount, or page order",
    ("DOC", "T2"): "Forge signature or add handwritten annotation",
    ("DOC", "T3"): "Insert paragraph, contract clause, or clinical note",
    ("DOC", "T4"): "Fabricate whole letter, memo, or report",
}


def _sources(pool: str, family: str, tier: str) -> tuple[str, ...]:
    if family == "EML":
        if pool == "TRN":
            if tier == "T4":
                return ("Enron-anchors",)
            return ("Enron", "RVL-CDIP-email")
        if tier == "T4":
            return ("Avocado-anchors",)
        return ("Avocado",)
    if family != "RCT":
        # Non-RCT families besides EML are not implemented yet; the registry
        # still enumerates cells for schema/seed validation.
        return ("<not-implemented>",)
    if pool == "TRN":
        if tier == "T3":
            return ("SROIE", "CORD", "FindItAgain")
        if tier == "T4":
            return ("SROIE-anchors", "CORD-anchors")
        return ("SROIE", "CORD")
    if tier == "T4":
        return ("SROIE-held-back-anchors", "CORD-held-back-anchors")
    return ("SROIE-held-back", "CORD-held-back")


def all_batches() -> list[BatchSpec]:
    batches: list[BatchSpec] = []
    for pool in POOLS:
        items = _TRAIN_ITEMS if pool == "TRN" else _TEST_ITEMS
        for family in FAMILIES:
            for tier in TIERS:
                for variant in VARIANTS:
                    tf, tool = _tool_info(tier, variant)
                    batch_id = f"{pool}-{family}-{tier}-{variant}"
                    batches.append(
                        BatchSpec(
                            batch_id=batch_id,
                            pool=pool,
                            family=family,
                            tier=tier,
                            variant=variant,
                            items=items,
                            seed=batch_seed(pool, family, tier, variant),
                            tool_family=tf,
                            tool_specific=tool,
                            source_datasets=_sources(pool, family, tier),
                            transformation=_TRANSFORMATIONS[(family, tier)],
                        )
                    )
    return batches


def sroie_batches() -> list[BatchSpec]:
    return [b for b in all_batches() if b.family == "RCT"]


# ---------------------------------------------------------------------------
# CORD-v2 parallel track (same RCT family, different source dataset).
#
# The SEC spec (§5) allows a receipts batch to draw from multiple sources. To
# keep the already-produced SROIE artifacts reproducible without touching
# their seeds, CORD is emitted as a parallel set of 32 batches whose batch_id
# carries a ``-CORD`` suffix and whose seeds live in a disjoint 14xxx / 84xxx
# band. The manifest family stays ``RCT``; only ``source_dataset`` changes.
# ---------------------------------------------------------------------------


def cord_batch_seed(pool: str, tier: str, variant: str) -> int:
    """Batch seed for a CORD batch. Disjoint from the 11xxx/81xxx SROIE band."""

    if pool not in POOLS:
        raise ValueError(pool)
    if tier not in TIERS:
        raise ValueError(tier)
    if variant not in VARIANTS:
        raise ValueError(variant)
    base = 14000 if pool == "TRN" else 84000
    return base + TIERS.index(tier) * 100 + VARIANTS.index(variant)


def cord_batches() -> list[BatchSpec]:
    out: list[BatchSpec] = []
    for pool in POOLS:
        items = _TRAIN_ITEMS if pool == "TRN" else _TEST_ITEMS
        for tier in TIERS:
            for variant in VARIANTS:
                tf, tool = _tool_info(tier, variant)
                batch_id = f"{pool}-RCT-{tier}-{variant}-CORD"
                out.append(
                    BatchSpec(
                        batch_id=batch_id,
                        pool=pool,
                        family="RCT",
                        tier=tier,
                        variant=variant,
                        items=items,
                        seed=cord_batch_seed(pool, tier, variant),
                        tool_family=tf,
                        tool_specific=tool,
                        source_datasets=("CORD-v2",),
                        transformation=_TRANSFORMATIONS[("RCT", tier)],
                    )
                )
    return out


def findit2_batch_seed(pool: str, tier: str, variant: str) -> int:
    """Batch seed for a FindIt2 batch. Disjoint from 11/81 (SROIE) and 14/84 (CORD)."""

    if pool not in POOLS:
        raise ValueError(pool)
    if tier not in TIERS:
        raise ValueError(tier)
    if variant not in VARIANTS:
        raise ValueError(variant)
    base = 16000 if pool == "TRN" else 86000
    return base + TIERS.index(tier) * 100 + VARIANTS.index(variant)


def findit2_batches() -> list[BatchSpec]:
    """Parallel RCT track for FindIt2 (non-forged source rows only; see loader)."""

    out: list[BatchSpec] = []
    for pool in POOLS:
        items = _TRAIN_ITEMS if pool == "TRN" else _TEST_ITEMS
        for tier in TIERS:
            for variant in VARIANTS:
                tf, tool = _tool_info(tier, variant)
                batch_id = f"{pool}-RCT-{tier}-{variant}-FIN"
                out.append(
                    BatchSpec(
                        batch_id=batch_id,
                        pool=pool,
                        family="RCT",
                        tier=tier,
                        variant=variant,
                        items=items,
                        seed=findit2_batch_seed(pool, tier, variant),
                        tool_family=tf,
                        tool_specific=tool,
                        source_datasets=("FindIt2",),
                        transformation=_TRANSFORMATIONS[("RCT", tier)],
                    )
                )
    return out


# ---------------------------------------------------------------------------
# RVL-CDIP email-class parallel track (EML family, training pool only).
#
# ``chainyo/rvl-cdip`` rows labeled ``email`` are cached as PNGs and run
# through the same EML tiers as Enron. Batch IDs use a ``-RVLCDIP`` suffix and
# seeds in the 17xxx band (disjoint from Enron EML 12xxx / FindIt 16xxx).
# ---------------------------------------------------------------------------


def rvl_cdip_eml_batch_seed(tier: str, variant: str) -> int:
    if tier not in TIERS:
        raise ValueError(tier)
    if variant not in VARIANTS:
        raise ValueError(variant)
    base = 17000
    return base + TIERS.index(tier) * 100 + VARIANTS.index(variant)


def rvl_cdip_eml_batches() -> list[BatchSpec]:
    """Training-only EML batches sourced from RVL-CDIP ``email`` pages."""

    out: list[BatchSpec] = []
    pool = "TRN"
    items = _TRAIN_ITEMS
    for tier in TIERS:
        for variant in VARIANTS:
            tf, tool = _tool_info(tier, variant)
            batch_id = f"{pool}-EML-{tier}-{variant}-RVLCDIP"
            out.append(
                BatchSpec(
                    batch_id=batch_id,
                    pool=pool,
                    family="EML",
                    tier=tier,
                    variant=variant,
                    items=items,
                    seed=rvl_cdip_eml_batch_seed(tier, variant),
                    tool_family=tf,
                    tool_specific=tool,
                    source_datasets=("RVL-CDIP-email",),
                    transformation=_TRANSFORMATIONS[("EML", tier)],
                )
            )
    return out


def eml_matrix_batches() -> list[BatchSpec]:
    """32-cell matrix ``{TRN,TST}-EML-T{1..4}-{A..D}`` (Enron train / Avocado test)."""

    return [b for b in all_batches() if b.family == "EML"]


def eml_batches() -> list[BatchSpec]:
    """Every EML batch id: matrix plus TRN-only ``*-RVLCDIP`` parallel track (16)."""

    merged = [*eml_matrix_batches(), *rvl_cdip_eml_batches()]
    return sorted(merged, key=lambda b: (b.pool, b.tier, b.variant, b.batch_id))


def get(batch_id: str) -> BatchSpec:
    for b in all_batches():
        if b.batch_id == batch_id:
            return b
    for b in cord_batches():
        if b.batch_id == batch_id:
            return b
    for b in findit2_batches():
        if b.batch_id == batch_id:
            return b
    for b in rvl_cdip_eml_batches():
        if b.batch_id == batch_id:
            return b
    raise KeyError(f"Unknown batch_id: {batch_id}")


def filter_batches(
    batches: Iterable[BatchSpec],
    *,
    pool: str | None = None,
    family: str | None = None,
    tier: str | None = None,
    variant: str | None = None,
) -> list[BatchSpec]:
    out: list[BatchSpec] = []
    for b in batches:
        if pool and b.pool != pool:
            continue
        if family and b.family != family:
            continue
        if tier and b.tier != tier:
            continue
        if variant and b.variant != variant:
            continue
        out.append(b)
    return out


def validate_registry() -> None:
    """Sanity checks mirroring the snippet in spec §7."""

    all_b = all_batches()
    assert len(all_b) == 96, f"Expected 96 batches, got {len(all_b)}"
    seeds = {b.seed for b in all_b}
    assert len(seeds) == 96, "Batch seeds must be unique"
    ids = {b.batch_id for b in all_b}
    assert len(ids) == 96, "Batch IDs must be unique"
    rct = [b for b in all_b if b.family == "RCT"]
    assert len(rct) == 32, f"Expected 32 RCT batches, got {len(rct)}"

    cord = cord_batches()
    assert len(cord) == 32, f"Expected 32 CORD batches, got {len(cord)}"
    cord_seeds = {b.seed for b in cord}
    assert len(cord_seeds) == 32, "CORD batch seeds must be unique"
    assert cord_seeds.isdisjoint(seeds), "CORD seeds overlap SROIE seeds"
    cord_ids = {b.batch_id for b in cord}
    assert cord_ids.isdisjoint(ids), "CORD batch IDs overlap SROIE batch IDs"

    fin = findit2_batches()
    assert len(fin) == 32, f"Expected 32 FindIt2 batches, got {len(fin)}"
    fin_seeds = {b.seed for b in fin}
    assert len(fin_seeds) == 32, "FindIt2 batch seeds must be unique"
    assert fin_seeds.isdisjoint(seeds), "FindIt2 seeds overlap SROIE seeds"
    assert fin_seeds.isdisjoint(cord_seeds), "FindIt2 seeds overlap CORD seeds"
    fin_ids = {b.batch_id for b in fin}
    assert fin_ids.isdisjoint(ids), "FindIt2 batch IDs overlap SROIE batch IDs"
    assert fin_ids.isdisjoint(cord_ids), "FindIt2 batch IDs overlap CORD batch IDs"

    rvl_eml = rvl_cdip_eml_batches()
    assert len(rvl_eml) == 16, f"Expected 16 RVL-CDIP EML batches, got {len(rvl_eml)}"
    rvl_seeds = {b.seed for b in rvl_eml}
    assert len(rvl_seeds) == 16, "RVL-CDIP EML batch seeds must be unique"
    assert rvl_seeds.isdisjoint(seeds), "RVL-CDIP EML seeds overlap base matrix seeds"
    assert rvl_seeds.isdisjoint(cord_seeds), "RVL-CDIP EML seeds overlap CORD seeds"
    assert rvl_seeds.isdisjoint(fin_seeds), "RVL-CDIP EML seeds overlap FindIt2 seeds"
    rvl_ids = {b.batch_id for b in rvl_eml}
    assert rvl_ids.isdisjoint(ids), "RVL-CDIP EML IDs overlap base batch IDs"
    assert rvl_ids.isdisjoint(cord_ids), "RVL-CDIP EML IDs overlap CORD batch IDs"
    assert rvl_ids.isdisjoint(fin_ids), "RVL-CDIP EML IDs overlap FindIt2 batch IDs"
