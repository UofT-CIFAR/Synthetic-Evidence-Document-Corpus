"""Deterministic seeding helpers.

The spec (§4.2) requires: `item_seed = batch_seed * 1000 + item_index`.
Sub-seeds for identity, style pool, and letterhead are derived from the item
seed so they can be individually swapped in ablation studies (spec §4.7).

The batch seed schema in spec §7 is:

    TRN-<family>-T<tier>-<variant> -> 11000 + family_offset * 1000 + tier_idx*100 + variant_idx
    TST-<family>-T<tier>-<variant> -> 81000 + family_offset * 1000 + tier_idx*100 + variant_idx

with FAMILIES = [RCT, EML, DOC] and VARIANTS = [A, B, C, D].
"""

from __future__ import annotations

from dataclasses import dataclass


FAMILIES = ["RCT", "EML", "DOC"]
TIERS = ["T1", "T2", "T3", "T4"]
VARIANTS = ["A", "B", "C", "D"]
POOLS = ["TRN", "TST"]


@dataclass(frozen=True)
class ItemSeeds:
    item_seed: int
    identity_seed: int
    style_pool_seed: int
    letterhead_seed: int
    sampler_seed: int
    perturbation_seed: int


def batch_seed(pool: str, family: str, tier: str, variant: str) -> int:
    """Return the batch seed for a given (pool, family, tier, variant) cell.

    Matches the table in spec §7. Seeds are unique across the 96 cells.
    """

    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    base = 11000 if pool == "TRN" else 81000
    f_idx = FAMILIES.index(family)
    t_idx = TIERS.index(tier)
    v_idx = VARIANTS.index(variant)
    return base + f_idx * 1000 + t_idx * 100 + v_idx


def item_seeds(batch_seed_value: int, item_index: int) -> ItemSeeds:
    """Derive the set of sub-seeds used by a single item."""

    item = batch_seed_value * 1000 + item_index
    return ItemSeeds(
        item_seed=item,
        identity_seed=item,
        style_pool_seed=item ^ 0xA5A5A5A5,
        letterhead_seed=item ^ 0x5A5A5A5A,
        sampler_seed=item ^ 0x13579BDF,
        perturbation_seed=item ^ 0x2468ACE0,
    )


def all_batch_cells() -> list[tuple[str, str, str, str, int]]:
    """Return every (pool, family, tier, variant, seed) cell in the corpus."""

    cells: list[tuple[str, str, str, str, int]] = []
    for pool in POOLS:
        for family in FAMILIES:
            for tier in TIERS:
                for variant in VARIANTS:
                    cells.append((pool, family, tier, variant, batch_seed(pool, family, tier, variant)))
    return cells
