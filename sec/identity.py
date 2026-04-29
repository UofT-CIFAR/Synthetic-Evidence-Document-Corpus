"""Fictional identity generator (spec §4.2).

Every fictional identity used anywhere in the corpus is generated with Faker,
seeded per item. Identities are never reused across items, even within the
same batch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from faker import Faker


@dataclass(frozen=True)
class Identity:
    item_seed: int
    name: str
    employer: str
    email: str
    phone: str
    address: str
    invalid_npi: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _invalid_luhn(seed: int) -> str:
    """Return a 10-digit NPI-style number whose Luhn checksum does NOT validate.

    Per spec §4.2 last paragraph: Tier-3 medical notes use an explicitly invalid
    Luhn checksum so the NPI cannot be confused with a real provider identifier.
    """

    import random

    rng = random.Random(seed)
    base = [rng.randint(0, 9) for _ in range(9)]
    # Compute the valid Luhn check digit, then deliberately invalidate.
    total = 0
    for i, digit in enumerate(reversed(base)):
        if i % 2 == 0:
            doubled = digit * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += digit
    check = (10 - (total % 10)) % 10
    bad_check = (check + rng.randint(1, 9)) % 10
    return "".join(str(d) for d in base) + str(bad_check)


def generate_identity(item_seed: int) -> Identity:
    """Generate a reproducible fictional identity from the given item seed."""

    fake = Faker()
    fake.seed_instance(item_seed)
    return Identity(
        item_seed=item_seed,
        name=fake.name(),
        employer=fake.company(),
        email=fake.email(),
        phone=fake.phone_number(),
        address=fake.address().replace("\n", ", "),
        invalid_npi=_invalid_luhn(item_seed),
    )


def generate_merchant(item_seed: int) -> dict[str, str]:
    """Generate a reproducible fictional merchant (Tier-4 receipts)."""

    fake = Faker()
    fake.seed_instance(item_seed ^ 0xDEADBEEF)
    return {
        "merchant_name": fake.company(),
        "merchant_address": fake.address().replace("\n", ", "),
        "merchant_phone": fake.phone_number(),
    }
