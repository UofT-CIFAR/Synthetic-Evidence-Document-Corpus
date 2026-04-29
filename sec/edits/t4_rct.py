"""X-T4-RCT: fabricate an entire receipt from nothing (spec §6 Tier 4)."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image

from ..adapters.base import AdapterCapabilityError, AdapterCredentialError, VariantAdapter
from ..identity import generate_identity, generate_merchant
from ..renderer import ReceiptDoc, render_receipt
from ..sources.sroie import SROIELoader, SROIEItem


@dataclass
class T4Result:
    image: Image.Image
    sub_variant: str  # 'consistent' | 'inconsistent'
    identity_seed: int
    letterhead_seed: int
    prompt: str
    response_raw: str
    anchor_ids: tuple[str, ...]
    doc: ReceiptDoc
    notes: str = ""


def _pick_anchors(loader: SROIELoader, seed: int, n: int = 3) -> list[SROIEItem]:
    candidates: list[SROIEItem] = []
    for item in loader.iter_items(include_test=False):
        if item.task1_lines:
            candidates.append(item)
    rng = random.Random(seed)
    if len(candidates) <= n:
        return candidates
    rng.shuffle(candidates)
    return candidates[:n]


def _anchor_block(anchors: list[SROIEItem]) -> str:
    blocks: list[str] = []
    for idx, anc in enumerate(anchors):
        lines = "\n".join(line.text for line in anc.task1_lines[:20])
        blocks.append(f"=== Anchor {idx + 1} ({anc.doc_id}) ===\n{lines}")
    return "\n\n".join(blocks)


def _render_prompt(template: str, substitutions: dict[str, str]) -> str:
    out = template
    for key, value in substitutions.items():
        out = out.replace("{" + key + "}", value)
    return out


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_receipt_json(response: str) -> dict[str, Any] | None:
    response = response.strip()
    match = _JSON_BLOCK_RE.search(response)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _local_receipt(
    customer: dict[str, Any],
    merchant: dict[str, Any],
    seed: int,
    sub_variant: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = rng.randint(3, 7)
    line_items = []
    subtotal = 0.0
    for i in range(n):
        amt = round(rng.uniform(0.99, 35.00), 2)
        label = rng.choice(
            [
                "COFFEE",
                "SANDWICH",
                "JUICE",
                "SNACK",
                "BAKERY",
                "BEVERAGE",
                "COMBO",
                "MISC",
            ]
        )
        line_items.append({"label": label, "amount": amt})
        subtotal += amt
    subtotal = round(subtotal, 2)
    tax = round(subtotal * 0.06, 2)
    if sub_variant == "inconsistent":
        tax = round(tax * rng.choice([0.5, 1.6, 2.2]), 2)
    total = round(subtotal + tax, 2)
    return {
        "merchant": merchant["merchant_name"],
        "merchant_address": merchant["merchant_address"],
        "date": (datetime.now() - timedelta(days=rng.randint(0, 365))).strftime("%d/%m/%Y"),
        "line_items": line_items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "payment_method": rng.choice(["CASH", "VISA ****1234", "MASTERCARD ****9988"]),
    }


def apply(
    *,
    adapter: VariantAdapter,
    loader: SROIELoader,
    item_index: int,
    batch_seed_value: int,
    prompts_dir: Path,
    forbid_adapter_fallback: bool = False,
) -> T4Result:
    item_seed = batch_seed_value * 1000 + item_index
    identity = generate_identity(item_seed)
    merchant = generate_merchant(item_seed)
    sub_variant = "consistent" if item_index % 2 == 0 else "inconsistent"

    anchor_seed = item_seed ^ 0xA4A4A4
    anchors = _pick_anchors(loader, anchor_seed)
    template_path = prompts_dir / "T4-RCT.md"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else (
        "Fabricate a receipt. Customer {customer_name} {customer_address}. Merchant {merchant_name} {merchant_address}. Anchors: {anchor_block}. sub_variant: {sub_variant}. Output JSON."
    )
    prompt = _render_prompt(
        template,
        {
            "customer_name": identity.name,
            "customer_address": identity.address,
            "merchant_name": merchant["merchant_name"],
            "merchant_address": merchant["merchant_address"],
            "merchant_phone": merchant["merchant_phone"],
            "anchor_block": _anchor_block(anchors),
            "sub_variant": sub_variant,
        },
    )

    response_raw = ""
    notes = ""
    try:
        response_raw = adapter.text_complete(prompt=prompt, seed=item_seed, max_tokens=700)
    except (AdapterCapabilityError, AdapterCredentialError) as e:
        if forbid_adapter_fallback:
            raise
        notes = f"adapter_fallback: {e}"

    parsed = _parse_receipt_json(response_raw) if response_raw else None
    if parsed is None:
        if forbid_adapter_fallback:
            raise AdapterCapabilityError(
                "Variant B requires adapter text_complete JSON for Tier-4; "
                "local synthetic receipt fallback disabled when forbid_adapter_fallback is True."
            )
        parsed = _local_receipt(identity.as_dict(), merchant, item_seed, sub_variant)

    line_items = [
        (str(li.get("label", "ITEM")), float(li.get("amount", 0.0)))
        for li in parsed.get("line_items", [])
    ]
    doc = ReceiptDoc(
        merchant=str(parsed.get("merchant", merchant["merchant_name"])),
        merchant_address=str(parsed.get("merchant_address", merchant["merchant_address"])),
        date=str(parsed.get("date", datetime.now().strftime("%d/%m/%Y"))),
        line_items=line_items,
        subtotal=float(parsed.get("subtotal", sum(a for _, a in line_items))),
        tax=float(parsed.get("tax", 0.0)),
        total=float(parsed.get("total", sum(a for _, a in line_items))),
        payment_method=str(parsed.get("payment_method", "CASH")),
        customer_name=identity.name,
    )
    image = render_receipt(doc, seed=item_seed)
    return T4Result(
        image=image,
        sub_variant=sub_variant,
        identity_seed=identity.item_seed,
        letterhead_seed=item_seed ^ 0x5A5A5A5A,
        prompt=prompt,
        response_raw=response_raw,
        anchor_ids=tuple(a.doc_id for a in anchors),
        doc=doc,
        notes=notes,
    )
