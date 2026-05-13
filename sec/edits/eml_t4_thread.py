"""X-T4-EML: fabricate a whole email thread image (spec §6 Tier 4).

Few-shot reference screenshots mirror ``t4_rct.T4`` anchors; prompts follow the
same markdown structure as ``prompts/T4-RCT-image.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from faker import Faker

from ..adapters.base import VariantAdapter
from ..email_render import baseline_email_rgb, render_email_png
from ..sources.mail_base import EmailItem, load_email_bytes, parse_email_message, plain_body_from_item


T4_EML_IMAGE_SIZE: tuple[int, int] = (720, 1400)


@dataclass
class EmlT4Result:
    image: Image.Image
    style_variant: str
    identity_seed_a: int
    identity_seed_b: int
    prompt: str
    response_raw: str
    anchor_ids: tuple[str, ...]
    topic: str
    notes: str = ""


def _pick_anchors(loader: Any, seed: int, n: int = 3) -> list[EmailItem]:
    items = list(loader.iter_items())
    rng = random.Random(seed)
    if len(items) <= n:
        return items
    rng.shuffle(items)
    return items[:n]


def _anchor_ocr_block(anchors: list[EmailItem]) -> str:
    blocks: list[str] = []
    for idx, it in enumerate(anchors):
        if getattr(it, "modality", "rfc822") == "rvl_email_page":
            snippet = plain_body_from_item(it)[:450].strip()
            blocks.append(f"=== Anchor {idx + 1} ({it.doc_id}) ===\n(page OCR excerpt)\n{snippet}")
            continue
        try:
            msg = parse_email_message(load_email_bytes(it.path))
        except Exception:
            continue
        subj = (msg.get("Subject") or "").strip()[:120]
        snippet = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        snippet = payload.decode(errors="replace")[:400]
                    break
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                snippet = payload.decode(errors="replace")[:400]
        blocks.append(f"=== Anchor {idx + 1} ({it.doc_id}) ===\nSubject: {subj}\n{snippet}")
    return "\n\n".join(blocks)


def _load_anchor_images(anchors: list[EmailItem]) -> list[Image.Image]:
    refs: list[Image.Image] = []
    target_w = 680
    for it in anchors:
        try:
            if getattr(it, "modality", "rfc822") == "rvl_email_page":
                img = baseline_email_rgb(it).convert("RGB")
            else:
                msg = parse_email_message(load_email_bytes(it.path))
                img = render_email_png(msg, width=target_w)[0].convert("RGB")
            if img.width != target_w:
                ratio = target_w / img.width
                img = img.resize((target_w, max(1, int(img.height * ratio))), Image.LANCZOS)
            refs.append(img.convert("RGB"))
        except Exception:
            continue
    return refs


def _topic(path: Path, seed: int) -> str:
    if not path.exists():
        return "quarterly compliance checklist"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return "quarterly compliance checklist"
    return lines[seed % len(lines)]


def _participants(seed_a: int, seed_b: int) -> tuple[str, str]:
    fa = Faker()
    fa.seed_instance(seed_a)
    fb = Faker()
    fb.seed_instance(seed_b)
    a = f"{fa.name()} — {fa.job()} — {fa.company()} — {fa.email()}"
    b = f"{fb.name()} — {fb.job()} — {fb.company()} — {fb.email()}"
    return a, b


def apply(
    *,
    adapter: VariantAdapter,
    loader: Any,
    item_index: int,
    batch_seed_value: int,
    prompts_dir: Path,
    assets_dir: Path,
) -> EmlT4Result:
    item_seed = batch_seed_value * 1000 + item_index
    identity_seed_a = item_seed
    identity_seed_b = item_seed + 1
    pa, pb = _participants(identity_seed_a, identity_seed_b)

    styles = ("with_typos", "clean", "quoted_thread", "plain")
    style_variant = styles[item_index % len(styles)]

    anchor_seed = item_seed ^ 0xB7B7B7B7
    anchors = _pick_anchors(loader, anchor_seed)
    ref_images = _load_anchor_images(anchors)
    anchor_block = _anchor_ocr_block(anchors)
    topic = _topic(assets_dir / "email_topics.txt", item_seed)

    template_path = prompts_dir / "T4-EML-image.md"
    template = (
        template_path.read_text(encoding="utf-8")
        if template_path.exists()
        else (
            "Generate one photorealistic screenshot of an email thread UI.\n"
            "Participants:\nA: {participant_a}\nB: {participant_b}\n"
            "Topic: {topic}. Style: {style_variant}.\n"
            "Anchors:\n{anchor_block}\n"
            "Portrait canvas, no JSON."
        )
    )
    prompt = (
        template.replace("{participant_a}", pa)
        .replace("{participant_b}", pb)
        .replace("{topic}", topic)
        .replace("{style_variant}", style_variant)
        .replace("{anchor_block}", anchor_block)
    )

    image = adapter.few_shot_image(ref_images, prompt, item_seed, size=T4_EML_IMAGE_SIZE)
    w, h = image.size
    response_raw = f"few_shot_image {w}x{h}"
    return EmlT4Result(
        image=image,
        style_variant=style_variant,
        identity_seed_a=identity_seed_a,
        identity_seed_b=identity_seed_b,
        prompt=prompt,
        response_raw=response_raw,
        anchor_ids=tuple(a.doc_id for a in anchors),
        topic=topic,
        notes="",
    )
