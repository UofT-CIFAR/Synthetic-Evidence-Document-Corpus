"""X-T3-EMAIL: insert a fabricated reply into a rendered thread pane (spec §6).

Uses ``adapter.text_complete`` with the same markdown prompt pattern as receipt
Tier 3, then a single full-frame vision pass to show an extra message below the
original body while preserving headers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..adapters.base import AdapterCapabilityError, VariantAdapter
from ..email_render import baseline_email_rgb, render_email_png
from ..sources.mail_base import EmailItem, load_email_bytes, parse_email_message, plain_body_from_item
from .common import apply_full_image_inpaint


@dataclass
class EmlT3Result:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    inserted_text: str
    target: str
    impersonation_subject_id: str
    prompt: str
    response_raw: str
    notes: str = ""


def _pick_target(path: Path, seed: int) -> str:
    if not path.exists():
        return "scheduling follow-up"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return "scheduling follow-up"
    return lines[seed % len(lines)]


def _subject_core(subject: str) -> str:
    s = subject.strip()
    while s.lower().startswith("re:"):
        s = s[3:].strip()
    return s


def _inpaint_prompt(inserted_text: str, impersonated_from: str, bbox: tuple[int, int, int, int]) -> str:
    x, y, w, h = bbox
    return (
        "Recreate this email screenshot. Keep the header block and original body "
        "unchanged. Below the existing body, add one additional quoted reply "
        f'formatted like a mail client (monospace), as if From: {impersonated_from} wrote '
        f'a short follow-up that conveys:\n"{inserted_text}"\n'
        f"Place the new block in the lower area (guide region x={x}, y={y}, w={w}, h={h}). "
        "Match font size and margins. Same width and height as the input. "
        "No borders or watermarks."
    )


def _participants_and_subject(item: EmailItem) -> tuple[list[str], str]:
    if getattr(item, "modality", "rfc822") != "rvl_email_page":
        data = load_email_bytes(item.path)
        msg = parse_email_message(data)
        from_addrs = [
            a.strip() for a in (msg.get("From") or "").replace("\n", " ").split(",") if a.strip()
        ]
        to_addrs = [a.strip() for a in (msg.get("To") or "").replace("\n", " ").split(",") if a.strip()]
        participants: list[str] = []
        for a in from_addrs + to_addrs:
            if a and a not in participants:
                participants.append(a)
        return participants, _subject_core(msg.get("Subject", ""))

    blob = plain_body_from_item(item)
    addrs = re.findall(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}", blob)
    participants = []
    for a in addrs:
        if a not in participants:
            participants.append(a)
    participants = participants[:8]
    if not participants:
        participants = ["participant-a@document.local", "participant-b@document.local"]
    head = blob.strip().splitlines()[0][:200] if blob.strip() else "(no subject)"
    return participants, head


def apply(
    item: EmailItem,
    *,
    adapter: VariantAdapter,
    item_index: int,
    seed: int,
    assets_dir: Path,
    prompts_dir: Path,
    image_edit_scope: str = "full_image",
) -> EmlT3Result:
    participants, thread_subject = _participants_and_subject(item)
    if not participants:
        raise ValueError("no participants for impersonation")

    pick = participants[seed % len(participants)]
    impersonation_subject_id = hashlib.sha256(pick.encode("utf-8")).hexdigest()[:24]

    target = _pick_target(assets_dir / "email_targets.txt", seed + item_index)
    template_path = prompts_dir / "T3-EMAIL.md"
    body_excerpt = plain_body_from_item(item)[:4000]
    prior_voice = body_excerpt[:1200]
    template = (
        template_path.read_text(encoding="utf-8")
        if template_path.exists()
        else (
            "Draft one email reply (50-200 words) in the same voice as the participant.\n"
            "Target outcome: {target}\n"
            "Prior excerpt:\n{anchor_excerpt}\n"
            "Return plain text only."
        )
    )
    llm_prompt = (
        template.replace("{target}", target)
        .replace("{anchor_excerpt}", prior_voice)
        .replace("{impersonated_address}", pick)
        .replace("{thread_subject}", thread_subject)
    )

    response_raw = adapter.text_complete(prompt=llm_prompt, seed=seed, max_tokens=400)
    if not (response_raw or "").strip():
        raise AdapterCapabilityError("Tier-3 email requires non-empty text_complete output.")

    inserted = " ".join(response_raw.strip().split())[:1200]

    base = baseline_email_rgb(item).convert("RGB")
    w, h = base.size
    bbox = (int(w * 0.06), int(h * 0.62), int(w * 0.88), int(h * 0.28))
    vision_prompt = _inpaint_prompt(inserted, pick, bbox)

    scope = (image_edit_scope or "full_image").strip().lower()
    if scope != "full_image":
        scope = "full_image"

    image = apply_full_image_inpaint(base, adapter=adapter, prompt=vision_prompt, seed=seed)

    recorded = (
        "### Tier-3 text adapter (draft inserted reply)\n"
        f"{llm_prompt}\n\n"
        "### Tier-3 vision (full-frame clone + inserted reply block)\n"
        f"{vision_prompt}"
    )
    return EmlT3Result(
        image=image.convert("RGB"),
        bbox=bbox,
        inserted_text=inserted,
        target=target,
        impersonation_subject_id=impersonation_subject_id,
        prompt=recorded,
        response_raw=response_raw,
        notes=f"impersonation_subject_id={impersonation_subject_id}",
    )
